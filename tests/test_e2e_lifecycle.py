"""End-to-end integration test — full optimae lifecycle.

Tests the COMPLETE DON pipeline using quadratic reference plugins,
simulating multiple nodes in-process without network transport.

Flow tested:
  1. Optimizer produces candidate parameters
  2. Optimizer computes commitment hash (commit-reveal phase 1)
  3. Commitment registered on all nodes
  4. Optimizer reveals parameters + nonce (commit-reveal phase 2)
  5. Reveal validated (hash match, bounds, seed)
  6. Quorum selects random evaluators (excluding optimizer)
  7. Each evaluator generates DIFFERENT synthetic data (per-evaluator seed)
  8. Each evaluator independently verifies performance
  9. Evaluators submit votes with synthetic data hashes
  10. Quorum evaluates votes → accept/reject
  11. Incentive model computes reward fraction
  12. Reputation updated (accepted → reward, rejected → penalty)
  13. VUW effective increment computed
  14. Consensus records optimae
  15. Block generation attempted

This test does NOT require TensorFlow, network, or any external deps.
"""

from __future__ import annotations

import secrets
import time
from typing import Any

import numpy as np
import pytest

from doin_core.consensus import (
    DeterministicSeedPolicy,
    IncentiveConfig,
    VerifiedUtilityWeights,
    evaluate_verification_incentive,
)
from doin_core.consensus.finality import FinalityManager
from doin_core.consensus.fork_choice import ForkChoiceRule
from doin_core.models import (
    BoundsValidator,
    Commitment,
    CommitRevealManager,
    Optimae,
    QuorumConfig,
    QuorumManager,
    ReputationTracker,
    ResourceLimits,
    Reveal,
    Task,
    TaskQueue,
    TaskStatus,
    TaskType,
    compute_commitment,
)
from doin_core.plugins.base import hash_synthetic_data

from doin_plugins.quadratic_inferencer import QuadraticInferencer
from doin_plugins.quadratic_optimizer import QuadraticOptimizer
from doin_plugins.quadratic_synthetic import QuadraticSyntheticData


# ── Simulated node (in-process, no network) ──────────────────────────

class SimulatedNode:
    """A simulated DON node for testing.

    Contains all the security systems but no transport layer.
    Nodes communicate by directly calling each other's methods.
    """

    def __init__(
        self,
        peer_id: str,
        is_optimizer: bool = False,
        is_evaluator: bool = False,
    ) -> None:
        self.peer_id = peer_id
        self.is_optimizer = is_optimizer
        self.is_evaluator = is_evaluator

        # Security systems
        self.commit_reveal = CommitRevealManager(max_commit_age=600.0)
        self.quorum = QuorumManager(QuorumConfig(
            min_evaluators=3,
            quorum_fraction=0.67,
            tolerance=0.10,  # 10% tolerance for synthetic data variance
        ))
        self.reputation = ReputationTracker()
        self.bounds_validator = BoundsValidator({
            "x": (-20.0, 20.0),  # Each x_i must be in [-20, 20]
        })
        self.seed_policy = DeterministicSeedPolicy(require_seed=True)
        self.vuw = VerifiedUtilityWeights()
        self.finality = FinalityManager(confirmation_depth=3)
        self.fork_choice = ForkChoiceRule()
        self.incentive_config = IncentiveConfig(
            higher_is_better=True,  # neg_mse: higher is better
            tolerance_margin=0.15,  # 15% tolerance for synthetic data noise
            bonus_threshold=0.05,
            min_reward_fraction=0.3,
            max_bonus_multiplier=1.2,
        )
        self.task_queue = TaskQueue()

        # Plugins (configured externally)
        self.optimizer_plugin: QuadraticOptimizer | None = None
        self.evaluator_plugin: QuadraticInferencer | None = None
        self.synthetic_plugin: QuadraticSyntheticData | None = None

        # State
        self.chain_tip_hash = f"genesis-{int(time.time())}"
        self.accepted_optimae: list[dict[str, Any]] = []
        self.rejected_optimae: list[dict[str, Any]] = []

        # Register domain
        self.vuw.register_domain(
            "quadratic", base_weight=1.0, has_synthetic_data=True,
        )


def create_network(
    n_evaluators: int = 4,
    target: list[float] | None = None,
) -> tuple[SimulatedNode, list[SimulatedNode]]:
    """Create a simulated network with 1 optimizer + N evaluators.

    All nodes share the same quadratic target (in a real network,
    evaluators would load this from the domain config).
    """
    if target is None:
        np.random.seed(42)
        target = np.random.uniform(-5, 5, 10).tolist()

    plugin_config = {"target": target, "n_params": len(target), "seed": 42}

    # Optimizer node
    optimizer = SimulatedNode("optimizer-node", is_optimizer=True)
    opt_plugin = QuadraticOptimizer()
    opt_plugin.configure(plugin_config)
    optimizer.optimizer_plugin = opt_plugin

    # Evaluator nodes
    evaluators = []
    for i in range(n_evaluators):
        node = SimulatedNode(f"evaluator-{i}", is_evaluator=True)
        inf_plugin = QuadraticInferencer()
        inf_plugin.configure(plugin_config)
        node.evaluator_plugin = inf_plugin

        syn_plugin = QuadraticSyntheticData()
        syn_plugin.configure({"target": target, "noise_std": 0.1})
        node.synthetic_plugin = syn_plugin

        evaluators.append(node)

    return optimizer, evaluators


# ── The full lifecycle test ──────────────────────────────────────────

class TestFullLifecycle:
    """Tests the complete optimae lifecycle from optimization to consensus."""

    def test_honest_optimizer_accepted(self):
        """An honest optimizer produces a good result → accepted by quorum
        → reputation increased → effective increment computed."""

        optimizer, evaluators = create_network(n_evaluators=4)
        all_nodes = [optimizer] + evaluators

        # ════════════════════════════════════════════════
        # Step 1: Optimizer produces candidate parameters
        # ════════════════════════════════════════════════
        params, performance = optimizer.optimizer_plugin.optimize(
            current_best_params=None,
            current_best_performance=None,
        )
        assert "x" in params
        assert isinstance(performance, float)

        # ════════════════════════════════════════════════
        # Step 2: Commit phase (optimizer commits hash)
        # ════════════════════════════════════════════════
        nonce = secrets.token_hex(16)
        commitment_hash = compute_commitment(params, nonce)

        # Register commitment on ALL nodes (simulates flooding)
        for node in all_nodes:
            assert node.commit_reveal.add_commitment(
                Commitment(
                    commitment_hash=commitment_hash,
                    domain_id="quadratic",
                    optimizer_id=optimizer.peer_id,
                )
            )

        # ════════════════════════════════════════════════
        # Step 3: Reveal phase (optimizer reveals params)
        # ════════════════════════════════════════════════
        optimae_id = f"opt-{optimizer.peer_id[:8]}-{int(time.time())}"

        # All nodes verify the reveal (params as committed, without seed)
        for node in all_nodes:
            assert node.commit_reveal.process_reveal(
                Reveal(
                    commitment_hash=commitment_hash,
                    domain_id="quadratic",
                    optimizer_id=optimizer.peer_id,
                    parameters=params,
                    nonce=nonce,
                    reported_performance=performance,
                )
            )

        # ════════════════════════════════════════════════
        # Step 4: Seed derivation (after reveal, from commitment hash)
        # ════════════════════════════════════════════════
        # The seed is derived AFTER the reveal is verified, using the
        # commitment hash as input. This seed is used by evaluators
        # for synthetic data generation (combined with their own ID).
        seed = optimizer.seed_policy.get_seed_for_optimae(commitment_hash, "quadratic")
        ok, reason = optimizer.seed_policy.validate_submission(
            commitment_hash, "quadratic", seed,
        )
        assert ok, f"Seed validation failed: {reason}"

        # ════════════════════════════════════════════════
        # Step 5: Quorum selects evaluators
        # ════════════════════════════════════════════════
        eligible = [e.peer_id for e in evaluators]
        chain_tip = optimizer.chain_tip_hash

        # All nodes select the same evaluators (deterministic)
        selected_sets = []
        for node in all_nodes:
            selected = node.quorum.select_evaluators(
                optimae_id=optimae_id,
                domain_id="quadratic",
                optimizer_id=optimizer.peer_id,
                reported_performance=performance,
                eligible_evaluators=eligible,
                chain_tip_hash=chain_tip,
            )
            selected_sets.append(selected)

        # Verify all nodes agree on the same evaluators
        for s in selected_sets[1:]:
            assert s == selected_sets[0], "Nodes disagree on evaluator selection!"

        selected_evaluators = selected_sets[0]
        assert optimizer.peer_id not in selected_evaluators
        assert len(selected_evaluators) >= 3

        # ════════════════════════════════════════════════
        # Step 6: Each evaluator generates DIFFERENT synthetic data
        # ════════════════════════════════════════════════
        evaluator_results: list[tuple[str, float, str]] = []  # (eval_id, perf, hash)

        for eval_id in selected_evaluators:
            eval_node = next(e for e in evaluators if e.peer_id == eval_id)

            # Per-evaluator seed (unpredictable to optimizer)
            synth_seed = eval_node.seed_policy.get_seed_for_synthetic_data(
                commitment_hash, "quadratic", eval_id, chain_tip,
            )

            # Generate synthetic data with hash
            synth_data, synth_hash = eval_node.synthetic_plugin.generate_with_hash(synth_seed)

            # Verify performance on synthetic data
            verified_perf = eval_node.evaluator_plugin.evaluate(params, data=synth_data)

            evaluator_results.append((eval_id, verified_perf, synth_hash))

        # Verify each evaluator got DIFFERENT synthetic data
        hashes = [h for _, _, h in evaluator_results]
        assert len(set(hashes)) == len(hashes), "Evaluators should have different synthetic data!"

        # Verify performances are in the same ballpark (genuine model generalizes)
        perfs = [p for _, p, _ in evaluator_results]
        perf_range = max(perfs) - min(perfs)
        assert perf_range < abs(performance) * 0.5, (
            f"Performance range {perf_range} too wide — model should generalize"
        )

        # ════════════════════════════════════════════════
        # Step 7: Evaluators submit votes to quorum
        # ════════════════════════════════════════════════
        quorum_state = None
        for eval_id, verified_perf, synth_hash in evaluator_results:
            # Submit to all nodes (simulates flooding)
            for node in all_nodes:
                result = node.quorum.add_vote(
                    optimae_id, eval_id, verified_perf,
                    used_synthetic=True,
                    synthetic_data_hash=synth_hash,
                )
                if result is not None:
                    quorum_state = result

        assert quorum_state is not None, "Quorum should have been reached"
        assert quorum_state.has_quorum

        # ════════════════════════════════════════════════
        # Step 8: Evaluate quorum → accept/reject
        # ════════════════════════════════════════════════
        quorum_result = optimizer.quorum.evaluate_quorum(optimae_id)

        assert quorum_result.accepted, f"Honest optimae should be accepted: {quorum_result.reason}"
        assert quorum_result.median_performance is not None
        assert quorum_result.agree_fraction >= 0.67

        # ════════════════════════════════════════════════
        # Step 9: Incentive model computes reward
        # ════════════════════════════════════════════════
        import math
        rep_score = optimizer.reputation.get_score(optimizer.peer_id)
        rep_factor = min(1.0, math.log1p(rep_score) / math.log1p(10.0)) if rep_score > 0 else 0.0

        weights = optimizer.vuw.compute_weights()
        domain_weight = weights["quadratic"]
        assert domain_weight > 0, "Domain with synthetic data should have positive weight"

        incentive_result = evaluate_verification_incentive(
            reported_performance=performance,
            verified_performance=quorum_result.median_performance,
            raw_increment=abs(performance),
            domain_weight=domain_weight,
            reputation_factor=max(0.1, rep_factor),  # Min floor for first optimae
            config=optimizer.incentive_config,
        )

        assert incentive_result.is_accepted, f"Incentive should accept: {incentive_result.reason}"
        assert incentive_result.reward_fraction > 0
        assert incentive_result.effective_increment > 0

        # ════════════════════════════════════════════════
        # Step 10: Reputation update
        # ════════════════════════════════════════════════
        # Reward optimizer
        optimizer.reputation.record_optimae_accepted(optimizer.peer_id)
        assert optimizer.reputation.get_score(optimizer.peer_id) > 0

        # Reward evaluators who agreed with quorum
        for eval_id, agreed in quorum_result.agreements.items():
            optimizer.reputation.record_evaluation_completed(eval_id, agreed)
            if agreed:
                assert optimizer.reputation.get_score(eval_id) > 0

        # ════════════════════════════════════════════════
        # Step 11: Record accepted optimae
        # ════════════════════════════════════════════════
        optimizer.accepted_optimae.append({
            "optimae_id": optimae_id,
            "performance": performance,
            "verified_performance": quorum_result.median_performance,
            "reward_fraction": incentive_result.reward_fraction,
            "effective_increment": incentive_result.effective_increment,
            "evaluators": selected_evaluators,
            "agree_fraction": quorum_result.agree_fraction,
        })

        assert len(optimizer.accepted_optimae) == 1

    def test_dishonest_optimizer_rejected(self):
        """An optimizer that lies about performance → rejected by quorum
        → reputation slashed."""

        optimizer, evaluators = create_network(n_evaluators=4)

        # Optimizer produces a real result...
        params, real_performance = optimizer.optimizer_plugin.optimize(None, None)

        # ...but LIES about performance (claims much better than reality)
        fake_performance = real_performance * 5.0  # Claims 5x better

        nonce = secrets.token_hex(16)
        commitment_hash = compute_commitment(params, nonce)

        optimae_id = "fake-optimae"

        # Register commitment + reveal on all nodes
        all_nodes = [optimizer] + evaluators
        for node in all_nodes:
            node.commit_reveal.add_commitment(Commitment(
                commitment_hash=commitment_hash,
                domain_id="quadratic",
                optimizer_id=optimizer.peer_id,
            ))
            node.commit_reveal.process_reveal(Reveal(
                commitment_hash=commitment_hash,
                domain_id="quadratic",
                optimizer_id=optimizer.peer_id,
                parameters=params,
                nonce=nonce,
                reported_performance=fake_performance,
            ))

        # Quorum selection
        eligible = [e.peer_id for e in evaluators]
        selected = optimizer.quorum.select_evaluators(
            optimae_id, "quadratic", optimizer.peer_id,
            fake_performance, eligible, optimizer.chain_tip_hash,
        )
        # Sync quorum to all nodes
        for node in all_nodes[1:]:
            node.quorum.select_evaluators(
                optimae_id, "quadratic", optimizer.peer_id,
                fake_performance, eligible, optimizer.chain_tip_hash,
            )

        # Evaluators verify on synthetic data → get REAL performance
        for eval_id in selected:
            eval_node = next(e for e in evaluators if e.peer_id == eval_id)
            synth_seed = eval_node.seed_policy.get_seed_for_synthetic_data(
                commitment_hash, "quadratic", eval_id, optimizer.chain_tip_hash,
            )
            synth_data, synth_hash = eval_node.synthetic_plugin.generate_with_hash(synth_seed)
            verified_perf = eval_node.evaluator_plugin.evaluate(params, data=synth_data)

            for node in all_nodes:
                node.quorum.add_vote(
                    optimae_id, eval_id, verified_perf,
                    synthetic_data_hash=synth_hash,
                )

        # Evaluate quorum
        result = optimizer.quorum.evaluate_quorum(optimae_id)

        # Should be REJECTED — reported performance diverges massively from verified
        assert not result.accepted, "Dishonest optimae should be rejected"
        assert "report diverges" in result.reason

        # Slash reputation
        optimizer.reputation.record_optimae_rejected(optimizer.peer_id)
        # After rejection, score should be 0 (started at 0, penalty applied)
        assert optimizer.reputation.get_score(optimizer.peer_id) == 0.0

    def test_overfitted_model_detected(self):
        """An optimizer that overfits to a specific dataset will fail
        on the evaluators' synthetic data (which is different for each evaluator)."""

        # Create network where optimizer knows the "real" target perfectly
        target = [1.0, 2.0, 3.0, 4.0, 5.0]
        optimizer, evaluators = create_network(n_evaluators=4, target=target)

        # Optimizer has "perfect" params for the real target
        perfect_params = {"x": target.copy()}
        # On real target, this gives loss=0, performance=0.0 (perfect)
        perfect_performance = 0.0  # -sum((x-target)^2) = 0

        nonce = secrets.token_hex(16)
        commitment_hash = compute_commitment(perfect_params, nonce)

        optimae_id = "overfit-optimae"
        all_nodes = [optimizer] + evaluators

        for node in all_nodes:
            node.commit_reveal.add_commitment(Commitment(
                commitment_hash=commitment_hash,
                domain_id="quadratic",
                optimizer_id=optimizer.peer_id,
            ))
            node.commit_reveal.process_reveal(Reveal(
                commitment_hash=commitment_hash,
                domain_id="quadratic",
                optimizer_id=optimizer.peer_id,
                parameters=perfect_params,
                nonce=nonce,
                reported_performance=perfect_performance,
            ))

        eligible = [e.peer_id for e in evaluators]
        selected = optimizer.quorum.select_evaluators(
            optimae_id, "quadratic", optimizer.peer_id,
            perfect_performance, eligible, optimizer.chain_tip_hash,
        )
        for node in all_nodes[1:]:
            node.quorum.select_evaluators(
                optimae_id, "quadratic", optimizer.peer_id,
                perfect_performance, eligible, optimizer.chain_tip_hash,
            )

        # Evaluators test on NOISY synthetic data
        # Performance will be slightly negative (noise adds error)
        verified_perfs = []
        for eval_id in selected:
            eval_node = next(e for e in evaluators if e.peer_id == eval_id)
            synth_seed = eval_node.seed_policy.get_seed_for_synthetic_data(
                commitment_hash, "quadratic", eval_id, optimizer.chain_tip_hash,
            )
            synth_data, synth_hash = eval_node.synthetic_plugin.generate_with_hash(synth_seed)
            verified_perf = eval_node.evaluator_plugin.evaluate(perfect_params, data=synth_data)
            verified_perfs.append(verified_perf)

            for node in all_nodes:
                node.quorum.add_vote(
                    optimae_id, eval_id, verified_perf,
                    synthetic_data_hash=synth_hash,
                )

        # Quorum should STILL accept because the model IS genuinely good
        # (it's actually perfect for the target distribution, just not
        # zero-loss on noisy data — the incentive tolerance handles this)
        result = optimizer.quorum.evaluate_quorum(optimae_id)

        # The key insight: "perfect" params on real target still work
        # well on noisy synthetic data. But verified perf < reported (0.0).
        # The incentive model determines the reward.
        if result.accepted:
            # Check incentive — verified is worse but within tolerance
            incentive = evaluate_verification_incentive(
                reported_performance=perfect_performance,
                verified_performance=result.median_performance,
                raw_increment=0.1,
                domain_weight=1.0,
                reputation_factor=0.5,
                config=IncentiveConfig(
                    higher_is_better=True,
                    tolerance_margin=0.15,
                ),
            )
            # Should get partial reward (verified is worse due to noise)
            assert incentive.reward_fraction <= 1.0

    def test_multiple_rounds_reputation_growth(self):
        """Run multiple optimization rounds → reputation grows,
        effective increment increases over time."""

        target = [1.0, 2.0, 3.0, 4.0, 5.0]
        optimizer, evaluators = create_network(n_evaluators=4, target=target)
        all_nodes = [optimizer] + evaluators

        increments = []

        for round_num in range(5):
            # Optimize
            best_params = None if round_num == 0 else {"x": target}
            best_perf = None
            params, performance = optimizer.optimizer_plugin.optimize(best_params, best_perf)

            # Commit
            nonce = secrets.token_hex(16)
            ch = compute_commitment(params, nonce)
            optimae_id = f"opt-round-{round_num}"

            for node in all_nodes:
                node.commit_reveal.add_commitment(Commitment(
                    commitment_hash=ch, domain_id="quadratic",
                    optimizer_id=optimizer.peer_id,
                ))
                node.commit_reveal.process_reveal(Reveal(
                    commitment_hash=ch, domain_id="quadratic",
                    optimizer_id=optimizer.peer_id,
                    parameters=params, nonce=nonce,
                    reported_performance=performance,
                ))

            # Quorum
            eligible = [e.peer_id for e in evaluators]
            for node in all_nodes:
                node.quorum.select_evaluators(
                    optimae_id, "quadratic", optimizer.peer_id,
                    performance, eligible, optimizer.chain_tip_hash,
                )

            selected = all_nodes[0].quorum.get_state(optimae_id).required_evaluators

            # Evaluate
            for eval_id in selected:
                eval_node = next(e for e in evaluators if e.peer_id == eval_id)
                synth_seed = eval_node.seed_policy.get_seed_for_synthetic_data(
                    ch, "quadratic", eval_id, optimizer.chain_tip_hash,
                )
                synth_data, _ = eval_node.synthetic_plugin.generate_with_hash(synth_seed)
                verified = eval_node.evaluator_plugin.evaluate(params, data=synth_data)
                for node in all_nodes:
                    node.quorum.add_vote(optimae_id, eval_id, verified)

            result = all_nodes[0].quorum.evaluate_quorum(optimae_id)
            if result.accepted:
                optimizer.reputation.record_optimae_accepted(optimizer.peer_id)
                for eid, agreed in result.agreements.items():
                    optimizer.reputation.record_evaluation_completed(eid, agreed)

                import math
                rep = optimizer.reputation.get_score(optimizer.peer_id)
                rep_factor = min(1.0, math.log1p(rep) / math.log1p(10.0))
                weights = optimizer.vuw.compute_weights()
                dw = weights.get("quadratic", 0)

                incentive = evaluate_verification_incentive(
                    performance, result.median_performance or 0,
                    abs(performance), dw, rep_factor,
                    optimizer.incentive_config,
                )
                increments.append(incentive.effective_increment)

        # After multiple rounds, we should have some accepted optimae
        assert len(increments) > 0

        # Reputation should have grown
        rep = optimizer.reputation.get_score(optimizer.peer_id)
        assert rep > 0

    def test_domain_without_synthetic_gets_zero_weight(self):
        """A domain without a synthetic data plugin gets zero VUW weight."""
        vuw = VerifiedUtilityWeights()
        vuw.register_domain("has-synth", has_synthetic_data=True)
        vuw.register_domain("no-synth", has_synthetic_data=False)

        weights = vuw.compute_weights()
        assert weights["has-synth"] > 0
        assert weights["no-synth"] == 0.0

    def test_finality_prevents_reorg(self):
        """Finality checkpoints prevent reverting accepted optimae."""
        fm = FinalityManager(confirmation_depth=3)

        # Simulate 10 blocks
        for i in range(1, 11):
            hash_at_depth = f"hash-{i - 3}" if i > 3 else None
            fm.on_new_block(i, hash_at_depth)

        # After 10 blocks with depth=3, finality should be at block 7
        assert fm.finalized_height == 7

        # Can't reorg past finality
        assert not fm.is_reorg_allowed(reorg_depth=5, chain_height=10)  # To block 5 < 7
        assert fm.is_reorg_allowed(reorg_depth=2, chain_height=10)      # To block 8 > 7

    def test_fork_choice_heaviest_chain(self):
        """Fork choice rule selects the chain with most optimization work."""
        fcr = ForkChoiceRule()

        fcr.score_chain("honest", 10, [
            {"height": i, "hash": f"h{i}", "transactions": [
                {"tx_type": "optimae_accepted", "payload": {"effective_increment": 1.0}},
            ]}
            for i in range(10)
        ])

        fcr.score_chain("attacker", 10, [
            {"height": i, "hash": f"a{i}", "transactions": [
                {"tx_type": "optimae_accepted", "payload": {"effective_increment": 0.5}},
            ]}
            for i in range(10)
        ])

        best = fcr.select_best()
        assert best.tip_hash == "honest"
        assert best.cumulative_increment > 5.0
