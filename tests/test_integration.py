"""Integration test — full DON pipeline end-to-end.

This test simulates the complete lifecycle:
1. Register a domain with quadratic plugins
2. Optimizer runs multiple steps, producing optimae
3. Evaluator verifies each optimae (with and without synthetic data)
4. Validator accepts/rejects based on tolerance
5. Consensus tracks weighted performance increments
6. Block is generated when threshold is met
7. Blockchain validates and stores the block

No network involved — everything runs in-process to verify the logic.
"""

import numpy as np
import pytest

from doin_core.consensus import ProofOfOptimization
from doin_core.models.block import Block
from doin_core.models.domain import Domain, DomainConfig
from doin_core.models.optimae import Optimae

from doin_node.blockchain.chain import Chain
from doin_node.validation.validator import OptimaeValidator

from doin_plugins.quadratic_optimizer import QuadraticOptimizer
from doin_plugins.quadratic_inferencer import QuadraticInferencer
from doin_plugins.quadratic_synthetic import QuadraticSyntheticData


# Shared target for all plugins
TARGET = [1.0, 2.0, 3.0, 4.0, 5.0]
PLUGIN_CONFIG = {"n_params": 5, "target": TARGET, "step_size": 1.0, "seed": 42, "noise_std": 0.05}


def make_domain() -> Domain:
    return Domain(
        id="quadratic-v1",
        name="Quadratic Reference Domain",
        performance_metric="neg_mse",
        higher_is_better=True,
        weight=1.0,
        config=DomainConfig(
            optimization_plugin="simple_quadratic",
            inference_plugin="simple_quadratic",
            synthetic_data_plugin="simple_quadratic",
            plugin_config=PLUGIN_CONFIG,
        ),
    )


class TestFullPipeline:
    """End-to-end test of the entire DON pipeline."""

    def test_optimizer_produces_optimae(self) -> None:
        """Optimizer should produce improving optimae over time."""
        optimizer = QuadraticOptimizer()
        optimizer.configure(PLUGIN_CONFIG)

        params, perf = optimizer.optimize(None, None)
        improvements = 0

        for _ in range(100):
            new_params, new_perf = optimizer.optimize(params, perf)
            if new_perf > perf:
                params, perf = new_params, new_perf
                improvements += 1

        assert improvements > 5, f"Expected multiple improvements, got {improvements}"
        assert perf > -50.0, f"Performance should improve significantly, got {perf}"

    def test_evaluator_verifies_optimizer_results(self) -> None:
        """Evaluator should verify optimizer's reported performance."""
        optimizer = QuadraticOptimizer()
        optimizer.configure(PLUGIN_CONFIG)

        inferencer = QuadraticInferencer()
        inferencer.configure(PLUGIN_CONFIG)

        # Optimizer produces result
        params, reported_perf = optimizer.optimize(None, None)

        # Evaluator verifies
        verified_perf = inferencer.evaluate(params)

        # Should be very close (same target, same params)
        assert abs(reported_perf - verified_perf) < 0.01

    def test_evaluator_verifies_with_synthetic_data(self) -> None:
        """Evaluator should verify using synthetic data (slightly different results)."""
        optimizer = QuadraticOptimizer()
        optimizer.configure(PLUGIN_CONFIG)

        inferencer = QuadraticInferencer()
        inferencer.configure(PLUGIN_CONFIG)

        synth = QuadraticSyntheticData()
        synth.configure(PLUGIN_CONFIG)

        # Optimizer produces result
        params, reported_perf = optimizer.optimize(None, None)

        # Evaluator verifies with synthetic data
        synthetic_data = synth.generate(seed=99)
        verified_perf = inferencer.evaluate(params, data=synthetic_data)

        # Should be close but not identical due to noise
        # With noise_std=0.05, difference should be small
        assert abs(reported_perf - verified_perf) < 5.0

    def test_validator_accepts_valid_optimae(self) -> None:
        """Validator should accept optimae with matching performance."""
        domain = make_domain()
        validator = OptimaeValidator(tolerance=1.0)  # Generous tolerance for noisy verification
        validator.register_domain(domain)

        optimizer = QuadraticOptimizer()
        optimizer.configure(PLUGIN_CONFIG)

        inferencer = QuadraticInferencer()
        inferencer.configure(PLUGIN_CONFIG)

        params, reported_perf = optimizer.optimize(None, None)

        optimae = Optimae(
            domain_id="quadratic-v1",
            optimizer_id="test-optimizer",
            parameters=params,
            reported_performance=reported_perf,
        )

        validator.submit_for_validation(optimae)
        verified_perf = inferencer.evaluate(params)
        result = validator.record_evaluation(optimae.id, verified_perf)

        assert result.is_valid, f"Should be valid: reported={reported_perf}, verified={verified_perf}"

    def test_full_loop_block_generation(self) -> None:
        """Full loop: optimize → verify → accept → consensus → block."""
        domain = make_domain()

        # Initialize components
        chain = Chain()
        chain.initialize("test-node")

        consensus = ProofOfOptimization(
            target_block_time=600.0,
            initial_threshold=0.5,  # Low threshold for testing
        )
        consensus.register_domain(domain)

        validator = OptimaeValidator(tolerance=1.0)
        validator.register_domain(domain)

        optimizer = QuadraticOptimizer()
        optimizer.configure(PLUGIN_CONFIG)

        inferencer = QuadraticInferencer()
        inferencer.configure(PLUGIN_CONFIG)

        # Run optimization loop until block is generated
        best_params = None
        best_perf = None
        block_generated = False
        total_steps = 0
        accepted_optimae = 0

        for step in range(500):
            total_steps += 1

            # Optimizer step
            new_params, new_perf = optimizer.optimize(best_params, best_perf)
            if best_perf is not None and new_perf <= best_perf:
                continue  # No improvement

            # Create optimae
            increment = abs(new_perf - best_perf) if best_perf is not None else abs(new_perf)
            optimae = Optimae(
                domain_id="quadratic-v1",
                optimizer_id="test-optimizer",
                parameters=new_params,
                reported_performance=new_perf,
                performance_increment=increment,
            )

            # Evaluator verifies
            verified_perf = inferencer.evaluate(new_params)

            # Validator checks
            validator.submit_for_validation(optimae)
            result = validator.record_evaluation(optimae.id, verified_perf)

            if result.is_valid:
                accepted_optimae += 1
                optimae.verified_performance = verified_perf
                consensus.record_optimae(optimae)
                best_params = new_params
                best_perf = new_perf

                # Try to generate block
                if consensus.can_generate_block():
                    block = consensus.generate_block(chain.tip, "test-node")  # type: ignore
                    assert block is not None
                    chain.append_block(block)
                    block_generated = True
                    break

        # Assertions
        assert block_generated, (
            f"Block should have been generated after {total_steps} steps "
            f"({accepted_optimae} accepted optimae, weighted_sum={consensus.weighted_sum})"
        )
        assert chain.height == 2  # Genesis + 1 block
        assert chain.tip is not None
        assert chain.tip.header.index == 1
        assert len(chain.tip.transactions) > 0

        print(f"\n✅ Full pipeline passed!")
        print(f"   Steps: {total_steps}")
        print(f"   Accepted optimae: {accepted_optimae}")
        print(f"   Final performance: {best_perf:.6f}")
        print(f"   Block transactions: {len(chain.tip.transactions)}")
        print(f"   Block hash: {chain.tip.hash[:16]}...")

    def test_multi_block_chain_growth(self) -> None:
        """Run long enough to generate multiple blocks."""
        domain = make_domain()

        chain = Chain()
        chain.initialize("test-node")

        consensus = ProofOfOptimization(
            target_block_time=600.0,
            initial_threshold=0.3,  # Very low for fast blocks
        )
        consensus.register_domain(domain)

        validator = OptimaeValidator(tolerance=2.0)
        validator.register_domain(domain)

        optimizer = QuadraticOptimizer()
        optimizer.configure({**PLUGIN_CONFIG, "step_size": 2.0})

        inferencer = QuadraticInferencer()
        inferencer.configure(PLUGIN_CONFIG)

        best_params = None
        best_perf = None
        blocks_generated = 0

        for _ in range(1000):
            new_params, new_perf = optimizer.optimize(best_params, best_perf)
            if best_perf is not None and new_perf <= best_perf:
                continue

            increment = abs(new_perf - best_perf) if best_perf is not None else abs(new_perf)
            optimae = Optimae(
                domain_id="quadratic-v1",
                optimizer_id="test-optimizer",
                parameters=new_params,
                reported_performance=new_perf,
                performance_increment=increment,
            )

            verified_perf = inferencer.evaluate(new_params)
            validator.submit_for_validation(optimae)
            result = validator.record_evaluation(optimae.id, verified_perf)

            if result.is_valid:
                optimae.verified_performance = verified_perf
                consensus.record_optimae(optimae)
                best_params = new_params
                best_perf = new_perf

                if consensus.can_generate_block():
                    block = consensus.generate_block(chain.tip, "test-node")  # type: ignore
                    if block:
                        chain.append_block(block)
                        blocks_generated += 1
                        if blocks_generated >= 3:
                            break

        assert blocks_generated >= 3, f"Expected ≥3 blocks, got {blocks_generated}"
        assert chain.height >= 4  # Genesis + 3

        print(f"\n✅ Multi-block test passed!")
        print(f"   Chain height: {chain.height}")
        print(f"   Final performance: {best_perf:.6f}")
        for i in range(chain.height):
            b = chain.get_block(i)
            assert b is not None
            print(f"   Block #{i}: hash={b.hash[:16]}... txs={len(b.transactions)}")
