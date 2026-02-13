"""Network integration test — full DON pipeline over HTTP.

Spins up a real node, evaluator, and optimizer communicating over
HTTP transport. Tests the actual network path end-to-end.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

import pytest
from aiohttp import ClientSession, ClientTimeout

from doin_core.models.domain import Domain, DomainConfig
from doin_core.models.optimae import Optimae
from doin_core.protocol.messages import Message, MessageType, OptimaeAnnouncement

from doin_node.node import Node, NodeConfig
from doin_evaluator.service import EvaluatorService, EvaluatorConfig

from doin_plugins.quadratic_optimizer import QuadraticOptimizer
from doin_plugins.quadratic_inferencer import QuadraticInferencer
from doin_plugins.quadratic_synthetic import QuadraticSyntheticData

logger = logging.getLogger(__name__)

TARGET = [1.0, 2.0, 3.0, 4.0, 5.0]
PLUGIN_CONFIG = {"n_params": 5, "target": TARGET, "step_size": 1.0, "seed": 42, "noise_std": 0.05}


def make_domain() -> Domain:
    return Domain(
        id="quadratic-v1",
        name="Quadratic Reference",
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


@pytest.fixture
async def node():
    """Start a DON node on port 18470."""
    config = NodeConfig(
        host="127.0.0.1",
        port=18470,
        data_dir="/tmp/don-test-node",
        target_block_time=600.0,
        initial_threshold=0.5,
    )
    n = Node(config)
    n.register_domain(make_domain())
    await n.start()
    yield n
    await n.stop()


@pytest.fixture
async def evaluator():
    """Start a DON evaluator on port 18471."""
    config = EvaluatorConfig(
        host="127.0.0.1",
        port=18471,
        node_endpoint="127.0.0.1:18470",
    )
    svc = EvaluatorService(config)
    domain = make_domain()

    inf = QuadraticInferencer()
    inf.configure(PLUGIN_CONFIG)
    synth = QuadraticSyntheticData()
    synth.configure(PLUGIN_CONFIG)
    svc.set_domain_plugins("quadratic-v1", domain, inf, synth)

    await svc.start()
    yield svc
    await svc.stop()


class TestNetworkIntegration:
    """Tests with actual HTTP connections between components."""

    @pytest.mark.asyncio
    async def test_node_health(self, node: Node) -> None:
        """Node should respond to health checks."""
        async with ClientSession(timeout=ClientTimeout(total=5)) as session:
            async with session.get("http://127.0.0.1:18470/health") as resp:
                assert resp.status == 200
                data = await resp.json()
                assert data["status"] == "healthy"

    @pytest.mark.asyncio
    async def test_evaluator_health(self, evaluator: EvaluatorService) -> None:
        """Evaluator should respond to health checks."""
        async with ClientSession(timeout=ClientTimeout(total=5)) as session:
            async with session.get("http://127.0.0.1:18471/health") as resp:
                assert resp.status == 200
                data = await resp.json()
                assert data["status"] == "healthy"
                assert "quadratic-v1" in data["domains"]

    @pytest.mark.asyncio
    async def test_evaluator_domains(self, evaluator: EvaluatorService) -> None:
        """Evaluator should list registered domains."""
        async with ClientSession(timeout=ClientTimeout(total=5)) as session:
            async with session.get("http://127.0.0.1:18471/domains") as resp:
                assert resp.status == 200
                data = await resp.json()
                assert len(data["domains"]) == 1
                assert data["domains"][0]["id"] == "quadratic-v1"
                assert data["domains"][0]["has_synthetic_data"] is True

    @pytest.mark.asyncio
    async def test_evaluator_verify_via_http(self, evaluator: EvaluatorService) -> None:
        """Evaluator should verify parameters via HTTP API."""
        async with ClientSession(timeout=ClientTimeout(total=5)) as session:
            payload = {
                "domain_id": "quadratic-v1",
                "parameters": {"x": TARGET},
                "use_synthetic": False,
            }
            async with session.post(
                "http://127.0.0.1:18471/verify",
                json=payload,
            ) as resp:
                assert resp.status == 200
                data = await resp.json()
                assert "verified_performance" in data
                # Exact match with target should be ~0
                assert data["verified_performance"] == pytest.approx(0.0, abs=0.01)

    @pytest.mark.asyncio
    async def test_evaluator_infer_via_http(self, evaluator: EvaluatorService) -> None:
        """Evaluator should serve inference requests once optimae is set."""
        # First set an optimae
        optimae = Optimae(
            domain_id="quadratic-v1",
            optimizer_id="test",
            parameters={"x": [1.1, 2.1, 3.1, 4.1, 5.1]},
            reported_performance=-0.05,
        )
        evaluator.update_optimae("quadratic-v1", optimae)

        async with ClientSession(timeout=ClientTimeout(total=5)) as session:
            payload = {
                "domain_id": "quadratic-v1",
                "input_data": {},
            }
            async with session.post(
                "http://127.0.0.1:18471/infer",
                json=payload,
            ) as resp:
                assert resp.status == 200
                data = await resp.json()
                assert data["domain_id"] == "quadratic-v1"
                assert data["optimae_id"] == optimae.id

    @pytest.mark.asyncio
    async def test_submit_optimae_to_node(self, node: Node) -> None:
        """Optimizer should be able to submit optimae to node via HTTP."""
        optimizer = QuadraticOptimizer()
        optimizer.configure(PLUGIN_CONFIG)
        params, perf = optimizer.optimize(None, None)

        announcement = OptimaeAnnouncement(
            domain_id="quadratic-v1",
            optimae_id="test-optimae-001",
            parameters=params,
            reported_performance=perf,
        )
        message = Message(
            msg_type=MessageType.OPTIMAE_ANNOUNCEMENT,
            sender_id="test-optimizer",
            payload=json.loads(announcement.model_dump_json()),
        )

        async with ClientSession(timeout=ClientTimeout(total=5)) as session:
            async with session.post(
                "http://127.0.0.1:18470/message",
                json=json.loads(message.model_dump_json()),
            ) as resp:
                assert resp.status == 200
                data = await resp.json()
                assert data["status"] == "ok"

        # Node should have the optimae pending validation
        assert node.validator.pending_count == 1

    @pytest.mark.asyncio
    async def test_full_network_loop(self, node: Node, evaluator: EvaluatorService) -> None:
        """Full network loop: optimizer → node → evaluator verification → block."""
        optimizer = QuadraticOptimizer()
        optimizer.configure(PLUGIN_CONFIG)

        # Use relative tolerance — synthetic data noise is proportional to performance scale
        node.validator.relative_tolerance = 0.02  # 2% of |reported_performance|

        async with ClientSession(timeout=ClientTimeout(total=10)) as session:
            best_params = None
            best_perf = None
            blocks_before = node.chain.height

            for step in range(50):
                new_params, new_perf = optimizer.optimize(best_params, best_perf)
                if best_perf is not None and new_perf <= best_perf:
                    continue

                # Submit to node via HTTP
                announcement = OptimaeAnnouncement(
                    domain_id="quadratic-v1",
                    optimae_id=f"opt-{step}",
                    parameters=new_params,
                    reported_performance=new_perf,
                )
                message = Message(
                    msg_type=MessageType.OPTIMAE_ANNOUNCEMENT,
                    sender_id="test-optimizer",
                    payload=json.loads(announcement.model_dump_json()),
                )

                async with session.post(
                    "http://127.0.0.1:18470/message",
                    json=json.loads(message.model_dump_json()),
                ) as resp:
                    assert resp.status == 200

                # Verify via evaluator HTTP API
                async with session.post(
                    "http://127.0.0.1:18471/verify",
                    json={
                        "domain_id": "quadratic-v1",
                        "parameters": new_params,
                        "use_synthetic": True,
                    },
                ) as resp:
                    assert resp.status == 200
                    verify_data = await resp.json()

                # Feed verification back to node's validator
                verified_perf = verify_data["verified_performance"]
                increment = abs(new_perf - best_perf) if best_perf is not None else abs(new_perf)

                optimae = Optimae(
                    domain_id="quadratic-v1",
                    optimizer_id="test-optimizer",
                    parameters=new_params,
                    reported_performance=new_perf,
                    performance_increment=increment,
                )

                # Direct validator call (in production, node handles this internally)
                node.validator.submit_for_validation(optimae)
                result = node.validator.record_evaluation(optimae.id, verified_perf)

                if result.is_valid:
                    optimae.verified_performance = verified_perf
                    node.consensus.record_optimae(optimae)
                    best_params = new_params
                    best_perf = new_perf

                    # Update evaluator's current optimae
                    evaluator.update_optimae("quadratic-v1", optimae)

                    # Try block generation
                    block = await node.try_generate_block()
                    if block is not None:
                        break

            assert node.chain.height > blocks_before, "Should have generated at least one block"
            print(f"\n✅ Full network integration passed!")
            print(f"   Chain height: {node.chain.height}")
            print(f"   Final performance: {best_perf:.6f}")
