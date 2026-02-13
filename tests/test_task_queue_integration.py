"""Integration test: pull-based task queue flow.

Tests the full lifecycle:
1. Optimizer submits optimae → node creates verification task
2. Evaluator polls /tasks/pending → sees the task
3. Evaluator claims /tasks/claim → task moves to CLAIMED
4. Evaluator processes and reports /tasks/complete
5. Node records on-chain transaction and tries block generation
6. Client submits inference request → queued as task
"""

from __future__ import annotations

import asyncio
import json

import pytest
from aiohttp import ClientSession, ClientTimeout

from doin_core.models.domain import Domain, DomainConfig
from doin_core.models.task import TaskType
from doin_core.protocol.messages import Message, MessageType, OptimaeAnnouncement

from doin_node.node import Node, NodeConfig
from doin_evaluator.service import EvaluatorService, EvaluatorConfig

from doin_plugins.quadratic_optimizer import QuadraticOptimizer
from doin_plugins.quadratic_inferencer import QuadraticInferencer
from doin_plugins.quadratic_synthetic import QuadraticSyntheticData

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
    config = NodeConfig(
        host="127.0.0.1",
        port=18480,
        data_dir="/tmp/don-test-taskq-node",
        target_block_time=600.0,
        initial_threshold=0.5,
    )
    n = Node(config)
    n.register_domain(make_domain())
    await n.start()
    yield n
    await n.stop()


class TestTaskQueueIntegration:

    @pytest.mark.asyncio
    async def test_optimae_creates_verification_task(self, node: Node) -> None:
        """Submitting an optimae should create a verification task in the queue."""
        optimizer = QuadraticOptimizer()
        optimizer.configure(PLUGIN_CONFIG)
        params, perf = optimizer.optimize(None, None)

        announcement = OptimaeAnnouncement(
            domain_id="quadratic-v1",
            optimae_id="test-opt-001",
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
                "http://127.0.0.1:18480/message",
                json=json.loads(message.model_dump_json()),
            ) as resp:
                assert resp.status == 200

        # Task queue should have one pending verification task
        assert node.task_queue.pending_count == 1
        tasks = node.task_queue.get_pending()
        assert tasks[0].task_type == TaskType.OPTIMAE_VERIFICATION
        assert tasks[0].optimae_id == "test-opt-001"

    @pytest.mark.asyncio
    async def test_evaluator_pulls_and_processes_task(self, node: Node) -> None:
        """Evaluator should pull tasks via HTTP and complete them."""
        # Submit an optimae to create a task
        optimizer = QuadraticOptimizer()
        optimizer.configure(PLUGIN_CONFIG)
        params, perf = optimizer.optimize(None, None)

        announcement = OptimaeAnnouncement(
            domain_id="quadratic-v1",
            optimae_id="test-opt-002",
            parameters=params,
            reported_performance=perf,
        )
        message = Message(
            msg_type=MessageType.OPTIMAE_ANNOUNCEMENT,
            sender_id="test-optimizer",
            payload=json.loads(announcement.model_dump_json()),
        )

        async with ClientSession(timeout=ClientTimeout(total=5)) as session:
            # Submit optimae
            async with session.post(
                "http://127.0.0.1:18480/message",
                json=json.loads(message.model_dump_json()),
            ) as resp:
                assert resp.status == 200

            # Step 1: Poll for pending tasks
            async with session.get(
                "http://127.0.0.1:18480/tasks/pending?domains=quadratic-v1"
            ) as resp:
                assert resp.status == 200
                data = await resp.json()
                assert len(data["tasks"]) >= 1
                task = data["tasks"][0]
                task_id = task["id"]
                assert task["task_type"] == "optimae_verification"

            # Step 2: Claim the task
            async with session.post(
                "http://127.0.0.1:18480/tasks/claim",
                json={"task_id": task_id, "evaluator_id": "eval-001"},
            ) as resp:
                assert resp.status == 200
                data = await resp.json()
                assert data["status"] == "claimed"

            # Verify task is no longer pending
            assert node.task_queue.pending_count == 0
            assert node.task_queue.claimed_count == 1

            # Step 3: Process (verify the parameters)
            inferencer = QuadraticInferencer()
            inferencer.configure(PLUGIN_CONFIG)
            verified_perf = inferencer.evaluate(params)

            # Step 4: Report completion
            async with session.post(
                "http://127.0.0.1:18480/tasks/complete",
                json={
                    "task_id": task_id,
                    "verified_performance": verified_perf,
                    "result": {"used_synthetic": False},
                },
            ) as resp:
                assert resp.status == 200
                data = await resp.json()
                assert data["status"] == "completed"

            assert node.task_queue.completed_count == 1

    @pytest.mark.asyncio
    async def test_client_inference_request_queued(self, node: Node) -> None:
        """Client inference requests should be queued as tasks."""
        async with ClientSession(timeout=ClientTimeout(total=5)) as session:
            async with session.post(
                "http://127.0.0.1:18480/inference",
                json={
                    "domain_id": "quadratic-v1",
                    "input_data": {"x": [1, 2, 3]},
                    "client_id": "client-001",
                },
            ) as resp:
                assert resp.status == 200
                data = await resp.json()
                assert data["status"] == "queued"
                assert "task_id" in data

        # Should be in the queue as inference request
        tasks = node.task_queue.get_pending()
        assert any(t.task_type == TaskType.INFERENCE_REQUEST for t in tasks)

    @pytest.mark.asyncio
    async def test_verification_priority_over_inference(self, node: Node) -> None:
        """Verification tasks should be served before inference tasks."""
        # Submit inference first
        async with ClientSession(timeout=ClientTimeout(total=5)) as session:
            await session.post(
                "http://127.0.0.1:18480/inference",
                json={
                    "domain_id": "quadratic-v1",
                    "input_data": {},
                    "client_id": "client-001",
                },
            )

            # Then submit optimae (creates verification task)
            optimizer = QuadraticOptimizer()
            optimizer.configure(PLUGIN_CONFIG)
            params, perf = optimizer.optimize(None, None)

            message = Message(
                msg_type=MessageType.OPTIMAE_ANNOUNCEMENT,
                sender_id="test-optimizer",
                payload=json.loads(OptimaeAnnouncement(
                    domain_id="quadratic-v1",
                    optimae_id="test-opt-prio",
                    parameters=params,
                    reported_performance=perf,
                ).model_dump_json()),
            )
            await session.post(
                "http://127.0.0.1:18480/message",
                json=json.loads(message.model_dump_json()),
            )

            # Poll — verification should come first
            async with session.get(
                "http://127.0.0.1:18480/tasks/pending?domains=quadratic-v1"
            ) as resp:
                data = await resp.json()
                tasks = data["tasks"]
                assert len(tasks) >= 2
                assert tasks[0]["task_type"] == "optimae_verification"

    @pytest.mark.asyncio
    async def test_node_status_includes_task_stats(self, node: Node) -> None:
        """GET /status should include task queue statistics."""
        async with ClientSession(timeout=ClientTimeout(total=5)) as session:
            async with session.get("http://127.0.0.1:18480/status") as resp:
                assert resp.status == 200
                data = await resp.json()
                assert "task_queue" in data
                assert "pending" in data["task_queue"]
                assert "claimed" in data["task_queue"]
                assert "completed" in data["task_queue"]

    @pytest.mark.asyncio
    async def test_claim_nonexistent_task_returns_409(self, node: Node) -> None:
        """Claiming a task that doesn't exist should return 409."""
        async with ClientSession(timeout=ClientTimeout(total=5)) as session:
            async with session.post(
                "http://127.0.0.1:18480/tasks/claim",
                json={"task_id": "nonexistent", "evaluator_id": "eval-1"},
            ) as resp:
                assert resp.status == 409
