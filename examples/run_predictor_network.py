#!/usr/bin/env python3
"""Launch a minimal DON network for the predictor domain.

Boots up:
  1. One DON node (consensus + validation + chain)
  2. One DON evaluator (verifies optimizer claims)
  3. One DON optimizer (runs predictor training loop)

Usage:
    python run_predictor_network.py --config path/to/mimo_optimization_config.json

Or with defaults:
    python run_predictor_network.py

Requirements:
    - predictor repo installed (pip install -e /path/to/predictor)
    - doin-core, doin-node, doin-optimizer, doin-evaluator, doin-plugins installed
    - TensorFlow, numpy, pandas, etc. (predictor dependencies)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Any

from doin_core.crypto.identity import PeerIdentity
from doin_core.models.domain import Domain, DomainConfig
from doin_core.models.optimae import Optimae
from doin_core.protocol.messages import Message, MessageType, OptimaeAnnouncement

from doin_node.node import Node, NodeConfig
from doin_evaluator.service import EvaluatorService, EvaluatorConfig
from doin_optimizer.runner import OptimizationRunner, OptimizerConfig

from doin_plugins.predictor.optimizer import PredictorOptimizer
from doin_plugins.predictor.inferencer import PredictorInferencer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("doin.predictor_network")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run DON network for predictor domain")
    p.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to predictor optimization config JSON",
    )
    p.add_argument(
        "--predictor-root",
        type=str,
        default=None,
        help="Path to predictor repo root",
    )
    p.add_argument(
        "--node-port",
        type=int,
        default=8470,
        help="DON node port (default: 8470)",
    )
    p.add_argument(
        "--evaluator-port",
        type=int,
        default=8471,
        help="DON evaluator port (default: 8471)",
    )
    p.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Max optimization steps (default: unlimited)",
    )
    p.add_argument(
        "--step-interval",
        type=float,
        default=5.0,
        help="Seconds between optimization steps (default: 5.0)",
    )
    return p.parse_args()


def build_plugin_config(args: argparse.Namespace) -> dict[str, Any]:
    """Build the DON plugin config from CLI args."""
    config: dict[str, Any] = {}

    if args.predictor_root:
        config["predictor_root"] = args.predictor_root
    if args.config:
        config["load_config"] = args.config

    return config


def make_domain(plugin_config: dict[str, Any]) -> Domain:
    """Create the predictor DON domain."""
    return Domain(
        id="predictor-v1",
        name="Timeseries Predictor Optimization",
        performance_metric="neg_fitness_delta_from_naive",
        higher_is_better=True,
        weight=1.0,
        config=DomainConfig(
            optimization_plugin="predictor",
            inference_plugin="predictor",
            plugin_config=plugin_config,
        ),
    )


async def run_network(args: argparse.Namespace) -> None:
    """Boot and run the full DON network."""
    plugin_config = build_plugin_config(args)
    domain = make_domain(plugin_config)

    # --- Node ---
    node_config = NodeConfig(
        host="127.0.0.1",
        port=args.node_port,
        data_dir="/tmp/don-predictor-node",
        target_block_time=3600.0,  # 1 hour target (predictor evals are slow)
        initial_threshold=0.01,    # Low threshold — each improvement matters
    )
    node = Node(node_config)
    node.register_domain(domain)

    # --- Evaluator ---
    evaluator_config = EvaluatorConfig(
        host="127.0.0.1",
        port=args.evaluator_port,
        node_endpoint=f"127.0.0.1:{args.node_port}",
    )
    evaluator = EvaluatorService(evaluator_config)

    inf_plugin = PredictorInferencer()
    inf_plugin.configure(plugin_config)
    evaluator.set_domain_plugins("predictor-v1", domain, inf_plugin, None)

    # --- Optimizer ---
    optimizer_config = OptimizerConfig(
        domain_id="predictor-v1",
        plugin_name="predictor",
        plugin_config=plugin_config,
        node_endpoint=f"127.0.0.1:{args.node_port}",
        optimization_interval=args.step_interval,
        max_steps=args.max_steps,
    )
    optimizer = OptimizationRunner(optimizer_config)
    optimizer.load_plugin()

    # --- Start everything ---
    logger.info("=" * 60)
    logger.info("DON Predictor Network Starting")
    logger.info("=" * 60)
    logger.info("Node:      http://127.0.0.1:%d", args.node_port)
    logger.info("Evaluator: http://127.0.0.1:%d", args.evaluator_port)
    logger.info("Domain:    predictor-v1")
    if args.config:
        logger.info("Config:    %s", args.config)
    logger.info("=" * 60)

    try:
        await node.start()
        await evaluator.start()

        logger.info("Network ready — starting optimization loop...")
        logger.info(
            "Press Ctrl+C to stop. Stats at http://127.0.0.1:%d/health",
            args.node_port,
        )

        # Run optimizer (blocks until max_steps or Ctrl+C)
        await optimizer.start()

    except KeyboardInterrupt:
        logger.info("Shutting down...")
    finally:
        await optimizer.stop()
        await evaluator.stop()
        await node.stop()

        # Print final stats
        stats = optimizer.stats
        logger.info("=" * 60)
        logger.info("Final Results")
        logger.info("=" * 60)
        logger.info("Steps completed:  %d", stats["steps_completed"])
        logger.info("Improvements:     %d", stats["improvements_found"])
        logger.info("Best performance: %s", stats["current_best_performance"])
        logger.info("Chain height:     %d", node.chain.height)
        logger.info("=" * 60)


def main() -> None:
    args = parse_args()
    asyncio.run(run_network(args))


if __name__ == "__main__":
    main()
