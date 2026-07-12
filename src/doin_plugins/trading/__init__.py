"""DOIN adapters for the agent-multi trading domain.

The trading implementation remains independently usable from ``agent-multi``.
These classes only adapt that implementation to the existing DOIN entry-point
contracts; they do not add trading logic to doin-node.
"""

from .inferencer import TradingInferencer
from .optimizer import TradingOptimizer
from .synthetic import TradingScenarioSyntheticData

__all__ = [
    "TradingInferencer",
    "TradingOptimizer",
    "TradingScenarioSyntheticData",
]
