# doin-plugins

**Status: ACTIVE — component of the DOIN family.**

`doin-plugins` provides the reusable plugin **implementations** for DOIN, the
Decentralized Optimization and Inference Network. Every class here subclasses
the abstract interfaces defined in
[doin-core](https://github.com/harveybc/doin-core)
(`OptimizationPlugin`, `InferencePlugin`, `SyntheticDataPlugin`) and is
registered under the entry-point groups `doin.optimization`, `doin.inference`,
and `doin.synthetic_data`, so the unified runtime
[doin-node](https://github.com/harveybc/doin-node) can discover them by name
from a per-machine JSON config.

## Role and non-responsibilities

**Role:** concrete plugin implementations — a self-contained quadratic
reference domain plus adapters that connect external domain optimizers
(timeseries predictor, agent-multi trading) to the DOIN protocol.

**Not in this repository:**

- No plugin ABCs or entry-point group definitions — those live in
  [doin-core](https://github.com/harveybc/doin-core)
  (`doin_core.plugins.base` / `doin_core.plugins.loader`).
- No node runtime, consensus, networking, or OLAP — that is
  [doin-node](https://github.com/harveybc/doin-node).
- **Not the home of domain models.** Domain optimizers remain external
  installable packages that work locally without DOIN. The predictor and
  trading plugins here are *adapters*: the real optimizers stay in their own
  repositories, and these plugins only add DOIN migration callbacks and the
  plugin contract (see the module docstring of
  [`src/doin_plugins/trading/optimizer.py`](src/doin_plugins/trading/optimizer.py)).

## Registered entry points

From [`pyproject.toml`](pyproject.toml) (names as `doin-node` configs must
reference them):

| Name | `doin.optimization` | `doin.inference` | `doin.synthetic_data` |
|---|---|---|---|
| `simple_quadratic` | [`quadratic_optimizer.py`](src/doin_plugins/quadratic_optimizer.py) | [`quadratic_inferencer.py`](src/doin_plugins/quadratic_inferencer.py) | [`quadratic_synthetic.py`](src/doin_plugins/quadratic_synthetic.py) |
| `predictor` | [`predictor/optimizer.py`](src/doin_plugins/predictor/optimizer.py) | [`predictor/inferencer.py`](src/doin_plugins/predictor/inferencer.py) | [`predictor/synthetic.py`](src/doin_plugins/predictor/synthetic.py) |
| `binary_predictor` | [`predictor/binary_optimizer.py`](src/doin_plugins/predictor/binary_optimizer.py) | [`predictor/binary_inferencer.py`](src/doin_plugins/predictor/binary_inferencer.py) | — |
| `trading_asset` | [`trading/optimizer.py`](src/doin_plugins/trading/optimizer.py) | [`trading/inferencer.py`](src/doin_plugins/trading/inferencer.py) | — |
| `trading_scenario` | — | — | [`trading/synthetic.py`](src/doin_plugins/trading/synthetic.py) |

The plugin families:

- **`simple_quadratic`** — self-contained reference domain (hill-climbing on a
  quadratic loss). No ML frameworks; used to exercise the full DOIN pipeline.
- **`predictor` / `binary_predictor`** — adapters around the external
  [predictor](https://github.com/harveybc/predictor) timeseries system's
  genetic-algorithm optimizer, adding island-model champion
  migration callbacks. Require that package and its ML stack at runtime.
- **`trading_asset` / `trading_scenario`** — adapters around the external
  [agent-multi](https://github.com/harveybc/agent-multi) trading optimizer.
  [`src/doin_plugins/trading/runtime.py`](src/doin_plugins/trading/runtime.py)
  (`AgentMultiRuntime`) resolves an agent-multi checkout from the plugin
  config (`agent_multi_root`), loads its canonical experiment JSON and
  entry-point plugins, and exposes the same local pipeline that
  `agent-multi --load_config` uses. The local optimizer remains in
  agent-multi; this package never replaces it.

## Requirements

From [`pyproject.toml`](pyproject.toml):

- Python `>=3.10`
- `doin-core>=0.1.0`, `numpy>=1.24`, `pandas>=2.1`
- Runtime-only extras not declared in packaging metadata: the `predictor` /
  `binary_predictor` plugins import the external predictor package (with its
  TensorFlow stack), and the `trading_*` plugins import an agent-multi
  checkout resolved at configure time. The `simple_quadratic` family has no
  such requirement.

## Installation

```bash
git clone https://github.com/harveybc/doin-core.git
git clone https://github.com/harveybc/doin-plugins.git
pip install -e doin-core -e doin-plugins
```

Verified 2026-08-10 in the maintainer's Python 3.12 environment: importing
`doin_plugins` succeeds and all five entry-point names above are visible via
`importlib.metadata.entry_points`. No PyPI release; install from source.

## Smallest working example

Load the quadratic reference plugins through the entry-point loader, run one
optimization step, verify it, and hash deterministic synthetic data. Executed
successfully on 2026-08-10:

```python
from doin_core.plugins.loader import (
    load_optimization_plugin,
    load_inference_plugin,
    load_synthetic_data_plugin,
)

optimizer = load_optimization_plugin("simple_quadratic")()
inferencer = load_inference_plugin("simple_quadratic")()
synthetic = load_synthetic_data_plugin("simple_quadratic")()

config = {"n_params": 4, "step_size": 0.5, "seed": 42,
          "target": [1.0, -2.0, 3.0, 0.5]}
optimizer.configure(config)
inferencer.configure(config)
synthetic.configure(config)

params, reported = optimizer.optimize(None, None)
verified = inferencer.evaluate(params)          # same value as reported
data, data_hash = synthetic.generate_with_hash(seed=1234)
print(round(reported, 4), round(verified, 4), data_hash[:16])
```

## Using these plugins in a DOIN network

`doin-node` configs reference plugins by entry-point name, for example the
domain block of
[doin-node's single-node quadratic example](https://github.com/harveybc/doin-node/blob/master/examples/quadratic_single_node.json):

```json
{
  "domain_id": "quadratic",
  "optimize": true,
  "evaluate": true,
  "optimization_plugin": "simple_quadratic",
  "inference_plugin": "simple_quadratic",
  "synthetic_data_plugin": "simple_quadratic"
}
```

[`examples/run_predictor_network.py`](examples/run_predictor_network.py) is a
historical walkthrough that boots a node together with the retired standalone
`doin-optimizer` / `doin-evaluator` clients; it additionally requires the
external predictor stack and those legacy packages. Current deployments run
everything through `doin-node` roles instead.

## Tests

```bash
pip install -e .[dev]
pytest -q
```

Observed 2026-08-10: `pytest -q --collect-only | tail -1` reports
**44 tests collected** across 7 test files in [`tests/`](tests), including the
end-to-end optimae lifecycle
([`tests/test_e2e_lifecycle.py`](tests/test_e2e_lifecycle.py)) and the trading
adapter contract ([`tests/test_trading_plugins.py`](tests/test_trading_plugins.py)).
(Collection count only; run `pytest -q` for a full pass.)

## Artifacts and outputs

The quadratic plugins keep everything in memory. The predictor/trading
adapters delegate artifact handling (models, training stats) to their external
packages and report metrics back to the calling `doin-node`, which owns
on-chain persistence, deduplication, and OLAP recording.

## Safety and trading disclaimer

The `trading_asset` / `trading_scenario` plugins operate on historical or
synthetic market data through agent-multi's simulation/backtest pipeline. They
place no live orders and require no exchange, broker, or API credentials.
Nothing in this repository is financial advice.

## Limitations

- `predictor`, `binary_predictor`, and `trading_*` plugins are unusable
  without their external packages present at runtime; only
  `simple_quadratic` is fully self-contained.
- Version `0.1.0` (alpha); no PyPI distribution.

## Related repositories

- [doin-core](https://github.com/harveybc/doin-core) — protocol primitives and
  the plugin ABCs implemented here
- [doin-node](https://github.com/harveybc/doin-node) — unified participant
  runtime that loads these plugins by entry-point name
- [predictor](https://github.com/harveybc/predictor) and
  [agent-multi](https://github.com/harveybc/agent-multi) — external domain
  packages wrapped by the adapter plugins

## License

Declared MIT in [`pyproject.toml`](pyproject.toml); the repository does not
currently ship a standalone `LICENSE` file.
