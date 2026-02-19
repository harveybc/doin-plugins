# doin-plugins

Domain plugins for DOIN (Decentralized Optimization and Inference Network).

## Included Plugins

### Quadratic (Reference)
Simple quadratic function optimization — no ML frameworks needed. Used for testing the full DOIN pipeline.

- **Optimizer**: Hill-climbing on `f(x) = Σ(x_i - target_i)²`
- **Inferencer**: Evaluates parameters against target
- **Synthetic Data**: Generates noisy target variants for verification

### Predictor (ML — DEAP GA Wrapper)
Wraps [harveybc/predictor](https://github.com/harveybc/predictor) timeseries prediction system. Requires TensorFlow.

The predictor plugin **wraps predictor's full DEAP genetic algorithm** — it doesn't replace it. DOIN hooks into the GA via callbacks, adding decentralized island-model migration on top of the existing optimizer.

**DEAP GA callback hooks:**
- `on_generation_start` — called before each generation (inject migrated champions here)
- `on_generation_end` — called after each generation (collect stats, check convergence)
- `on_between_candidates` — called between candidate evaluations within a generation
- `on_champion_found` — called when a new best individual is found (broadcast to network)

**5-stage incremental optimization:**
Predictor's GA runs 5 stages of increasing complexity. DOIN wraps all stages, broadcasting champions at each stage boundary. Other nodes can inject received champions into their populations at any stage.

**Champion broadcasting and migration injection:**
When `on_champion_found` fires, DOIN broadcasts the champion's parameters as an optimae. Other nodes receiving this optimae inject the champion into their local DEAP population, creating new genetic material for crossover — the classic island model.

**Three-Level Patience System:**
| Level | Config Key | Controls | Default |
|-------|------------|----------|---------|
| **L1** — Candidate Training | `early_patience` | Keras early stopping per candidate | 80–100 |
| **L2** — Stage Progression | `optimization_patience` | GA generations before advancing stage | 8–10 |
| **L3** — Meta-Optimizer | *(not yet implemented)* | Network-level param→performance predictor | — |

- **Optimizer**: Wraps predictor's DEAP GA with callback hooks for island-model migration
- **Inferencer**: Evaluates model on test/synthetic data
- **Synthetic Data**: Wraps [harveybc/timeseries-gan](https://github.com/harveybc/timeseries-gan) (SC-VAE-GAN) with block bootstrap fallback

## Entry Points

```
doin.optimization/quadratic → QuadraticOptimizer
doin.optimization/predictor → PredictorOptimizer
doin.inference/quadratic → QuadraticInferencer
doin.inference/predictor → PredictorInferencer
doin.synthetic_data/quadratic → QuadraticSyntheticData
doin.synthetic_data/predictor → PredictorSyntheticData
```

## Install

```bash
pip install git+https://github.com/harveybc/doin-core.git
pip install git+https://github.com/harveybc/doin-plugins.git
```

## Tests

```bash
python -m pytest tests/ -v
# 33 tests passing (7 e2e lifecycle + unit + integration)
```

### Key Tests
- `test_e2e_lifecycle.py` — Full optimae lifecycle: optimize → commit → reveal → quorum → verify → incentive → reputation (7 tests)
- `test_plugins.py` — Quadratic plugin unit tests
- `test_network_integration.py` — Multi-component integration

## Part of DOIN

- [doin-core](https://github.com/harveybc/doin-core) — Consensus, models, crypto
- [doin-node](https://github.com/harveybc/doin-node) — Unified node
- [doin-plugins](https://github.com/harveybc/doin-plugins) — This package
