# doin-plugins

Domain plugins for DOIN (Decentralized Optimization and Inference Network).

## Included Plugins

### Quadratic (Reference)
Simple quadratic function optimization — no ML frameworks needed. Used for testing the full DOIN pipeline.

- **Optimizer**: Hill-climbing on `f(x) = Σ(x_i - target_i)²`
- **Inferencer**: Evaluates parameters against target
- **Synthetic Data**: Generates noisy target variants for verification

### Predictor (ML)
Wraps [harveybc/predictor](https://github.com/harveybc/predictor) timeseries prediction system. Requires TensorFlow.

- **Optimizer**: Runs predictor training with genetic algorithm
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
# 43 tests (including end-to-end lifecycle)
```

### Key Tests
- `test_e2e_lifecycle.py` — Full optimae lifecycle: optimize → commit → reveal → quorum → verify → incentive → reputation (7 tests)
- `test_plugins.py` — Quadratic plugin unit tests
- `test_network_integration.py` — Multi-component integration

## Part of DOIN

- [doin-core](https://github.com/harveybc/doin-core) — Consensus, models, crypto
- [doin-node](https://github.com/harveybc/doin-node) — Unified node
- [doin-plugins](https://github.com/harveybc/doin-plugins) — This package
