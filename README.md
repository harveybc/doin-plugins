# DON Plugins

**Reference plugins for the Decentralized Optimization Network (DON)**

Provides a simple quadratic optimization domain for testing and demonstrating the full DON pipeline without ML dependencies.

## Included Plugins

### Quadratic Optimizer (`simple_quadratic`)
Hill-climbing optimizer that minimizes `f(x) = Σ(x_i - target_i)²`. Uses random perturbations to explore the parameter space.

### Quadratic Inferencer (`simple_quadratic`)
Evaluates parameters against the quadratic function. Returns negative MSE as the performance metric (higher is better).

### Quadratic Synthetic Data (`simple_quadratic`)
Generates noisy versions of the target for evaluator verification, preventing overfitting to exact validation data.

## Installation

```bash
pip install -e ".[dev]"
```

Requires `doin-core` to be installed first.

## Usage

These plugins are registered via setuptools entry points. Once installed, DON components discover them automatically:

```python
from doin_core.plugins.loader import load_optimization_plugin

OptCls = load_optimization_plugin("simple_quadratic")
optimizer = OptCls()
optimizer.configure({"n_params": 5, "target": [1, 2, 3, 4, 5], "step_size": 0.5})
params, performance = optimizer.optimize(None, None)
```

## Integration Test

The integration test (`tests/test_integration.py`) runs the **complete DON pipeline**:

1. Optimizer produces improving parameters
2. Evaluator verifies reported performance
3. Validator accepts/rejects based on tolerance
4. Consensus tracks weighted performance increments
5. Block is generated when threshold is met
6. Chain grows with multiple blocks

```bash
pytest tests/test_integration.py -v -s
```

## License

MIT
