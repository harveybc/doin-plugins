# Adapt a New Optimization Domain to DOIN

This is the shortest honest path from an existing local optimizer to a DOIN
campaign. The optimizer remains in its own repository. `doin-plugins` contains
only a thin adapter to the protocol interfaces; do not move domain training
code into `doin-node` or `agent-multi`.

## Repositories and exact paths

Clone these beside your optimizer repository:

```text
doin-core/src/doin_core/plugins/base.py        protocol interfaces
doin-core/src/doin_core/plugins/loader.py      entry-point discovery
doin-plugins/src/doin_plugins/quadratic_*.py   complete reference domain
doin-plugins/pyproject.toml                    entry-point registration
doin-node/examples/quadratic_single_node.json  runnable node configuration
<your-optimizer>/                              your model, search and artifacts
```

Implement only the interfaces your trust model needs:

- `OptimizationPlugin.configure()` and `optimize()` call the existing local
  optimizer and return `(parameters, performance)`.
- `InferencePlugin.configure()` and `evaluate()` independently recompute the
  same performance from parameters and evaluation data.
- `SyntheticDataPlugin.configure()` and `generate(seed)` are required only for
  an untrusted setting where evaluators must challenge a reported result
  without disclosing or reusing the protected evaluation set.

`get_domain_metadata()` currently requires `performance_metric` and
`higher_is_better`. Until the protocol enforces metric comparability, do not
combine unrelated units such as accuracy, Sharpe ratio and negative MSE in one
weighted composite proof. Run one domain at a time or document that its metric
is `not_for_composite`; that marker is a human contract today, not a runtime
gate.

## Rung 1: prove the adapter locally

1. Copy the structure, not the arithmetic, of:
   `quadratic_optimizer.py`, `quadratic_inferencer.py` and
   `quadratic_synthetic.py`.
2. Add your package path under the matching entry-point groups in
   `doin-plugins/pyproject.toml` or in your own adapter package:

```toml
[project.entry-points."doin.optimization"]
my_domain = "my_doin_adapter.optimizer:MyOptimizer"

[project.entry-points."doin.inference"]
my_domain = "my_doin_adapter.inferencer:MyInferencer"

[project.entry-points."doin.synthetic_data"]
my_domain = "my_doin_adapter.synthetic:MySyntheticData"
```

3. Install the editable packages. From the directory that contains the sibling
   repositories, first prove the shipped reference plugin exactly as written:

```bash
pip install -e doin-core -e doin-plugins -e <your-optimizer>
python - <<'PY'
from doin_core.plugins.loader import load_optimization_plugin, load_inference_plugin

config = {
    "n_params": 3,
    "target": [1.0, -2.0, 0.5],
    "step_size": 0.25,
    "seed": 7,
}
optimizer = load_optimization_plugin("simple_quadratic")()
inferencer = load_inference_plugin("simple_quadratic")()
optimizer.configure(config)
inferencer.configure(config)
parameters, reported = optimizer.optimize(None, None)
verified = inferencer.evaluate(parameters)
assert abs(reported - verified) < 1e-12, (reported, verified)
print({"reported": reported, "verified": verified})
PY
```

Then replace `simple_quadratic` with the exact entry-point name you registered
(`my_domain` in the TOML example) and replace `config` with one tiny,
deterministic fixture for your optimizer. The optimizer and inferencer must
receive the same evaluation definition; do not leave a random/default target
on one side and call the resulting numbers independently verified.

Acceptance: one deterministic fixture, no network, no blockchain, and the
inferencer recomputes the metric rather than trusting the optimizer's number.

## Rung 2: trusted DOIN campaign

Copy `doin-node/examples/quadratic_single_node.json`, change the domain ID,
plugin names, bounds, target and `optimization_config`, then run it on an
unused port and output directory:

```bash
doin-node --config /path/to/my_domain_single_node.json \
  --port 8479 \
  --data-dir /tmp/my-domain-doin/data \
  --olap-db /tmp/my-domain-doin/olap.db \
  --stats-file /tmp/my-domain-doin/stats.csv
```

Acceptance: the dashboard names the intended plugin and domain, at least one
candidate is evaluated, the chain has one coherent tip, and the SQLite OLAP
rows can be joined to the exact experiment identity. A trusted/research profile
may skip synthetic challenge verification, but its documentation must say so;
it is not evidence for permissionless or adversarial validation.

Only after this single-node proof should the same immutable config and plugin
revision be deployed to multiple nodes. Peer independence is not established
by running several processes on one machine.

## Rung 3: untrusted verification

Do not enable this rung merely because a synthetic plugin exists. First prove,
on a held-out corpus, that challenges preserve the candidate ordering within a
declared tolerance and that an optimizer cannot improve its verified rank by
overfitting known test data. Persist seeds, generator revision, data hashes,
reported metric, independently verified metric and rejection reason.

If rank preservation, reproducibility or attack resistance fails, remain in
the trusted rung. Do not attach coin, reward-market or permissionless claims to
rungs 1 or 2.

## Pasteable agent assignment

```text
Act as a senior Python optimization-platform engineer. Adapt my existing
optimizer at <ABSOLUTE_OPTIMIZER_REPO> to DOIN without moving its domain logic.

Read first:
- doin-core/src/doin_core/plugins/base.py
- doin-core/src/doin_core/plugins/loader.py
- doin-plugins/src/doin_plugins/quadratic_optimizer.py
- doin-plugins/src/doin_plugins/quadratic_inferencer.py
- doin-plugins/src/doin_plugins/quadratic_synthetic.py
- doin-plugins/pyproject.toml
- doin-node/examples/quadratic_single_node.json
- doin-plugins/docs/ADAPT_A_NEW_DOMAIN_WITH_AN_AGENT.md

Fitness is <METRIC>; <HIGHER_OR_LOWER> is better. Keep my optimizer usable
locally. Add a thin adapter package and entry points named <PLUGIN_NAME>.
First prove a socket-free optimize/evaluate fixture. Then materialize a
single-node trusted config under /tmp on an unused port and show the exact
candidate, chain and OLAP evidence. Do not edit agent-multi unless this is the
trading domain. Do not claim untrusted verification unless a deterministic
synthetic challenge passes the held-out rank-preservation test. Do not combine
this metric with other domains unless their units are explicitly comparable.
Stop before any multi-machine deployment and report files, tests, remaining
unknowns and the exact next command.
```
