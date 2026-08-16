# AGENTS.md - doin-plugins

`doin-plugins` contains protocol adapters, not the optimizers they wrap. Keep
domain models and search code in their native repositories and implement the
ABCs from `doin-core/src/doin_core/plugins/base.py` here or in another
installable adapter package.

Before adding a domain, follow
[`docs/ADAPT_A_NEW_DOMAIN_WITH_AN_AGENT.md`](docs/ADAPT_A_NEW_DOMAIN_WITH_AN_AGENT.md).
Use `simple_quadratic` as the executable reference and prove the three rungs in
order: local interface, trusted single-node campaign, then an untrusted
synthetic challenge only when its held-out rank-preservation evidence exists.

Run:

```bash
pip install -e .[dev]
pytest -q
```

Do not launch fleet workers from this repository. Do not add broker credentials,
machine addresses or private experiment output. Do not present weighted sums of
unrelated domain metrics as meaningful composite optimization.

