"""I6-N swarm mode of the shared predictor wrapper (lane B).

Uses the REAL wrapper, the REAL predictor plugin loader and the REAL I6N DEAP
plugin from a predictor checkout (I6N_PREDICTOR_ROOT), with a fake evaluator
subprocess on CPU and a throwaway ledger. No TensorFlow, no data, no node.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import threading

import pytest

PREDICTOR = Path(os.environ.get("I6N_PREDICTOR_ROOT", "/var/tmp/satoshi-i6n-B-20261011"))
if not (PREDICTOR / "tools" / "i6n_swarm" / "bridge.py").exists():
    pytest.skip("I6N_PREDICTOR_ROOT with tools/i6n_swarm is required", allow_module_level=True)
if str(PREDICTOR) not in sys.path:
    sys.path.insert(0, str(PREDICTOR))

from doin_plugins.predictor.optimizer import I6N_SEED, PredictorOptimizer  # noqa: E402
from tools import i6n_search as search  # noqa: E402
from tools.i6n_swarm import common, ledger as L  # noqa: E402

FAKE = r'''
import json, os, sys, hashlib
req = json.load(sys.stdin)
open(os.path.join(os.environ["FAKE_COUNT_DIR"], "%s-%d" % (req["context"]["fold"], len(os.listdir(os.environ["FAKE_COUNT_DIR"])))), "w").write("x")
d = lambda v: hashlib.sha256(json.dumps(v, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
gid = hashlib.sha256(json.dumps({"schema": "i6n-search-v1", "config": req["config"]}, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("ascii")).hexdigest()
s = req["support"]
print(json.dumps(dict(status="ok", counts=s["counts"], naive_sums=s["naive_sums"], support_digest=s["support_digest"],
    error_sums={k: v * 0.9 for k, v in s["naive_sums"].items()},
    metadata=dict(candidate_receipt=dict(schema_version="i6n.candidate_receipt.v1", experiment="exp", fold=req["context"]["fold"],
        effective_config_sha256=d(req["config"]), horizon_hours=req["config"]["horizon"], genotype_canonical_sha256=gid,
        seed=6102026, rows=[])))))
'''


def support(scale):
    return dict(counts=dict(train=100, inner=10), naive_sums=dict(train=1.0 * scale, inner=0.1 * scale),
                support_digest=dict(train=common.sha256_json(["t", scale]), inner=common.sha256_json(["i", scale])))


@pytest.fixture
def setup(tmp_path, monkeypatch):
    (tmp_path / "fake.py").write_text(FAKE)
    counts = tmp_path / "counts"
    counts.mkdir()
    monkeypatch.setenv("FAKE_COUNT_DIR", str(counts))
    ledger_path = tmp_path / "ledger.sqlite3"
    L.Ledger(ledger_path).dag_upsert("search:h12", "search", 0, [], {"horizon": 12})
    node = dict(schema="i6n.swarm.node.v1", campaign_id="fx", owner="preferred-cpu_fixture", host_key="preferred",
                device=dict(cls="cpu_fixture", cuda_visible_devices="", fixture=True), fixture=True,
                exclusive_host=True, evaluator_argv=[sys.executable, str(tmp_path / "fake.py")], workdir=str(tmp_path),
                launcher=None, class_caps={}, lease=dict(ttl_seconds=5, renew_seconds=0.2, max_wait_seconds=5,
                                                         poll_seconds=0.1),
                ledger=dict(path=str(ledger_path)), logs_dir=str(tmp_path / "logs"), state_dir=str(tmp_path / "state"))
    node_path = tmp_path / "node.json"
    node_path.write_text(json.dumps(node))
    folds = [dict(fold=f, context=dict(fold=f), support=support(i + 1), binding_sha256=None)
             for i, f in enumerate(("2023Q1", "2023Q2"))]
    context = dict(schema="i6n.swarm.context.v1", experiment="exp", plan_sha256="p" * 64, horizon=12,
                   dag_node="search:h12", fixture=True, budget_new_candidates=4, classes=[], folds=folds)
    fold_set = dict(counts=dict(train=200, inner=20), naive_sums=dict(train=3.0, inner=0.3),
                    support_digest=dict(train="a" * 64, inner="b" * 64))
    config = dict(predictor_root=str(PREDICTOR), optimizer_plugin="i6n_optimizer", shared_population=True,
                  shared_population_size=2, shared_population_seed=I6N_SEED, n_generations=3,
                  optimization_patience=3, improvement_tolerance=1e-4, reproduction_failure_policy="fail_closed",
                  higher_is_better=False, i6n_budget=4, i6n_max_proposals=128, i6n_horizon=12,
                  i6n_support=fold_set, i6n_evaluation_context=context,
                  i6n_swarm=dict(node_config=str(node_path), horizon=12, dag_node="search:h12", seed=I6N_SEED))
    return config, counts, ledger_path


def configured(config):
    plugin = PredictorOptimizer()
    plugin.configure(config)
    return plugin


def test_swarm_mode_loads_only_the_real_i6n_plugin_without_tensorflow(setup):
    config, _, _ = setup
    plugin = configured(config)
    assert type(plugin._deap_optimizer).__module__ == "optimizer_plugins.i6n_optimizer"
    assert plugin._predictor_plugin is None and plugin._preprocessor_plugin is None and plugin._target_plugin is None
    assert "tensorflow" not in sys.modules
    meta = plugin.get_domain_metadata()
    assert meta["higher_is_better"] is False and meta["seed"] == I6N_SEED and meta["direction"] == "minimize"
    assert meta["mode"] == "i6n_swarm"


def test_no_silent_default_optimizer_fallback(setup, tmp_path):
    config, _, _ = setup
    bad = dict(config, optimizer_plugin="i6n_optimizer_missing")
    with pytest.raises(Exception) as info:
        configured(bad)
    assert "i6n" in str(info.value).lower()


@pytest.mark.parametrize("key,value", [("shared_population_seed", 42), ("higher_is_better", True)])
def test_seed_and_direction_must_agree_in_config(setup, key, value):
    config, _, _ = setup
    with pytest.raises(ValueError):
        configured(dict(config, **{key: value}))


def test_role_validation_hook(setup):
    plugin = configured(setup[0])
    plugin.validate_domain_role(dict(higher_is_better=False, shared_population_seed=I6N_SEED))
    with pytest.raises(ValueError, match="minimiz"):
        plugin.validate_domain_role(dict(higher_is_better=True, shared_population_seed=I6N_SEED))
    with pytest.raises(ValueError, match="6102026"):
        plugin.validate_domain_role(dict(higher_is_better=False, shared_population_seed=42))


def test_population_requires_campaign_seed_and_is_persisted(setup):
    config, _, ledger_path = setup
    plugin = configured(config)
    with pytest.raises(ValueError, match="6102026"):
        plugin.create_shared_population(2, seed=42)
    state = plugin.create_shared_population(2, seed=I6N_SEED)
    assert 1 <= len(state["population"]) <= 2
    stored = L.Ledger(ledger_path).search_state("search:h12")
    assert stored["generation"] == 0 and stored["state"]["population"] == state["population"]


def test_evaluation_in_node_thread_uses_folds_and_never_refits(setup):
    config, counts, _ = setup
    plugin = configured(config)
    state = plugin.create_shared_population(2, seed=I6N_SEED)
    wire = state["population"][0]
    box = {}
    worker = threading.Thread(target=lambda: box.update(result=plugin.evaluate_candidate(wire, 0)))
    worker.start()
    worker.join(30)
    result = box["result"]
    assert result["fitness"] == pytest.approx(0.9)
    assert result["candidate_identity"] == wire["identity"]
    assert result["hyper_dict"] == search.effective_config(wire["genome"])
    assert len(os.listdir(counts)) == 2
    again = configured(config).evaluate_candidate(wire, 0)
    assert again["fitness"] == result["fitness"] and len(os.listdir(counts)) == 2


def test_reproduction_seed_is_campaign_derived_and_terminal_is_recorded(setup):
    config, _, ledger_path = setup
    plugin = configured(config)
    state = plugin.create_shared_population(2, seed=I6N_SEED)
    evaluated = [dict(w, fitness=0.5 + i / 10) for i, w in enumerate(state["population"])]
    args = (state["innovation_tracker"], state["stage_schedule"], state["param_defaults"], 0, 0)
    a = plugin.reproduce_shared(evaluated, 0, 111, *args)
    b = configured(config).reproduce_shared(evaluated, 0, 999, *args)
    assert a["population"] == b["population"]
    assert a["seed_used"] == I6N_SEED + 1 and a["node_seed_ignored"] == 111
    led = L.Ledger(ledger_path)
    assert led.search_state("search:h12")["generation"] == 1
    last = a
    generation = 1
    while not last["converged"]:
        pop = [dict(w, fitness=0.4) for w in last["population"]]
        last = plugin.reproduce_shared(pop, generation, 5, last["innovation_tracker"], last["stage_schedule"],
                                       last["param_defaults"], 0, last["no_improve_count"])
        generation += 1
    events = led.dag_events("search:h12")
    assert events and events[0]["event"] == "SEARCH_TERMINAL"
    assert events[0]["detail"]["stop_reason"] == last["stop_reason"]


def test_node_loss_during_reproduction_recomputes_identically_without_refits(setup):
    """A node dies after its generation's fits but before reproduction completes."""
    config, counts, ledger_path = setup
    first = configured(config)
    state = first.create_shared_population(2, seed=I6N_SEED)
    results = [first.evaluate_candidate(w, 0) for w in state["population"]]
    fits_before = len(os.listdir(counts))
    del first  # lost during reproduce: nothing for generation 1 was persisted
    assert L.Ledger(ledger_path).search_state("search:h12")["generation"] == 0
    restarted = configured(config)
    again = [restarted.evaluate_candidate(w, 0) for w in state["population"]]
    assert [r["fitness"] for r in again] == [r["fitness"] for r in results]
    assert len(os.listdir(counts)) == fits_before  # adopted from the ledger, never refit
    evaluated = [dict(w, fitness=r["fitness"]) for w, r in zip(state["population"], again)]
    args = (state["innovation_tracker"], state["stage_schedule"], state["param_defaults"], 0, 0)
    a = restarted.reproduce_shared(evaluated, 0, 1, *args)
    b = configured(config).reproduce_shared(evaluated, 0, 2, *args)  # a peer recomputes the same generation
    assert a["population"] == b["population"] and a["generation"] == 1
    assert L.Ledger(ledger_path).search_state("search:h12")["generation"] == 1
