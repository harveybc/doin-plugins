"""I6-N swarm mode of the shared predictor wrapper (lane B).

Uses the REAL wrapper, the REAL predictor plugin loader and the REAL I6N DEAP
plugin from a predictor checkout (I6N_PREDICTOR_ROOT), with a fake evaluator
subprocess on CPU and a throwaway ledger. No TensorFlow, no data, no node.

Pinned dependency: verified with doin-core a90bca2c (OptimizationPlugin base). Set
DOIN_CORE_EXPECTED_COMMIT to make the run refuse any other doin-core checkout.
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

import doin_core  # noqa: E402

_EXPECTED_CORE = os.environ.get("DOIN_CORE_EXPECTED_COMMIT")
if _EXPECTED_CORE:
    import subprocess as _sp

    _core = Path(doin_core.__file__).resolve().parents[2]
    _head = _sp.run(["git", "-C", str(_core), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    if not _head.startswith(_EXPECTED_CORE):
        raise RuntimeError(f"doin-core at {_head}, expected {_EXPECTED_CORE}")

from doin_plugins.predictor.optimizer import I6N_SEED, PredictorOptimizer  # noqa: E402
from tools import i6n_search as search  # noqa: E402
from tools.i6n_swarm import common, ledger as L  # noqa: E402

FAKE = r'''
import json, os, sys, hashlib
req = json.load(sys.stdin)
ctx = req["context"]
open(os.path.join(os.environ["FAKE_COUNT_DIR"], "%s-%d" % (ctx["fold"], len(os.listdir(os.environ["FAKE_COUNT_DIR"])))), "w").write("x")
d = lambda v: hashlib.sha256(json.dumps(v, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
gid = hashlib.sha256(json.dumps({"schema": "i6n-search-v1", "config": req["config"]}, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("ascii")).hexdigest()
s = req["support"]
h = req["config"]["horizon"]
rows = [dict(head_offset_hours=h, head_role="ENDPOINT", comparison=True, split=sp.upper(), support_sha256=ctx["origins"][sp])
        for sp in ("train", "inner")]
print(json.dumps(dict(status="ok", counts=s["counts"], naive_sums=s["naive_sums"], support_digest=s["support_digest"],
    error_sums={k: v * 0.9 for k, v in s["naive_sums"].items()},
    metadata=dict(candidate_receipt=dict(schema_version="i6n.candidate_receipt.v1", experiment="exp", fold=ctx["fold"],
        effective_config_sha256=d(req["config"]), horizon_hours=h, genotype_canonical_sha256=gid,
        seed=6102026, rows=rows)))))
'''

E_PLAN = json.loads((PREDICTOR / "docs" / "i6n" / "swarm" / "I6N_SEARCH_PLAN_v1_2.json").read_text())
FOLDS = ("2023Q1_DCA", "2023Q2_DCA", "2023Q3_DCA", "2023Q4")


def support(scale):
    return dict(counts=dict(train=100, inner=10), naive_sums=dict(train=1.0 * scale, inner=0.1 * scale),
                support_digest=dict(train=common.sha256_json(["t", scale]), inner=common.sha256_json(["i", scale])))


@pytest.fixture
def setup(tmp_path, monkeypatch):
    """Config rendered by predictor's real campaign renderer (E's v2 path, sealed plan v1.2)."""
    from tools.i6n_swarm import plan as P

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
    folds = []
    for i, f in enumerate(FOLDS):
        origins = dict(train=common.sha256_json(["ot", f]), inner=common.sha256_json(["oi", f]))
        folds.append(dict(fold=f, context=dict(fold=f, origins=origins), support=support(i + 1), binding_sha256=None,
                          origins_sha256=origins))
    context = dict(schema="i6n.swarm.context.v1", experiment="exp", plan_sha256=E_PLAN["plan_sha256"], horizon=12,
                   dag_node="search:h12", fixture=True, budget_new_candidates=24, classes=[], folds=folds)
    template = dict(campaign_id="fx", node_label="x", port=1, data_dir=str(tmp_path), bootstrap_peers=[],
                    shared_min_peers=2, claim_timeout=600, predictor_root=str(PREDICTOR), node_config=str(node_path),
                    search=dict(population_size=8, generations=3, new_candidates_per_horizon=24, mutation=0.8,
                                immigration=0.2, tolerance=1e-4, patience=3), e_plan=E_PLAN, admitted_classes=["S1"])
    config = P.render_doin_config(template, {"12": context})["domains"][0]["optimization_config"]
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


@pytest.mark.parametrize("key,value", [("shared_population_seed", 42), ("higher_is_better", True), ("i6n_plan", None)])
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
    state = plugin.create_shared_population(8, seed=I6N_SEED)
    assert {w["schema"] for w in state["population"]} == {"i6n-shared-deap-v2"}
    stored = L.Ledger(ledger_path).search_state("search:h12")
    assert stored["generation"] == 0 and stored["state"]["population"] == state["population"]


def test_evaluation_in_node_thread_uses_folds_and_never_refits(setup):
    config, counts, _ = setup
    plugin = configured(config)
    state = plugin.create_shared_population(8, seed=I6N_SEED)
    wire = state["population"][0]
    box = {}
    worker = threading.Thread(target=lambda: box.update(result=plugin.evaluate_candidate(wire, 0)))
    worker.start()
    worker.join(30)
    result = box["result"]
    assert result["fitness"] == pytest.approx(0.9)
    assert result["candidate_identity"] == wire["identity"]
    assert result["hyper_dict"] == search.effective_config(wire["genome"])
    assert result["metrics"]["origins_binding"] == "VERIFIED" and result["metrics"]["aggregator"] == "E.score_folds"
    assert len(os.listdir(counts)) == 4
    again = configured(config).evaluate_candidate(wire, 0)
    assert again["fitness"] == result["fitness"] and len(os.listdir(counts)) == 4


def test_reproduction_seed_is_campaign_derived_and_terminal_is_recorded(setup):
    config, _, ledger_path = setup
    plugin = configured(config)
    state = plugin.create_shared_population(8, seed=I6N_SEED)
    evaluated = [dict(w, fitness=0.5 + i / 10) for i, w in enumerate(state["population"])]
    args = (state["innovation_tracker"], state["stage_schedule"], state["param_defaults"], 0, 0)
    a = plugin.reproduce_shared(evaluated, 0, 111, *args)
    b = configured(config).reproduce_shared(evaluated, 0, 999, *args)
    assert a["population"] == b["population"]
    assert a["seed_used"] == I6N_SEED and a["node_seed_ignored"] == 111
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
    state = first.create_shared_population(8, seed=I6N_SEED)
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


def test_e_b1_e_b2_v2_population_has_minimal_seed_and_reproduction_keeps_the_plan_seed(setup):
    from tools import i6n_search_space as space
    config, _, _ = setup
    plugin = configured(config)
    state = plugin.create_shared_population(8, seed=I6N_SEED)
    assert space.genotype_sha256(space.seed_genotype(12)) in [w["identity"] for w in state["population"]]
    assert {space.class_of(w["genotype"]) for w in state["population"]} == {"S1"}
    evaluated = [dict(w, fitness=1.0 + i / 100) for i, w in enumerate(state["population"])]
    out = plugin.reproduce_shared(evaluated, 0, 987654, state["innovation_tracker"], state["stage_schedule"],
                                  state["param_defaults"], 0, 0)  # E's v2 refuses any seed but the plan seed
    assert out["seed_used"] == I6N_SEED and out["generation"] == 1


def test_e_b11_disconnect_mid_reproduction_resumes_from_search_state(setup, monkeypatch):
    """The ledger becomes unreachable while a node persists generation 1: reproduction raises and nothing
    is half-written; the restarted node (or a peer) recomputes the same generation and persists it once."""
    config, counts, ledger_path = setup
    first = configured(config)
    state = first.create_shared_population(8, seed=I6N_SEED)
    evaluated = [dict(w, fitness=0.7 + i / 100) for i, w in enumerate(state["population"])]
    args = (state["innovation_tracker"], state["stage_schedule"], state["param_defaults"], 0, 0)
    ledger = first._i6n_ledger()

    def down(*a, **k):
        raise ConnectionError("ledger unreachable")
    monkeypatch.setattr(ledger, "put_search_state", down)
    with pytest.raises(ConnectionError):
        first.reproduce_shared(evaluated, 0, 1, *args)
    assert L.Ledger(ledger_path).search_state("search:h12")["generation"] == 0
    restarted = configured(config)
    a = restarted.reproduce_shared(evaluated, 0, 2, *args)
    b = configured(config).reproduce_shared(evaluated, 0, 3, *args)
    assert a["population"] == b["population"]
    stored = L.Ledger(ledger_path).search_state("search:h12")
    assert stored["generation"] == 1 and stored["state"]["population"] == a["population"]
    assert os.listdir(counts) == []  # reproduction never fits
