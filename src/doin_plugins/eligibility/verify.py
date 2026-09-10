"""Verify — never select, never promote."""
from __future__ import annotations

from doin_plugins.eligibility.adapter import load_gate


def verify_identity(config: dict, *, subject_id: str,
                    version: str, code_digest: str,
                    scope: str, subject_kind: str = "operator",
                    evidence_digest: str | None = None) -> dict:
    """Confirm that a reviewed manifest entry covers exactly this
    id, version and code bytes.

    Returns the reviewed entry's identity fields only. It
    deliberately returns no ranking, no score and no candidate
    list, so a caller cannot use verification as selection.
    """
    gate_mod, _ = load_gate(config)
    manifest = gate_mod.load_manifest(
        config["eligibility_manifest"],
        expected_sha256=config.get("eligibility_manifest_sha256"),
        max_age_days=config.get("eligibility_max_age_days"))
    entry = gate_mod.require_eligible(
        manifest, subject_id, scope=scope,
        subject_kind=subject_kind, version=version,
        code_digest=code_digest,
        evidence_digest=evidence_digest)
    return {
        "verified": True,
        "subject_id": entry["subject_id"],
        "version": entry["version"],
        "fit_scope": entry["fit_scope"],
        "decision_scope": entry["decision_scope"],
        "manifest_sha256": gate_mod.manifest_fingerprint(manifest),
        "promotion": "NONE — verification confers no eligibility "
                     "and no ranking",
    }
