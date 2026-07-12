"""Small runtime bridge from DOIN plugins to the installed agent-multi repo.

This module deliberately owns no optimization algorithm.  It resolves the
canonical experiment JSON, loads agent-multi's existing entry-point plugins,
and exposes the same local pipeline that ``agent-multi --load_config`` uses.
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path
from typing import Any


class AgentMultiRuntime:
    """Resolve an agent-multi installation and build local components."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.doin_config = copy.deepcopy(config)
        self.root = self._resolve_root(config.get("agent_multi_root"))
        root_text = str(self.root)
        if root_text not in sys.path:
            sys.path.insert(0, root_text)
        self._reject_app_namespace_collision()

        from app.canonical_config import load_json_object, resolve_config
        from app.config import DEFAULT_VALUES

        file_config = None
        config_file = config.get("load_config") or config.get("agent_multi_config")
        if config_file:
            path = Path(config_file).expanduser()
            if not path.is_absolute():
                path = self.root / path
            file_config = load_json_object(path)
            self.config_path = path.resolve()
        else:
            self.config_path = None

        # DOIN-only keys must not silently become agent-multi experiment keys.
        cli_overrides = {
            key: value
            for key, value in config.items()
            if key not in {
                "agent_multi_root", "load_config", "agent_multi_config",
                "optimization_callbacks", "initial_candidate_params", "runtime_overlay",
                "current_best_performance", "node_seed_offset",
            }
        }
        resolution = resolve_config(DEFAULT_VALUES, file_config=file_config, cli_overrides=cli_overrides)
        self.resolution = resolution
        self.runtime_manifest = None
        overlay_file = config.get("runtime_overlay")
        if overlay_file:
            overlay_path = Path(overlay_file).expanduser()
            if not overlay_path.is_absolute():
                overlay_path = self.root / overlay_path
            from app.runtime_overlay import resolve_runtime_overlay

            overlay_payload = load_json_object(overlay_path)
            runtime_resolution = resolve_runtime_overlay(
                resolution.runtime,
                overlay_payload=overlay_payload,
                overlay_base_dir=overlay_path.parent,
                expected_repositories=resolution.canonical.code.get("repositories", {}),
            )
            self.runtime_config = runtime_resolution.runtime
            self.runtime_manifest = runtime_resolution.manifest
        else:
            self.runtime_config = resolution.runtime
        self.runtime_config["agent_multi_root"] = str(self.root)
        self.config_hash = resolution.canonical.canonical_hash

    @staticmethod
    def _resolve_root(value: str | None) -> Path:
        candidates = []
        if value:
            candidates.append(Path(value).expanduser())
        candidates.extend([
            Path.home() / "Documents/GitHub/agent-multi",
            Path.home() / "agent-multi",
            Path.cwd() / "agent-multi",
            Path("/home/openclaw/agent-multi"),
        ])
        for candidate in candidates:
            if (candidate / "setup.py").exists() and (candidate / "app").is_dir():
                return candidate.resolve()
        raise FileNotFoundError(
            "Cannot find agent-multi. Set 'agent_multi_root' to the installed "
            "repository containing setup.py."
        )

    def _reject_app_namespace_collision(self) -> None:
        """Fail closed if another repository already owns the generic ``app`` package."""
        loaded = sys.modules.get("app")
        loaded_file = getattr(loaded, "__file__", None) if loaded else None
        if loaded_file and not Path(loaded_file).resolve().is_relative_to(self.root / "app"):
            raise RuntimeError(
                "agent-multi cannot be loaded because Python module namespace "
                f"'app' is already owned by {loaded_file!r}; start a fresh DOIN "
                "process or isolate the repositories in separate workers"
            )

    def build_components(self, overrides: dict[str, Any] | None = None):
        """Build fresh env/agent/pipeline instances for one local run."""
        from app.plugin_loader import load_plugin

        config = copy.deepcopy(self.runtime_config)
        if overrides:
            config.update(_copy_runtime_overrides(overrides))

        def load(group: str, key: str):
            name = config.get(key)
            if not name:
                raise ValueError(f"agent-multi config is missing {key!r}")
            klass, _ = load_plugin(group, name)
            instance = klass(config)
            instance.set_params(**config)
            return instance

        return (
            load("env.plugins", "env_plugin"),
            load("agent.plugins", "agent_plugin"),
            load("pipeline.plugins", "pipeline_plugin"),
            config,
        )

    def run(self, overrides: dict[str, Any] | None = None, *, mode: str) -> dict[str, Any]:
        env, agent, pipeline, config = self.build_components(overrides)
        return pipeline.run_pipeline(config=config, env_plugin=env, agent_plugin=agent, mode=mode)

    def load_local_optimizer(self, config: dict[str, Any]):
        from app.plugin_loader import load_plugin

        name = config.get("optimizer_plugin")
        if not name:
            raise ValueError("agent-multi config is missing 'optimizer_plugin'")
        klass, _ = load_plugin("optimizer.plugins", name)
        optimizer = klass()
        optimizer.set_params(**config)
        return optimizer


def _copy_runtime_overrides(overrides: dict[str, Any]) -> dict[str, Any]:
    """Copy configuration data while preserving process-local callbacks.

    Bound DOIN callbacks close over locks and events, so they are intentionally
    not serializable configuration. The callback mapping is shallow-copied;
    all declarative values remain deeply copied.
    """
    copied: dict[str, Any] = {}
    for key, value in overrides.items():
        if key == "optimization_callbacks" and isinstance(value, dict):
            copied[key] = dict(value)
        else:
            copied[key] = copy.deepcopy(value)
    return copied
