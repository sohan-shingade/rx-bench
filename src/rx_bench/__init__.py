"""rx-bench: a benchmark for voice-agent-patient interaction in medical
front-office workflows, built on top of tau2-bench.

Importing this package has two deliberate side effects:

1. tau2's environment-sensitive defaults are patched from environment
   variables (``TAU2_LLM_NL_ASSERTIONS``, ``TAU2_LLM_ENV_INTERFACE``, the
   ``TAU2_MAX_RETRIES`` / ``TAU2_RETRY_*`` knobs). Upstream tau2 hardcodes
   these as constants; several tau2 modules bind them at import time, so the
   patch must land before those modules are imported. That is why anything
   that uses tau2 through this benchmark must ``import rx_bench`` FIRST (the
   ``rxbench`` CLI does this for you).

2. The ``medical_reception`` domain and its task set are registered with the
   tau2 registry singleton, so ``tau2 run --domain medical_reception ...``
   (via ``rxbench run ...``) works exactly like a built-in domain.

It also points ``TAU2_DATA_DIR`` at this repository's ``data/`` directory
(if the variable is unset), so simulation outputs land in
``data/simulations/`` here rather than inside the installed tau2 package.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

# tau2 normally loads this during its eager package import. Load it first so the
# config patch below can see the same values before any tau2 consumer binds them.
load_dotenv()

# ---------------------------------------------------------------------------
# Naming. The project name is provisional; keep it in this one place.
# ---------------------------------------------------------------------------

#: ASCII project name (also the repository and PyPI-style name).
NAME = "rx-bench"
#: Display name used in docs and reports.
DISPLAY_NAME = "℞-bench"  # ℞-bench

__version__ = "0.1.0"

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DATA_ROOT = _REPO_ROOT / "data"

# Must happen before the first ``import tau2.utils`` anywhere: tau2 resolves
# its DATA_DIR from this variable at import time. Only set it when the repo
# data directory actually exists (an installed-wheel layout without the repo
# checkout should fall back to tau2's own resolution / an explicit env var).
if "TAU2_DATA_DIR" not in os.environ and _DATA_ROOT.is_dir():
    os.environ["TAU2_DATA_DIR"] = str(_DATA_ROOT)


def _load_tau2_config():
    """Load ``tau2.config`` without executing tau2's eager package import."""
    if "tau2.config" in sys.modules:
        return sys.modules["tau2.config"]

    tau2_spec = importlib.util.find_spec("tau2")
    if tau2_spec is None or not tau2_spec.submodule_search_locations:
        raise ImportError("Could not find tau2")
    config_path = Path(next(iter(tau2_spec.submodule_search_locations))) / "config.py"
    config_spec = importlib.util.spec_from_file_location("tau2.config", config_path)
    if config_spec is None or config_spec.loader is None:
        raise ImportError("Could not load tau2.config")
    config = importlib.util.module_from_spec(config_spec)
    sys.modules["tau2.config"] = config
    config_spec.loader.exec_module(config)
    return config


def _patch_tau2_config() -> None:
    """Make tau2's hardcoded defaults overridable from the environment.

    Importing ``tau2.config`` normally executes ``tau2.__init__`` first. That
    eagerly imports the evaluator and retry modules, which bind these constants
    before control returns. Load the standalone config module directly instead,
    patch it, and only then let tau2's package import proceed.
    """
    already = [
        m
        for m in (
            "tau2.evaluator.evaluator_nl_assertions",
            "tau2.environment.utils.interface_agent",
            "tau2.utils.retry",
            "tau2.utils.llm_utils",
        )
        if m in sys.modules
    ]
    config = _load_tau2_config()
    env = os.environ

    overrides = {
        "DEFAULT_LLM_NL_ASSERTIONS": ("TAU2_LLM_NL_ASSERTIONS", str),
        "DEFAULT_LLM_ENV_INTERFACE": ("TAU2_LLM_ENV_INTERFACE", str),
        "DEFAULT_MAX_RETRIES": ("TAU2_MAX_RETRIES", int),
        "DEFAULT_RETRY_ATTEMPTS": ("TAU2_RETRY_ATTEMPTS", int),
        "DEFAULT_RETRY_MIN_WAIT": ("TAU2_RETRY_MIN_WAIT", float),
        "DEFAULT_RETRY_MAX_WAIT": ("TAU2_RETRY_MAX_WAIT", float),
        "DEFAULT_RETRY_MULTIPLIER": ("TAU2_RETRY_MULTIPLIER", float),
    }
    for attr, (env_name, convert) in overrides.items():
        value = env.get(env_name)
        if value:
            setattr(config, attr, convert(value))

    if already and any(env.get(env_name) for env_name, _ in overrides.values()):
        import warnings

        warnings.warn(
            f"rx_bench was imported after {already}; TAU2_* overrides may not "
            "affect values those modules already bound. Import rx_bench first "
            "(or use the rxbench CLI).",
            RuntimeWarning,
            stacklevel=2,
        )


def register() -> None:
    """Register the medical_reception domain with the tau2 registry.

    Idempotent: safe to call more than once, and safe when a fork of tau2
    already ships the domain.
    """
    _patch_tau2_config()

    from rx_bench.voice_native import tau2_patches as _voice_patches

    _voice_patches.apply()

    from tau2.registry import registry

    from rx_bench.domain.environment import (
        get_environment,
        get_tasks,
        get_tasks_split,
    )

    if "medical_reception" not in registry.get_domains():
        registry.register_domain(get_environment, "medical_reception")
        registry.register_tasks(
            get_tasks,
            "medical_reception",
            get_task_splits=get_tasks_split,
        )


register()
