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

import os
import sys
from pathlib import Path

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


def _patch_tau2_config() -> None:
    """Make tau2's hardcoded defaults overridable from the environment.

    Two classes of knob, both invisible to the tau2 CLI:

    * ``TAU2_LLM_NL_ASSERTIONS`` / ``TAU2_LLM_ENV_INTERFACE`` — the
      NL-assertion judge and env-interface models. Left alone they default to
      an OpenAI model, so a run driven entirely through another provider
      completes every simulation and only then dies at grading time on a
      missing OPENAI_API_KEY — after the tokens are spent.
    * ``TAU2_MAX_RETRIES`` / ``TAU2_RETRY_ATTEMPTS`` / ``TAU2_RETRY_MIN_WAIT``
      / ``TAU2_RETRY_MAX_WAIT`` / ``TAU2_RETRY_MULTIPLIER`` — retry behaviour.
      Note the first two MULTIPLY (litellm's inner retries x tau2's tenacity
      wrapper); keep the inner count small and let the outer one back off.

    Values are rebound on ``tau2.config`` before the tau2 modules that read
    them at import time (``tau2.utils.retry``, ``tau2.utils.llm_utils``,
    ``tau2.evaluator.evaluator_nl_assertions``, ...) are imported. If any of
    those modules is already imported, the patch cannot reach it and we warn
    rather than mislead.
    """
    already = [
        m
        for m in ("tau2.utils.retry", "tau2.utils.llm_utils", "tau2.registry")
        if m in sys.modules
    ]
    import tau2.config as _cfg

    env = os.environ
    _cfg.DEFAULT_LLM_NL_ASSERTIONS = env.get(
        "TAU2_LLM_NL_ASSERTIONS", _cfg.DEFAULT_LLM_NL_ASSERTIONS
    )
    _cfg.DEFAULT_LLM_ENV_INTERFACE = env.get(
        "TAU2_LLM_ENV_INTERFACE", _cfg.DEFAULT_LLM_ENV_INTERFACE
    )
    _cfg.DEFAULT_MAX_RETRIES = int(
        env.get("TAU2_MAX_RETRIES", _cfg.DEFAULT_MAX_RETRIES)
    )
    _cfg.DEFAULT_RETRY_ATTEMPTS = int(
        env.get("TAU2_RETRY_ATTEMPTS", _cfg.DEFAULT_RETRY_ATTEMPTS)
    )
    _cfg.DEFAULT_RETRY_MIN_WAIT = float(
        env.get("TAU2_RETRY_MIN_WAIT", _cfg.DEFAULT_RETRY_MIN_WAIT)
    )
    _cfg.DEFAULT_RETRY_MAX_WAIT = float(
        env.get("TAU2_RETRY_MAX_WAIT", _cfg.DEFAULT_RETRY_MAX_WAIT)
    )
    _cfg.DEFAULT_RETRY_MULTIPLIER = float(
        env.get("TAU2_RETRY_MULTIPLIER", _cfg.DEFAULT_RETRY_MULTIPLIER)
    )

    if already and any(
        env.get(v)
        for v in (
            "TAU2_LLM_NL_ASSERTIONS",
            "TAU2_LLM_ENV_INTERFACE",
            "TAU2_MAX_RETRIES",
            "TAU2_RETRY_ATTEMPTS",
            "TAU2_RETRY_MIN_WAIT",
            "TAU2_RETRY_MAX_WAIT",
            "TAU2_RETRY_MULTIPLIER",
        )
    ):
        import warnings

        warnings.warn(
            f"rx_bench was imported after {already}; TAU2_* overrides may not "
            "take effect for values those modules bound at import time. "
            "Import rx_bench before any tau2 module (or use the rxbench CLI).",
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
