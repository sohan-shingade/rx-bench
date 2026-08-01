"""Path resolution for the medical_reception domain data.

The benchmark data ships in the repository under ``data/<DATA_VERSION>/`` with
this layout (flat, not the upstream ``data/tau2/domains/<domain>/`` nesting):

    data/v1/
        db.json
        policy.md
        tasks.json
        split_tasks.json
        cases/*.json
        mutants/*.md

Resolution order for the domain data directory:

1. ``RX_BENCH_DATA_DIR`` environment variable, if set (must point at the
   directory that directly contains ``db.json``);
2. ``<repository root>/data/<DATA_VERSION>`` for a source checkout (the
   supported installation mode is an editable install / ``uv sync`` of the
   cloned repository).
"""

import os
from pathlib import Path

#: Data schema/content version. Results are NOT comparable across versions.
DATA_VERSION = "v1"


def _default_data_dir() -> Path:
    # src/rx_bench/domain/utils.py -> parents[3] == repository root
    return Path(__file__).resolve().parents[3] / "data" / DATA_VERSION


MEDICAL_DATA_DIR = Path(
    os.environ.get("RX_BENCH_DATA_DIR") or _default_data_dir()
)
MEDICAL_DB_PATH = MEDICAL_DATA_DIR / "db.json"
MEDICAL_POLICY_PATH = MEDICAL_DATA_DIR / "policy.md"
MEDICAL_TASK_SET_PATH = MEDICAL_DATA_DIR / "tasks.json"

# Mutated policies for mutation-testing the suite (see rx_bench.harness.mutate)
MEDICAL_MUTANT_POLICY_DIR = MEDICAL_DATA_DIR / "mutants"
