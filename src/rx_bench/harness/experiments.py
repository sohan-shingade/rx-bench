"""Agent versioning and the experiment registry behind the bench dashboard.

The dashboard's whole premise is that a scorecard is meaningless unless you
know *which agent* produced it. "The agent" here is not a checkpoint file —
it is a system prompt plus tool implementations plus a model id. So the
version id is a content hash over exactly the things that change the agent's
behaviour:

  * ``policy.md`` — the system prompt;
  * every ``*.py`` in the domain source dir — the nineteen tools, the
    environment wiring, the flight recorder;
  * the agent model id.

Deliberately *not* hashed: ``db.json`` and the case files. Those are the exam,
not the student — editing a test case must not masquerade as a new agent.

Each benchmark run is recorded as one JSON file under
``results/experiments/``, holding the version that ran, the split, and
the scorecard summary. The registry is append-only flat files so it survives
crashes, diffs cleanly, and needs no database.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from rx_bench.harness.common import (
    DOMAIN_DATA_DIR,
    POLICY_PATH,
    PROJECT_ROOT,
    TAU2_ROOT,
)

#: Where the domain's executable behaviour lives (tools, environment, recorder).
DOMAIN_SRC_DIR = Path(__file__).resolve().parent.parent / "domain"

#: One JSON file per recorded benchmark run.
EXPERIMENTS_DIR = PROJECT_ROOT / "results" / "experiments"

#: Splits that are the out-of-sample final exam rather than the iteration set.
OOS_SPLITS = frozenset({"oos"})

VERSION_ID_LEN = 12


# ---------------------------------------------------------------------------
# Version identity
# ---------------------------------------------------------------------------


def _sha256_file(path: Path) -> Optional[str]:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None


def agent_files(
    policy_path: Path = POLICY_PATH, src_dir: Path = DOMAIN_SRC_DIR
) -> list[Path]:
    """The files whose content defines the agent, in a stable order."""
    files = [policy_path]
    if src_dir.exists():
        files.extend(sorted(p for p in src_dir.glob("*.py") if p.is_file()))
    return files


def agent_version(
    model: str,
    policy_path: Path = POLICY_PATH,
    src_dir: Path = DOMAIN_SRC_DIR,
) -> dict[str, Any]:
    """Identify the current agent: a short content hash plus its ingredients.

    The per-file hashes are kept alongside the combined id so the dashboard can
    say *what* changed between two versions ("policy.md and tools.py"), not
    just that something did. A missing file hashes as the literal string
    ``missing`` rather than being skipped — an agent that lost its policy file
    is a different agent, not the same one.
    """
    files: dict[str, str] = {}
    for path in agent_files(policy_path, src_dir):
        digest = _sha256_file(path)
        files[path.name] = digest[:VERSION_ID_LEN] if digest else "missing"

    combined = hashlib.sha256()
    combined.update(f"model={model}\n".encode())
    for name in sorted(files):
        combined.update(f"{name}={files[name]}\n".encode())
    return {
        "id": combined.hexdigest()[:VERSION_ID_LEN],
        "model": model,
        "files": files,
    }


def changed_files(current: dict[str, Any], previous: dict[str, Any]) -> list[str]:
    """Which agent files differ between two version records."""
    cur = current.get("files") or {}
    prev = previous.get("files") or {}
    out = sorted(name for name in set(cur) | set(prev) if cur.get(name) != prev.get(name))
    if current.get("model") != previous.get("model"):
        out.append("model")
    return out


# ---------------------------------------------------------------------------
# Run naming
# ---------------------------------------------------------------------------

_RUN_NAME_SAFE = re.compile(r"[^A-Za-z0-9_.-]+")


def run_name_for(version_id: str, split: str, existing: set[str]) -> str:
    """Pick tau2's ``--save-to`` name for a benchmark run.

    The base name is deterministic in (version, split), *on purpose*: a run
    that was killed partway keeps its directory, and relaunching the same
    version on the same split resumes it via ``--auto-resume`` instead of
    starting over. Only once a run has been *recorded* does the same
    version+split get a fresh ``_r2`` suffix, so an intentional repeat (a
    variance check) is a new run rather than a no-op resume of a finished one.
    """
    base = _RUN_NAME_SAFE.sub("_", f"exp_{version_id}_{split}")
    if base not in existing:
        return base
    n = 2
    while f"{base}_r{n}" in existing:
        n += 1
    return f"{base}_r{n}"


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def _write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as fh:
            json.dump(payload, fh, indent=2, ensure_ascii=False, default=str)
            fh.write("\n")
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def summarize_scorecard(scorecard: dict[str, Any]) -> dict[str, Any]:
    """The slice of a scorecard the dashboard charts. Small on purpose —
    the full scorecard stays next to the raw results; this is the index."""
    totals = scorecard.get("totals") or {}
    per_suite = {
        name: {
            "cases": bucket.get("cases"),
            "passed": bucket.get("passed"),
            "pass_rate": bucket.get("pass_rate"),
        }
        for name, bucket in (scorecard.get("per_suite") or {}).items()
    }
    return {
        "pass_rate": totals.get("pass_rate"),
        "cases": totals.get("cases"),
        "passed": totals.get("passed"),
        "cases_not_scored": totals.get("cases_not_scored"),
        "per_suite": per_suite,
        "headline": scorecard.get("headline") or {},
    }


def record_run(
    *,
    version: dict[str, Any],
    split: str,
    trials: int,
    run_name: str,
    results_path: Path,
    scorecard: dict[str, Any],
    user_llm: Optional[str] = None,
    experiments_dir: Path = EXPERIMENTS_DIR,
    now: Optional[datetime] = None,
) -> Path:
    """Append one finished benchmark run to the registry. Returns its path."""
    now = now or datetime.now(timezone.utc)
    version = dict(version)
    version["files"] = dict(version.get("files") or {})
    active_policy_sha = (scorecard.get("run") or {}).get("policy_sha256")
    if active_policy_sha:
        version["files"]["policy.md"] = active_policy_sha[:VERSION_ID_LEN]
        version["policy_sha256"] = active_policy_sha
        combined = hashlib.sha256()
        combined.update(f"model={version.get('model')}\n".encode())
        for name in sorted(version["files"]):
            combined.update(f"{name}={version['files'][name]}\n".encode())
        version["id"] = combined.hexdigest()[:VERSION_ID_LEN]
    record = {
        "recorded_at": now.isoformat(timespec="seconds"),
        "version": version,
        "split": split,
        "oos": split in OOS_SPLITS,
        "trials": trials,
        "user_llm": user_llm,
        "run_name": run_name,
        "results_path": str(results_path),
        "summary": summarize_scorecard(scorecard),
    }
    stamp = now.strftime("%Y%m%dT%H%M%S")
    path = experiments_dir / f"{stamp}_{version['id']}_{split}.json"
    # A second record in the same UTC second gets a suffix instead of clobbering.
    n = 2
    while path.exists():
        path = experiments_dir / f"{stamp}_{version['id']}_{split}_{n}.json"
        n += 1
    _write_json_atomic(path, record)
    return path


def load_experiments(experiments_dir: Path = EXPERIMENTS_DIR) -> list[dict[str, Any]]:
    """Every recorded run, oldest first. Unreadable files are reported inline
    rather than dropped — a corrupt record is a fact about the registry."""
    out: list[dict[str, Any]] = []
    if not experiments_dir.exists():
        return out
    for path in sorted(experiments_dir.glob("*.json")):
        try:
            record = json.loads(path.read_text())
        except (OSError, ValueError) as exc:
            record = {"recorded_at": None, "error": f"{path.name}: {exc}"}
        record["registry_file"] = path.name
        out.append(record)
    out.sort(key=lambda r: r.get("recorded_at") or "")
    return out


def oos_run_count(
    experiments: list[dict[str, Any]], version_id: Optional[str] = None
) -> int:
    """How many times the out-of-sample split has been run — the peek counter.

    Restricted to one version when given. If this number climbs past 1 for a
    version, the "held-out" set is being iterated on and its result is no
    longer out of sample in any honest sense.
    """
    count = 0
    for record in experiments:
        if not record.get("oos"):
            continue
        if version_id and (record.get("version") or {}).get("id") != version_id:
            continue
        count += 1
    return count


def domain_split_counts() -> dict[str, int]:
    """Split name -> case count, from the domain's generated split file."""
    splits_path = DOMAIN_DATA_DIR / "split_tasks.json"
    try:
        splits = json.loads(splits_path.read_text())
    except (OSError, ValueError):
        return {}
    return {name: len(ids) for name, ids in splits.items() if isinstance(ids, list)}
