import os
from pathlib import Path
from typing import Optional

from tau2.data_model.tasks import Task
from rx_bench.domain.data_model import MedicalReceptionDB
from rx_bench.domain.tools import MedicalReceptionTools
from rx_bench.domain.utils import (
    MEDICAL_DB_PATH,
    MEDICAL_MUTANT_POLICY_DIR,
    MEDICAL_POLICY_PATH,
    MEDICAL_TASK_SET_PATH,
)
from tau2.environment.environment import Environment
from tau2.utils import load_file


def _resolve_policy_path() -> Path:
    """Return the policy file to load.

    Setting MEDICAL_POLICY_MUTANT=<name> swaps in data/.../mutants/<name>.md.
    This is how the suite is mutation-tested: a deliberately weakened policy
    should make specific safety cases fail. If every case still passes under a
    mutant, the suite is not actually measuring the thing it claims to measure.
    """
    mutant = os.environ.get("MEDICAL_POLICY_MUTANT")
    if not mutant:
        return MEDICAL_POLICY_PATH
    path = MEDICAL_MUTANT_POLICY_DIR / f"{mutant}.md"
    if not path.exists():
        raise FileNotFoundError(
            f"MEDICAL_POLICY_MUTANT={mutant} but {path} does not exist"
        )
    return path


def get_environment(
    db: Optional[MedicalReceptionDB] = None,
    solo_mode: bool = False,
) -> Environment:
    if solo_mode:
        raise ValueError("Solo mode is not supported for the medical_reception domain")
    if db is None:
        db = MedicalReceptionDB.load(MEDICAL_DB_PATH)
    else:
        db = db.model_copy(deep=True)
    tools = MedicalReceptionTools(db)
    with open(_resolve_policy_path(), "r") as fp:
        policy = fp.read()
    return Environment(
        domain_name="medical_reception",
        policy=policy,
        tools=tools,
    )


def get_tasks(task_split_name: Optional[str] = None) -> list[Task]:
    tasks = load_file(MEDICAL_TASK_SET_PATH)
    tasks = [Task.model_validate(task) for task in tasks]
    if task_split_name is None:
        return tasks
    task_splits = get_tasks_split()
    if task_split_name not in task_splits:
        raise ValueError(
            f"Invalid task split name: {task_split_name}. "
            f"Valid splits are: {list(task_splits.keys())}"
        )
    return [task for task in tasks if task.id in task_splits[task_split_name]]


def get_tasks_split() -> dict[str, list[str]]:
    split_file = (
        Path(MEDICAL_TASK_SET_PATH).parent
        / f"split_{Path(MEDICAL_TASK_SET_PATH).stem}.json"
    )
    return load_file(split_file)
