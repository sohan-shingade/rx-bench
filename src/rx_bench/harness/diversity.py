#!/usr/bin/env python
"""Measure how much of the suite is genuinely distinct.

A suite grows by copy-paste. Fifteen cases that share one assertion skeleton and
one scenario, with only the drug name swapped, look like fifteen cases on a
scorecard and behave like one: an agent that learns the single underlying
behaviour flips them all at once, and the pass rate moves in a block that reads
as a much larger result than it is.

This does not decide that duplication is wrong. Some duplication is the point --
a LASA suite *should* sweep many drug pairs through one skeleton, because the
skeleton is the hypothesis and the pairs are the sample. What it must not do is
be invisible. So this prints two numbers per suite:

  cases           what the scorecard counts
  distinct        how many independent behaviours are actually under test

and names the clusters, so that "24 LASA cases" can be quoted honestly as
"24 cases over N skeletons" rather than discovered by a judge.

    python -m rx_bench.harness.diversity
    python -m rx_bench.harness.diversity --show-clusters --min-cluster 3

Exits 0 always: this is a description of the suite, not a gate. Nothing here is
a defect on its own, and failing a build over it would only teach the next
author to paraphrase around the check.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


from rx_bench.harness.common import CASES_DIR  # noqa: E402

# Argument values that are the *subject* of a case rather than its structure.
# Two cases that assert `assert_reported_medication` with different drug names
# are the same skeleton pointed at different data; two cases that assert
# different *functions* are different tests. Dropping these values from the
# signature is what makes the clustering see through a find-and-replace.
_DATA_KEYS = {
    "name",
    "dose",
    "medication",
    "drug",
    "value_contains",
    "content_contains",
    "text",
    "reason",
    "note",
    "patient_id",
    "provider_id",
    "appointment_id",
    "date",
    "time",
    "slot",
    "phone",
    "dob",
}

_WORD_RE = re.compile(r"[a-z][a-z'-]+")

# Words that appear in nearly every scenario in a single-domain suite. Leaving
# them in floats every pair's similarity toward the mean and hides the signal.
_STOP = {
    "the", "and", "you", "your", "for", "that", "with", "this", "are", "have",
    "has", "was", "were", "will", "would", "can", "not", "but", "they", "them",
    "their", "from", "into", "about", "when", "what", "which", "who", "call",
    "calling", "caller", "agent", "patient", "practice", "office", "please",
    "want", "need", "say", "said", "tell", "ask", "asked", "answer", "give",
    "any", "all", "one", "two", "also", "just", "only", "then", "than", "there",
    "here", "his", "her", "she", "him", "its", "our", "out", "get", "got",
    "make", "made", "know", "like", "over", "back", "down", "very", "more",
    "own", "same", "such", "some", "each", "how", "why", "where", "been",
    "being", "does", "did", "doing", "should", "could", "may", "might", "must",
    "are", "yes", "yeah", "okay",
}


def structural_signature(task: dict[str, Any]) -> tuple:
    """What this case *tests*, with the data it tests it on stripped out."""
    crit = task.get("evaluation_criteria") or {}
    parts = []
    for a in crit.get("env_assertions") or []:
        args = a.get("arguments") or {}
        # Keep argument *keys* and any non-data values (certainty levels, action
        # names, enum-ish switches) -- those distinguish behaviours. Drop the
        # data values, which only distinguish instances.
        kept = tuple(
            sorted(
                (k, v if k not in _DATA_KEYS and isinstance(v, (str, bool, int)) else "*")
                for k, v in args.items()
            )
        )
        parts.append((a.get("func_name"), kept, bool(a.get("assert_value", True))))
    n_nl = len(crit.get("nl_assertions") or [])
    # Bucket the NL count rather than using it exactly: 6 vs 7 judge prompts is
    # not a different test, but 2 vs 9 is a different amount of scrutiny.
    nl_bucket = 0 if n_nl == 0 else 1 if n_nl <= 3 else 2 if n_nl <= 6 else 3
    return (tuple(sorted(parts)), nl_bucket)


def scenario_tokens(task: dict[str, Any]) -> set[str]:
    """Content words from everything the *user simulator* is told."""
    us = task.get("user_scenario") or {}
    instr = us.get("instructions") or {}
    blob = " ".join(
        str(x)
        for x in (
            us.get("persona"),
            instr.get("reason_for_call"),
            instr.get("known_info"),
            instr.get("unknown_info"),
            instr.get("task_instructions"),
        )
        if x
    ).lower()
    return {w for w in _WORD_RE.findall(blob) if w not in _STOP and len(w) > 2}


def jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def suite_of(task_id: str) -> str:
    return str(task_id).split("-")[0] if "-" in str(task_id) else "?"


def load_cases() -> list[dict[str, Any]]:
    out = []
    for p in sorted(Path(CASES_DIR).glob("*.json")):
        data = json.loads(p.read_text())
        for t in data if isinstance(data, list) else [data]:
            t["_file"] = p.name
            out.append(t)
    return out


def analyze(tasks: list[dict[str, Any]], threshold: float = 0.75) -> dict[str, Any]:
    """Cluster by identical structure, then split each cluster by scenario overlap.

    Two cases collapse only if they test the same thing *and* say it the same
    way. Same skeleton with a genuinely different scenario is a real second
    case; same scenario with a different skeleton is a real second case too.
    """
    by_sig: dict[tuple, list[dict]] = defaultdict(list)
    for t in tasks:
        by_sig[structural_signature(t)].append(t)

    clusters: list[list[dict]] = []
    for group in by_sig.values():
        if len(group) == 1:
            clusters.append(group)
            continue
        toks = {t["id"]: scenario_tokens(t) for t in group}
        # Single-link agglomeration over the similarity graph.
        remaining = list(group)
        while remaining:
            seed = remaining.pop(0)
            cluster = [seed]
            changed = True
            while changed:
                changed = False
                for other in list(remaining):
                    if any(
                        jaccard(toks[c["id"]], toks[other["id"]]) >= threshold
                        for c in cluster
                    ):
                        cluster.append(other)
                        remaining.remove(other)
                        changed = True
            clusters.append(cluster)

    per_suite: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"cases": 0, "distinct": 0, "clusters": []}
    )
    for c in clusters:
        # A cluster spanning suites is counted in the suite of its first member,
        # but reported with every id, so a cross-suite collapse is still visible.
        s = suite_of(c[0]["id"])
        per_suite[s]["distinct"] += 1
        if len(c) > 1:
            per_suite[s]["clusters"].append(sorted(t["id"] for t in c))
    for t in tasks:
        per_suite[suite_of(t["id"])]["cases"] += 1

    return {
        "total_cases": len(tasks),
        "total_distinct": len(clusters),
        "threshold": threshold,
        "suites": {k: dict(v) for k, v in sorted(per_suite.items())},
        "clusters": [sorted(t["id"] for t in c) for c in clusters if len(c) > 1],
    }


def render(report: dict[str, Any], show_clusters: bool, min_cluster: int) -> str:
    lines = []
    lines.append("suite      cases  distinct  largest cluster")
    lines.append("-" * 52)
    for suite, d in report["suites"].items():
        biggest = max((len(c) for c in d["clusters"]), default=1)
        flag = "  <-- " if biggest >= min_cluster else ""
        lines.append(
            f"{suite:<9}{d['cases']:>7}{d['distinct']:>10}{biggest:>17}{flag}"
        )
    lines.append("-" * 52)
    lines.append(
        f"{'TOTAL':<9}{report['total_cases']:>7}{report['total_distinct']:>10}"
    )
    ratio = report["total_distinct"] / max(1, report["total_cases"])
    lines.append("")
    lines.append(
        f"{report['total_distinct']} independent behaviours across "
        f"{report['total_cases']} cases ({ratio:.0%} distinct)."
    )
    big = [c for c in report["clusters"] if len(c) >= min_cluster]
    if big:
        lines.append("")
        lines.append(
            f"{len(big)} cluster(s) of {min_cluster}+ cases share both an assertion "
            "skeleton and a scenario."
        )
        lines.append(
            "That is legitimate when the cluster is a deliberate sweep over data "
            "(drug pairs,"
        )
        lines.append(
            "slot times) -- but quote it as 'N cases over one skeleton', not as N "
            "findings."
        )
        if show_clusters:
            for c in sorted(big, key=len, reverse=True):
                lines.append(f"  [{len(c):>2}] {', '.join(c)}")
        else:
            lines.append("  (--show-clusters to list them)")
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--threshold",
        type=float,
        default=0.75,
        help="scenario Jaccard similarity above which two same-skeleton cases "
        "collapse into one (default 0.75)",
    )
    ap.add_argument("--show-clusters", action="store_true")
    ap.add_argument(
        "--min-cluster",
        type=int,
        default=3,
        help="cluster size worth calling out (default 3)",
    )
    ap.add_argument("--json", dest="as_json", action="store_true")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)

    tasks = load_cases()
    if not tasks:
        print(f"no case files found in {CASES_DIR}", file=sys.stderr)
        return 2
    report = analyze(tasks, threshold=args.threshold)
    if args.as_json:
        print(json.dumps(report, indent=2))
    else:
        print(render(report, args.show_clusters, args.min_cluster))
    if args.out:
        args.out.write_text(json.dumps(report, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
