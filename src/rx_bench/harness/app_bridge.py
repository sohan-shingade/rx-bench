#!/usr/bin/env python
"""Drive the TypeScript safety gateway with the tau2 LASA cases.

The tau2 suite and the gateway test *different objects*. tau2 drives a
conversational agent with nineteen tools against ``policy.md``; the gateway is a
deterministic pipeline with no agent loop, one fixture, and a two-turn
confirmation handshake. Nothing in tau2 can be pointed at the gateway directly.

What *does* transfer is the case data. Every S1 case carries a machine-readable
``transcript_injection`` block --- the corrupted medication token, its word
confidence, and the ground truth --- and encodes the right answer as
``assert_reported_medication`` and the dangerous wrong answer as
``assert_not_reported_medication``. That is exactly the input the gateway takes
and exactly the label needed to score its FHIR writes, so the same case file can
grade both architectures and the two numbers are about the same scenario.

This module turns each case into a ``FinalTranscript``, drives
``npm run eval:app`` through the full handshake, and scores the resulting
``MedicationStatement``.

    python -m rx_bench.harness.app_bridge --root <gateway checkout> --out results/app_lasa.json

Note: the TypeScript gateway itself is a separate application and is NOT
shipped in this repository; ``--root`` must point at a checkout providing
``npm run eval:app``. Everything else here (case -> FinalTranscript
conversion, verdict scoring) is self-contained and unit-tested offline.

Coverage is the first thing the report prints, and it is not optional. The
gateway recognises exactly one confusion pair; the other eleven pairs in the
suite produce no candidate, no decision and no action --- silence, not a refusal.
A case the gateway cannot see is ``unsupported``, never ``passed`` and never
``failed``, because scoring silence as either would be the same move this
project exists to catch. The headline accuracy is computed over supported cases
only and is reported alongside the count it was computed from.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional

from rx_bench.harness.common import CASES_DIR, PROJECT_ROOT, safe_rate
from rx_bench.harness.scorers import canonicalize_drug

#: The gateway rejects any request whose scenarioId is not the hero fixture.
SCENARIO_ID = "synthetic-lasa-001"

#: Filled into the FinalTranscript envelope. The gateway validates these are
#: non-empty strings but does not otherwise interpret them.
TRANSCRIPT_PROVIDER = "tau2-case-injection"
TRANSCRIPT_MODEL = "s1-lasa-transcript-injection"
CAPTURED_AT = "2026-08-01T16:00:00.000Z"

#: Confidence given to every non-medication word. The medication token gets the
#: case's own ``word_confidence``; the surrounding words are deliberately high so
#: that any low-confidence signal the gateway raises is attributable to the drug
#: name rather than to the carrier phrase.
CARRIER_CONFIDENCE = 0.97

_METADATA_RE = re.compile(r"MACHINE-READABLE CASE METADATA:\s*(\{.*)$", re.S)


# ---------------------------------------------------------------------------
# Cases
# ---------------------------------------------------------------------------


@dataclass
class BridgeCase:
    """One tau2 case reduced to what the gateway can consume and be graded on."""

    task_id: str
    text: str
    medication_token: str
    word_confidence: float
    ground_truth: str
    audio_profile: Optional[str] = None
    expected_name: Optional[str] = None
    expected_dose: Optional[str] = None
    forbidden: list[str] = field(default_factory=list)

    @property
    def transcript_id(self) -> str:
        return f"tau2-{self.task_id}"


def _case_metadata(notes: str) -> Optional[dict[str, Any]]:
    match = _METADATA_RE.search(notes or "")
    if not match:
        return None
    try:
        return json.loads(match.group(1))
    except json.JSONDecodeError:
        # A malformed metadata block is a defect in the case file, but it must
        # not take the whole bridge down -- the caller sees it as a skipped case.
        return None


def _expectations(case: dict[str, Any]) -> tuple[Optional[str], Optional[str], list[str]]:
    """Expected name/dose and forbidden names, read from the case's own labels.

    Mirrors ``scorers._expected_from_task`` but reads raw case JSON rather than
    tau2 ``Task`` objects, so the bridge needs no tau2 import and runs without
    the benchmark installed.
    """
    criteria = case.get("evaluation_criteria") or {}
    assertions = criteria.get("env_assertions") or []
    expected_name: Optional[str] = None
    expected_dose: Optional[str] = None
    forbidden: list[str] = []

    for assertion in assertions:
        if assertion.get("func_name") != "assert_reported_medication":
            continue
        if not assertion.get("assert_value", True):
            continue
        args = assertion.get("arguments") or {}
        if args.get("name") and expected_name is None:
            expected_name = args["name"]
            expected_dose = args.get("dose")

    for assertion in assertions:
        func = assertion.get("func_name")
        args = assertion.get("arguments") or {}
        want = assertion.get("assert_value", True)
        name = args.get("name")
        if not name:
            continue
        if func == "assert_not_reported_medication" and want:
            forbidden.append(name)
        elif func == "assert_reported_medication" and not want:
            # A negative on the *expected* name is a wrong-dose assertion, not a
            # forbidden drug; letting it through would make the right answer
            # forbid itself.
            if canonicalize_drug(name) != canonicalize_drug(expected_name or ""):
                forbidden.append(name)

    return expected_name, expected_dose, forbidden


def load_lasa_cases(cases_dir: Path = CASES_DIR) -> list[BridgeCase]:
    """Every case carrying a ``transcript_injection`` block, in id order."""
    out: list[BridgeCase] = []
    for path in sorted(Path(cases_dir).glob("*.json")):
        raw = json.loads(path.read_text())
        cases = raw if isinstance(raw, list) else raw.get("tasks", [])
        for case in cases:
            notes = ((case.get("description") or {}).get("notes")) or ""
            meta = _case_metadata(notes)
            injection = (meta or {}).get("transcript_injection")
            if not injection or not injection.get("text"):
                continue
            expected_name, expected_dose, forbidden = _expectations(case)
            out.append(
                BridgeCase(
                    task_id=case["id"],
                    text=injection["text"],
                    medication_token=injection.get("medication_token", ""),
                    word_confidence=float(injection.get("word_confidence", 0.0)),
                    ground_truth=injection.get("ground_truth_medication", ""),
                    audio_profile=(meta or {}).get("audio_profile"),
                    expected_name=expected_name,
                    expected_dose=expected_dose,
                    forbidden=forbidden,
                )
            )
    out.sort(key=lambda case: case.task_id)
    return out


# ---------------------------------------------------------------------------
# Transcript construction
# ---------------------------------------------------------------------------


def _tokens(text: str) -> list[str]:
    """Word tokens with punctuation stripped, preserving order.

    The gateway matches its medication regex against ``transcript.text`` but
    reads confidence from ``transcript.words``, so the word list has to line up
    with the text it was supposedly transcribed from.
    """
    return [token for token in re.split(r"\s+", text.strip()) if token]


def build_transcript(case: BridgeCase) -> dict[str, Any]:
    """A schema-valid ``FinalTranscript`` for one case.

    The medication token carries the case's planted confidence; every other word
    carries ``CARRIER_CONFIDENCE``. Word timings are synthetic and evenly spaced:
    the gateway validates they are non-negative but nothing in the policy path
    reads them, and inventing plausible-looking real timings would imply an
    alignment this bridge does not have.
    """
    target = canonicalize_drug(case.medication_token)
    words = []
    for index, token in enumerate(_tokens(case.text)):
        # Strip surrounding punctuation but keep an internal decimal point, so
        # "metformin." becomes a word and "2.5" survives intact.
        bare = re.sub(r"^[^A-Za-z0-9]+|[^A-Za-z0-9]+$", "", token)
        is_medication = bool(bare) and canonicalize_drug(bare) == target and target != ""
        words.append(
            {
                "word": bare or token,
                "confidence": case.word_confidence if is_medication else CARRIER_CONFIDENCE,
                "startSeconds": round(index * 0.25, 2),
                "endSeconds": round(index * 0.25 + 0.2, 2),
            }
        )
    return {
        "schemaVersion": "1.0",
        "id": case.transcript_id,
        "text": case.text,
        "words": words,
        "provider": TRANSCRIPT_PROVIDER,
        "model": TRANSCRIPT_MODEL,
        "capturedAt": CAPTURED_AT,
    }


# ---------------------------------------------------------------------------
# Driving the gateway
# ---------------------------------------------------------------------------


class EvalAppSession:
    """A live ``npm run eval:app`` process spoken to one NDJSON line at a time.

    The handshake cannot be batched: turn two needs the ``challengeId`` that
    turn one returns. The challenge ids happen to be a deterministic counter, so
    they *could* be predicted and the whole run piped in one shot --- but a
    bridge that predicts the token it is supposed to be handed would keep
    passing after the gateway stopped issuing real ones.
    """

    def __init__(self, root: Path = PROJECT_ROOT, command: Optional[list[str]] = None):
        self.command = command or ["npm", "--silent", "run", "eval:app"]
        self.root = Path(root)
        self.proc: Optional[subprocess.Popen] = None

    def __enter__(self) -> "EvalAppSession":
        self.proc = subprocess.Popen(
            self.command,
            cwd=str(self.root),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        return self

    def __exit__(self, *_exc: Any) -> None:
        if self.proc is None:
            return
        try:
            if self.proc.stdin:
                self.proc.stdin.close()
            self.proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            self.proc.kill()

    def send(self, request: dict[str, Any]) -> dict[str, Any]:
        if self.proc is None or self.proc.stdin is None or self.proc.stdout is None:
            raise RuntimeError("session is not running")
        self.proc.stdin.write(json.dumps(request) + "\n")
        self.proc.stdin.flush()
        line = self.proc.stdout.readline()
        if not line:
            stderr = self.proc.stderr.read() if self.proc.stderr else ""
            raise RuntimeError(
                "the gateway closed stdout without answering. Run `npm run build` "
                f"first.\nstderr:\n{stderr[:2000]}"
            )
        return json.loads(line)


def _action(response: dict[str, Any], kind: str) -> Optional[dict[str, Any]]:
    for action in response.get("actions") or []:
        if action.get("type") == kind:
            return action
    return None


def _option_for_ground_truth(
    decision: dict[str, Any], ground_truth: str
) -> Optional[str]:
    """The clarification option a truthful caller would pick.

    Resolved through the decision's own ``alternatives`` rather than a hardcoded
    indication table: find the alternative whose display name is the ground
    truth, then the option whose ``candidateId`` is that alternative's code. If
    the gateway later learns a new confusion pair this keeps working, and if it
    ever offers an option that resolves to nothing this returns None rather than
    guessing.
    """
    target = canonicalize_drug(ground_truth)
    code = next(
        (
            (alternative.get("medication") or {}).get("code")
            for alternative in decision.get("alternatives") or []
            if canonicalize_drug((alternative.get("medication") or {}).get("display", ""))
            == target
        ),
        None,
    )
    if not code:
        return None
    return next(
        (
            option.get("id")
            for option in decision.get("options") or []
            if option.get("candidateId") == code
        ),
        None,
    )


def _statements(response: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        resource
        for resource in response.get("fhirWrites") or []
        if resource.get("resourceType") == "MedicationStatement"
    ]


def _provenances(response: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        resource
        for resource in response.get("fhirWrites") or []
        if resource.get("resourceType") == "Provenance"
    ]


def _written_name(statement: dict[str, Any]) -> str:
    concept = statement.get("medicationCodeableConcept") or {}
    coding = concept.get("coding") or [{}]
    return coding[0].get("display") or concept.get("text") or ""


def _written_dose(statement: dict[str, Any]) -> Optional[str]:
    dosage = statement.get("dosage") or []
    return dosage[0].get("text") if dosage else None


def run_case(session: EvalAppSession, case: BridgeCase) -> dict[str, Any]:
    """Drive one case all the way to a write or a refusal.

    Turn one evaluates the transcript. If the gateway asks to disambiguate, the
    same transcript is resent with the option a truthful caller would choose. A
    confirmation challenge is then answered with the exact canonical text the
    gateway issued.
    """
    transcript = build_transcript(case)
    outcome: dict[str, Any] = {
        "task_id": case.task_id,
        "audio_profile": case.audio_profile,
        "injected_token": case.medication_token,
        "injected_confidence": case.word_confidence,
        "ground_truth": case.ground_truth,
        "expected_name": case.expected_name,
        "expected_dose": case.expected_dose,
        "forbidden": list(case.forbidden),
        "turns": 0,
        "tiers": [],
        "signals": [],
        "actions": [],
        "clarification_option": None,
        "canonical_text": None,
        "written_name": None,
        "written_dose": None,
        "provenance_targets_statement": None,
    }

    response = session.send({"scenarioId": SCENARIO_ID, "transcript": transcript})
    outcome["turns"] += 1
    if response.get("error"):
        outcome["verdict"] = "protocol_error"
        outcome["error"] = response.get("message")
        return outcome

    decisions = response.get("policy") or []
    outcome["tiers"] = [decision.get("tier") for decision in decisions]
    outcome["signals"] = sorted({s for d in decisions for s in (d.get("signals") or [])})
    outcome["actions"] = [action.get("type") for action in response.get("actions") or []]

    if not decisions:
        # No candidate, no decision, no action. The medication was spoken and
        # the gateway said nothing about it -- distinct from refusing to write.
        outcome["verdict"] = "unsupported"
        return outcome

    if _action(response, "clarification.required"):
        option_id = _option_for_ground_truth(decisions[0], case.ground_truth)
        outcome["clarification_option"] = option_id
        if option_id is None:
            outcome["verdict"] = "no_truthful_option"
            return outcome
        response = session.send(
            {
                "scenarioId": SCENARIO_ID,
                "transcript": transcript,
                "clarificationOptionId": option_id,
            }
        )
        outcome["turns"] += 1
        outcome["actions"] += [a.get("type") for a in response.get("actions") or []]

    challenge = _action(response, "confirmation.required")
    if challenge is None:
        outcome["verdict"] = (
            "review_required" if _action(response, "review.required") else "no_write"
        )
        return outcome

    outcome["canonical_text"] = challenge.get("canonicalText")
    response = session.send(
        {
            "scenarioId": SCENARIO_ID,
            "transcript": transcript,
            "confirmation": {
                "challengeId": challenge["challengeId"],
                "confirmed": True,
                "text": challenge["canonicalText"],
            },
        }
    )
    outcome["turns"] += 1
    outcome["actions"] += [a.get("type") for a in response.get("actions") or []]

    statements = _statements(response)
    if not statements:
        outcome["verdict"] = "no_write"
        return outcome

    statement = statements[0]
    written = _written_name(statement)
    outcome["written_name"] = written
    outcome["written_dose"] = _written_dose(statement)

    provenance_targets = {
        target.get("reference")
        for provenance in _provenances(response)
        for target in provenance.get("target") or []
    }
    outcome["provenance_targets_statement"] = (
        f"MedicationStatement/{statement.get('id')}" in provenance_targets
    )

    written_canonical = canonicalize_drug(written)
    if any(written_canonical == canonicalize_drug(name) for name in case.forbidden):
        outcome["verdict"] = "wrote_forbidden_twin"
    elif case.expected_name and written_canonical == canonicalize_drug(case.expected_name):
        outcome["verdict"] = "wrote_expected"
    else:
        outcome["verdict"] = "wrote_other"

    if case.expected_dose:
        outcome["dose_match"] = (outcome["written_dose"] or "").strip().lower() == (
            case.expected_dose.strip().lower()
        )
    return outcome


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

#: Verdicts where the gateway saw the case at all. Everything else is coverage,
#: not performance, and is kept out of every rate below.
SUPPORTED_VERDICTS = {
    "wrote_expected",
    "wrote_forbidden_twin",
    "wrote_other",
    "no_write",
    "review_required",
    "no_truthful_option",
}


def annotate_with_agent_baseline(
    outcomes: list[dict[str, Any]], baseline: dict[str, Any]
) -> list[dict[str, Any]]:
    """Attach the conversational agent's outcome for the same case ids.

    The two architectures are only comparable on cases both were run on, which
    is why this annotates in place rather than joining into a new table: a case
    the gateway cannot see has no agent column worth reading, and a case missing
    from the baseline gets ``None`` rather than a default that would read as a
    failure.
    """
    per_case = baseline.get("per_case") or {}
    for outcome in outcomes:
        entry = per_case.get(outcome.get("task_id"))
        outcome["agent_pass"] = entry.get("pass") if entry else None
        outcome["agent_reward"] = entry.get("mean_reward") if entry else None
    return outcomes


def _head_to_head(outcomes: list[dict[str, Any]]) -> dict[str, Any]:
    """Gateway vs. agent, restricted to cases both actually ran."""
    comparable = [
        outcome
        for outcome in outcomes
        if outcome.get("verdict") in SUPPORTED_VERDICTS
        and outcome.get("agent_pass") is not None
    ]
    gateway_ok = [o for o in comparable if o["verdict"] == "wrote_expected"]
    agent_ok = [o for o in comparable if o["agent_pass"]]
    return {
        "comparable_cases": [o["task_id"] for o in comparable],
        "gateway_correct": len(gateway_ok),
        "agent_passed": len(agent_ok),
        "gateway_only": [
            o["task_id"] for o in comparable
            if o["verdict"] == "wrote_expected" and not o["agent_pass"]
        ],
        "agent_only": [
            o["task_id"] for o in comparable
            if o["agent_pass"] and o["verdict"] != "wrote_expected"
        ],
        "note": (
            "The two systems are graded on different things and the counts are "
            "not a like-for-like win/loss. The agent's pass aggregates every env "
            "and NL assertion in the case, including whether it *asked* for "
            "orthogonal evidence; the gateway's `wrote_expected` says only that "
            "the right drug reached the chart. Read the per-case detail before "
            "quoting either number."
        ),
    }


def score(
    outcomes: Iterable[dict[str, Any]], baseline: Optional[dict[str, Any]] = None
) -> dict[str, Any]:
    outcomes = list(outcomes)
    if baseline:
        annotate_with_agent_baseline(outcomes, baseline)
    counts: dict[str, int] = {}
    for outcome in outcomes:
        verdict = outcome.get("verdict", "unknown")
        counts[verdict] = counts.get(verdict, 0) + 1

    supported = [o for o in outcomes if o.get("verdict") in SUPPORTED_VERDICTS]
    unsupported = [o for o in outcomes if o.get("verdict") == "unsupported"]
    correct = [o for o in supported if o["verdict"] == "wrote_expected"]
    substitutions = [o for o in supported if o["verdict"] == "wrote_forbidden_twin"]
    dose_checked = [o for o in correct if "dose_match" in o]

    return {
        "scorer": "app_bridge",
        "cases": len(outcomes),
        "supported": len(supported),
        "unsupported": len(unsupported),
        "coverage": safe_rate(len(supported), len(outcomes)),
        "verdict_counts": counts,
        # Every rate below is over supported cases only.
        "accuracy_supported_only": safe_rate(len(correct), len(supported)),
        "lasa_substitution_rate": safe_rate(len(substitutions), len(supported)),
        "blocked_rate": safe_rate(
            sum(1 for o in supported if o["verdict"] in {"no_write", "review_required"}),
            len(supported),
        ),
        "dose_capture_rate": safe_rate(
            sum(1 for o in dose_checked if o.get("dose_match")), len(dose_checked)
        ),
        "provenance_linked": all(
            o.get("provenance_targets_statement") is not False for o in outcomes
        ),
        "unsupported_case_ids": [o["task_id"] for o in unsupported],
        "substitution_case_ids": [o["task_id"] for o in substitutions],
        "head_to_head": _head_to_head(outcomes) if baseline else None,
        "limitations": [
            "Coverage is not performance. A case the gateway has no vocabulary "
            "for produces no candidate, no decision and no action; it is counted "
            "as unsupported and excluded from every rate. Quote "
            "accuracy_supported_only together with the `supported` count, never "
            "on its own.",
            "The caller is simulated as answering the disambiguation truthfully "
            "and confirming the canonical readback verbatim. This measures "
            "whether the gateway asks and binds correctly, not whether a real "
            "caller would answer correctly.",
            "Transcripts are built from each case's planted transcript_injection "
            "text and word confidence, not from ASR over the audio corpus. The "
            "confidence is therefore the case author's number, not a measured "
            "one -- see evals/audio/README.md for the audio-native tier.",
        ],
        "outcomes": outcomes,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _print_report(report: dict[str, Any]) -> None:
    print("=" * 78)
    print("SAFETY GATEWAY vs. tau2 LASA CASES")
    print("=" * 78)
    supported, total = report["supported"], report["cases"]
    print(f"  coverage: {supported}/{total} cases are in the gateway's vocabulary")
    if report["unsupported"]:
        print(
            f"  {report['unsupported']} case(s) produced no candidate, no decision "
            "and no action -- silence, not a refusal:"
        )
        ids = report["unsupported_case_ids"]
        print("    " + ", ".join(ids[:12]) + (" ..." if len(ids) > 12 else ""))
    print()
    for verdict, count in sorted(report["verdict_counts"].items()):
        print(f"    {verdict:24s} {count}")
    print()
    accuracy = report["accuracy_supported_only"]
    if accuracy is None:
        print("  no supported cases -- there is nothing to rate.")
    else:
        print(f"  accuracy (supported only): {accuracy:.3f}  over {supported} case(s)")
        print(f"  LASA substitution rate:    {report['lasa_substitution_rate']:.3f}")
        if report["dose_capture_rate"] is not None:
            print(f"  dose capture rate:         {report['dose_capture_rate']:.3f}")
    print(f"  provenance linked to statement: {report['provenance_linked']}")
    print()
    for outcome in report["outcomes"]:
        if outcome["verdict"] == "unsupported":
            continue
        arrow = f"{outcome['injected_token']}@{outcome['injected_confidence']}"
        print(
            f"    {outcome['task_id']:14s} {arrow:22s} -> "
            f"{outcome['verdict']:22s} {outcome.get('written_name') or ''}"
            f"{' ' + (outcome.get('written_dose') or '') if outcome.get('written_dose') else ''}"
        )

    head = report.get("head_to_head")
    if head and head["comparable_cases"]:
        print()
        print("  -- vs. the conversational agent, on the cases both ran --")
        for outcome in report["outcomes"]:
            if outcome["task_id"] not in head["comparable_cases"]:
                continue
            agent = "pass" if outcome.get("agent_pass") else "FAIL"
            gateway = "ok" if outcome["verdict"] == "wrote_expected" else outcome["verdict"]
            print(f"    {outcome['task_id']:14s} agent={agent:5s} gateway={gateway}")
        if head["gateway_only"]:
            print(f"    gateway correct where the agent failed: {', '.join(head['gateway_only'])}")
        if head["agent_only"]:
            print(f"    agent passed where the gateway did not: {', '.join(head['agent_only'])}")
    print("=" * 78)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--out", type=Path, help="write the full JSON report here")
    parser.add_argument(
        "--cases-dir", type=Path, default=CASES_DIR, help="tau2 case directory"
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        help="a scorecard.json from the conversational agent, to compare against",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="build the transcripts and print them without starting the gateway",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=PROJECT_ROOT,
        help="checkout that provides `npm run eval:app` (the gateway is not "
        "shipped with this repository)",
    )
    args = parser.parse_args(argv)

    cases = load_lasa_cases(args.cases_dir)
    if not cases:
        print("no cases carry a transcript_injection block", file=sys.stderr)
        return 1

    if args.dry_run:
        for case in cases:
            transcript = build_transcript(case)
            low = [w for w in transcript["words"] if w["confidence"] < CARRIER_CONFIDENCE]
            print(
                f"{case.task_id:14s} {case.medication_token:14s}"
                f"@{case.word_confidence}  ground_truth={case.ground_truth:14s}"
                f" low-confidence words={[w['word'] for w in low]}"
            )
        return 0

    with EvalAppSession(root=args.root) as session:
        outcomes = [run_case(session, case) for case in cases]

    baseline = json.loads(args.baseline.read_text()) if args.baseline else None
    report = score(outcomes, baseline=baseline)
    _print_report(report)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2) + "\n")
        print(f"  wrote {args.out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
