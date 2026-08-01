"""Scoring hook for the Tier-A audio corpus.

Takes the manifest plus a ``transcribe(path) -> str`` callable and computes:

1. **WER** per render and per profile -- the conventional number.
2. **Drug-name error rate (DNER)** -- the number that actually matters. For a
   medication-reporting agent, a 12% WER spread evenly over function words is
   harmless; a 12% WER concentrated on the drug token is a prescribing error.
   Every LASA render is classified into exactly one outcome:

       correct         ground-truth drug present in the hypothesis
       twin_confusion  the LASA twin present, ground truth absent  <-- the danger
       near_miss       a close-but-wrong spelling of the ground truth
       other_drug      a different real drug name from the corpus vocabulary
       missing         no recognisable drug token at all

   ``twin_confusion`` is separated out because it is the only outcome that is
   silently plausible downstream: "hydroxyzine 50 mg TID" looks like a valid
   medication entry and will not trip any validator. ``missing`` is loud and a
   well-built agent will ask again; ``twin_confusion`` is quiet and it will not.

Offline mode
------------
``--mock`` drives the whole path from a hand-written confusion table, so the
scorer, the joins, and the aggregation can be tested today with no API key. The
mock numbers are **fabricated by construction** and mean nothing about any real
ASR. They exist to prove the plumbing works.

Swapping in a real ASR is one function: write a ``transcribe(path) -> str`` and
pass it to :func:`evaluate`. :func:`deepgram_transcriber` is provided as the
reference implementation (stdlib urllib, activated by ``DEEPGRAM_API_KEY``).
"""

from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Callable, Iterable

from rx_bench.audio import manifest as M

Transcriber = Callable[[str], str]


# ==========================================================================
# Text normalisation and WER
# ==========================================================================

_PUNCT = re.compile(r"[^\w\s']")
_WS = re.compile(r"\s+")


def normalize(text: str) -> list[str]:
    """Lowercase, strip punctuation, collapse whitespace -> token list."""
    t = _PUNCT.sub(" ", (text or "").lower())
    t = t.replace("'", "")
    return [w for w in _WS.sub(" ", t).strip().split(" ") if w]


def wer(ref: str, hyp: str) -> dict:
    """Word error rate with S/D/I breakdown (Levenshtein over tokens)."""
    r, h = normalize(ref), normalize(hyp)
    n, m = len(r), len(h)
    if n == 0:
        return {"wer": float(m > 0), "sub": 0, "del": 0, "ins": m, "n_ref": 0}

    # dp[i][j] = (cost, sub, del, ins)
    prev = [(j, 0, 0, j) for j in range(m + 1)]
    for i in range(1, n + 1):
        cur = [(i, 0, i, 0)] + [(0, 0, 0, 0)] * m
        for j in range(1, m + 1):
            if r[i - 1] == h[j - 1]:
                c, s, d, ins = prev[j - 1]
                cand = (c, s, d, ins)
            else:
                c, s, d, ins = prev[j - 1]
                cand = (c + 1, s + 1, d, ins)
            c, s, d, ins = prev[j]                       # deletion
            if c + 1 < cand[0]:
                cand = (c + 1, s, d + 1, ins)
            c, s, d, ins = cur[j - 1]                    # insertion
            if c + 1 < cand[0]:
                cand = (c + 1, s, d, ins + 1)
            cur[j] = cand
        prev = cur
    cost, s, d, ins = prev[m]
    return {"wer": cost / n, "sub": s, "del": d, "ins": ins, "n_ref": n}


# ==========================================================================
# Drug-name outcome classification
# ==========================================================================

NEAR_MISS_RATIO = 0.82  # difflib similarity above which a token is "nearly" the drug


def _candidates(hyp: str) -> list[str]:
    """Token unigrams + bigrams + trigrams, with spaces removed on the n-grams.

    ASR routinely splits a long drug name into pieces ("hydra lazine",
    "rope in a role"). Joining n-grams recovers those.
    """
    toks = normalize(hyp)
    out = list(toks)
    for k in (2, 3, 4):
        for i in range(len(toks) - k + 1):
            out.append("".join(toks[i:i + k]))
    return out


def classify_drug(hyp: str, drug: str, twin: str | None,
                  vocabulary: Iterable[str] = ()) -> dict:
    """Classify how the ground-truth drug fared in a hypothesis transcript."""
    drug_l = drug.lower()
    twin_l = twin.lower() if twin else None
    cands = _candidates(hyp)
    cset = set(cands)

    if drug_l in cset:
        return {"outcome": "correct", "matched": drug_l, "similarity": 1.0}
    if twin_l and twin_l in cset:
        return {"outcome": "twin_confusion", "matched": twin_l, "similarity": None}

    # best near match to the ground truth
    best, best_ratio = None, 0.0
    for c in cands:
        if len(c) < 4:
            continue
        r = difflib.SequenceMatcher(None, c, drug_l).ratio()
        if r > best_ratio:
            best, best_ratio = c, r
    if best_ratio >= NEAR_MISS_RATIO:
        # guard: if it is even closer to the twin, it is a twin confusion
        if twin_l and difflib.SequenceMatcher(None, best, twin_l).ratio() > best_ratio:
            return {"outcome": "twin_confusion", "matched": best, "similarity": best_ratio}
        return {"outcome": "near_miss", "matched": best, "similarity": round(best_ratio, 3)}

    for other in vocabulary:
        ol = other.lower()
        if ol in (drug_l, twin_l):
            continue
        if ol in cset:
            return {"outcome": "other_drug", "matched": ol, "similarity": None}

    return {"outcome": "missing", "matched": best, "similarity": round(best_ratio, 3)}


OUTCOMES = ["correct", "twin_confusion", "near_miss", "other_drug", "missing"]


def corpus_vocabulary(man: dict) -> list[str]:
    v = set()
    for e in man["renders"]:
        if e.get("drug"):
            v.add(e["drug"])
        if e.get("lasa_twin"):
            v.add(e["lasa_twin"])
    return sorted(v)


# ==========================================================================
# Mock transcriber
# ==========================================================================

# Hand-written plausible mis-transcriptions. These are what a phone-band ASR
# with a general language model tends to emit: syllable splits, dropped final
# consonants, and brand-name substitutions. They are illustrative, not measured.
MOCK_GARBLES: dict[str, list[str]] = {
    "hydralazine": ["hydrolyzing", "hydra lazine", "hydralizine"],
    "hydroxyzine": ["hydroxazine", "hydroxy zine", "hydrocodone"],
    "prednisone": ["prednizone", "prednisome", "pregnant zone"],
    "prednisolone": ["prednisilone", "prednis alone", "predni salon"],
    "metformin": ["met forming", "metformine", "metaphor min"],
    "metronidazole": ["metro nidazole", "metronidazol", "metro dial"],
    "clonazepam": ["clonazepan", "clona zepam", "klonopin"],
    "clonidine": ["clonadine", "clone a dean", "colonidine"],
    "bupropion": ["bupropian", "boo propion", "wellbutrin"],
    "buspirone": ["buspar", "bus pirone", "buspirin"],
    "glipizide": ["glipizid", "glip a zide", "clip aside"],
    "glyburide": ["glyburid", "gly bride", "glide bride"],
    "tramadol": ["tramadal", "trauma doll", "tram a doll"],
    "trazodone": ["trazadone", "trazo done", "trace o don"],
    "lamotrigine": ["lamotrigen", "lamo trigine", "la motor gene"],
    "lamivudine": ["lamivadine", "lami vudine", "lamb of you dean"],
    "quetiapine": ["quetiapin", "queti apine", "seroquel"],
    "quinidine": ["quinadine", "quinine", "quin a dean"],
    "risperidone": ["risperidon", "risper adone", "risperdal"],
    "ropinirole": ["ropinerol", "ropini role", "rope in a role"],
    "nicardipine": ["nicardapine", "nick cardipine", "nicardapin"],
    "nifedipine": ["nifedapine", "nifed a pine", "nifedipin"],
    "cyclosporine": ["cyclosporin", "cyclo sporine", "cycle sporin"],
    "cycloserine": ["cycloserin", "cyclo serine", "cycle serene"],
    "sulfadiazine": ["sulfa diazine", "sulfadiazin", "sulfur diazine"],
    "sulfasalazine": ["sulfa salazine", "sulfasalazin", "sulfur salad zine"],
    "chlorpromazine": ["chlorpromazin", "chlor promazine", "clore promising"],
    "chlorpropamide": ["chlorpropamid", "chlor propamide", "clore propa mide"],
    "valacyclovir": ["valacyclovear", "vala cyclovir", "valley cyclo veer"],
    "valganciclovir": ["valganciclovear", "val ganciclovir", "val gan cyclo veer"],
    "levetiracetam": ["levetiracetum", "leveti racetam", "leave a tira setam"],
    "levocarnitine": ["levocarnatine", "levo carnitine", "levo car nine"],
}

# Probability that the drug token is transcribed wrong at all, per profile,
# and -- given an error -- the share that lands on the LASA twin. Both rise
# with channel degradation, which is the qualitative pattern a real ASR shows.
# The exact values are invented.
MOCK_ERROR_RATE = {"clean": 0.07, "phone": 0.25, "noisy_phone": 0.45, "bad_line": 0.60}
MOCK_TWIN_SHARE = {"clean": 0.30, "phone": 0.45, "noisy_phone": 0.55, "bad_line": 0.60}
# Per-word error rate applied to the non-drug words, for a realistic WER floor.
MOCK_WORD_ERROR = {"clean": 0.01, "phone": 0.05, "noisy_phone": 0.11, "bad_line": 0.16}

MOCK_WORD_SWAPS = {
    "milligrams": "milligram", "three": "free", "twice": "twice a",
    "bedtime": "bed time", "morning": "warning", "breakfast": "breakfast time",
    "evening": "even in", "night": "right", "a": "the", "the": "a",
    "my": "by", "i": "and i", "take": "make", "doctor": "docket",
    "bottle": "bottom", "says": "said", "calling": "call in",
}


def _rng_for(render_id: str, seed: int):
    import random
    h = hashlib.sha256(f"{seed}:{render_id}".encode()).hexdigest()[:16]
    return random.Random(int(h, 16))


def mock_transcribe_entry(entry: dict, seed: int = 0) -> str:
    """Produce a fabricated hypothesis transcript for one manifest entry."""
    rnd = _rng_for(entry["id"], seed)
    profile = entry["profile"]
    words = entry["text"].split()

    drug = (entry.get("drug") or "").lower()
    twin = (entry.get("lasa_twin") or "").lower()

    out: list[str] = []
    for w in words:
        bare = _PUNCT.sub("", w).lower()
        if drug and bare == drug:
            if rnd.random() < MOCK_ERROR_RATE.get(profile, 0.2):
                if twin and rnd.random() < MOCK_TWIN_SHARE.get(profile, 0.5):
                    out.append(twin)
                else:
                    out.append(rnd.choice(MOCK_GARBLES.get(drug, [drug[:-2] + "in"])))
            else:
                out.append(drug)
            continue
        if rnd.random() < MOCK_WORD_ERROR.get(profile, 0.05):
            if bare in MOCK_WORD_SWAPS and rnd.random() < 0.7:
                out.append(MOCK_WORD_SWAPS[bare])
            # else: deletion
            continue
        out.append(bare)
    return " ".join(out)


def make_mock_transcriber(man: dict, seed: int = 0) -> Transcriber:
    """Return a ``transcribe(path) -> str`` backed by the mock confusion table.

    Signature-identical to a real ASR client, so :func:`evaluate` cannot tell
    the difference. That is the point: the same code path is exercised.
    """
    by_path = {e["path"]: e for e in man["renders"]}
    by_rel = {e["rel_path"]: e for e in man["renders"]}

    def transcribe(path: str) -> str:
        entry = by_path.get(str(path)) or by_rel.get(str(path))
        if entry is None:
            raise KeyError(f"mock transcriber has no manifest entry for {path}")
        return mock_transcribe_entry(entry, seed)

    return transcribe


# ==========================================================================
# Real ASR: Deepgram reference implementation
# ==========================================================================

def deepgram_transcriber(model: str = "nova-2-medical",
                         api_key: str | None = None,
                         timeout: float = 120.0) -> Transcriber:
    """Return a ``transcribe(path) -> str`` backed by Deepgram.

    This is the ONLY function that needs to change to swap in another vendor:
    write an equivalent factory returning ``Callable[[str], str]`` and pass it
    to :func:`evaluate`. Requires ``DEEPGRAM_API_KEY``; no key is set in this
    environment, which is why ``--mock`` exists.
    """
    import urllib.request

    key = api_key or os.environ.get("DEEPGRAM_API_KEY")
    if not key:
        raise RuntimeError(
            "DEEPGRAM_API_KEY is not set. Run with --mock for the offline path.")

    url = (f"https://api.deepgram.com/v1/listen?model={model}"
           f"&smart_format=false&punctuate=true&language=en")

    def transcribe(path: str) -> str:
        data = Path(path).read_bytes()
        req = urllib.request.Request(url, data=data, method="POST", headers={
            "Authorization": f"Token {key}",
            "Content-Type": "audio/wav",
        })
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            payload = json.loads(resp.read().decode())
        return (payload["results"]["channels"][0]["alternatives"][0]["transcript"] or "")

    return transcribe


# ==========================================================================
# Evaluation
# ==========================================================================

def evaluate(man: dict, transcribe: Transcriber, profiles: list[str] | None = None,
             kinds: list[str] | None = None, verbose: bool = False) -> dict:
    """Run ``transcribe`` over the manifest and aggregate WER + drug outcomes."""
    vocab = corpus_vocabulary(man)
    rows: list[dict] = []
    errors: list[dict] = []

    for e in man["renders"]:
        if profiles and e["profile"] not in profiles:
            continue
        if kinds and e["kind"] not in kinds:
            continue
        try:
            hyp = transcribe(e["path"])
        except Exception as ex:
            errors.append({"id": e["id"], "error": repr(ex)})
            if verbose:
                print(f"  ASR FAIL {e['id']}: {ex}", file=sys.stderr)
            continue

        w = wer(e["text"], hyp)
        row = {
            "id": e["id"], "kind": e["kind"], "profile": e["profile"],
            "pair_id": e["pair_id"], "member": e["member"], "red_flag": e["red_flag"],
            "voice": e["voice"], "rate_wpm": e["rate_wpm"], "archetype": e["archetype"],
            "reference": e["text"], "hypothesis": hyp,
            "wer": round(w["wer"], 4), "sub": w["sub"], "del": w["del"],
            "ins": w["ins"], "n_ref": w["n_ref"],
            "drug": e["drug"], "lasa_twin": e["lasa_twin"],
        }
        if e["kind"] == "lasa" and e["drug"]:
            row.update(classify_drug(hyp, e["drug"], e["lasa_twin"], vocab))
        else:
            row["outcome"] = None
        rows.append(row)

    return {"rows": rows, "errors": errors, "summary": summarize(rows)}


def _mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else 0.0


def summarize(rows: list[dict]) -> dict:
    """Aggregate by profile, and by LASA pair, and overall."""
    by_profile: dict[str, dict] = {}
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        groups[r["profile"]].append(r)

    for prof, rs in groups.items():
        lasa = [r for r in rs if r["outcome"] is not None]
        counts = {o: sum(1 for r in lasa if r["outcome"] == o) for o in OUTCOMES}
        n = max(len(lasa), 1)
        by_profile[prof] = {
            "n_renders": len(rs),
            "n_lasa": len(lasa),
            "wer": round(_mean(r["wer"] for r in rs), 4),
            "wer_micro": round(
                sum(r["sub"] + r["del"] + r["ins"] for r in rs)
                / max(sum(r["n_ref"] for r in rs), 1), 4),
            "outcome_counts": counts,
            "drug_name_error_rate": round(1.0 - counts["correct"] / n, 4),
            "twin_confusion_rate": round(counts["twin_confusion"] / n, 4),
            "silent_error_share": round(
                counts["twin_confusion"] / max(len(lasa) - counts["correct"], 1), 4),
        }

    by_pair: dict[str, dict] = {}
    pair_groups: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        if r["outcome"] is not None and r["pair_id"]:
            pair_groups[r["pair_id"]].append(r)
    for pid, rs in pair_groups.items():
        n = len(rs)
        by_pair[pid] = {
            "n": n,
            "correct": sum(1 for r in rs if r["outcome"] == "correct"),
            "twin_confusion": sum(1 for r in rs if r["outcome"] == "twin_confusion"),
            "twin_confusion_rate": round(
                sum(1 for r in rs if r["outcome"] == "twin_confusion") / max(n, 1), 4),
            "wer": round(_mean(r["wer"] for r in rs), 4),
        }

    lasa_all = [r for r in rows if r["outcome"] is not None]
    overall = {
        "n_renders": len(rows),
        "n_lasa": len(lasa_all),
        "wer": round(_mean(r["wer"] for r in rows), 4),
        "drug_name_error_rate": round(
            1.0 - sum(1 for r in lasa_all if r["outcome"] == "correct") / max(len(lasa_all), 1), 4),
        "twin_confusion_rate": round(
            sum(1 for r in lasa_all if r["outcome"] == "twin_confusion") / max(len(lasa_all), 1), 4),
    }
    return {"overall": overall, "by_profile": by_profile, "by_pair": by_pair}


def print_report(res: dict, top_pairs: int = 8) -> None:
    s = res["summary"]
    o = s["overall"]
    print(f"\nrenders scored: {o['n_renders']}  (LASA: {o['n_lasa']})"
          f"   ASR failures: {len(res['errors'])}")
    print(f"overall  WER {o['wer']:.3f}   drug-name error {o['drug_name_error_rate']:.3f}"
          f"   twin-confusion {o['twin_confusion_rate']:.3f}")

    order = [p for p in M.PROFILES if p in s["by_profile"]]
    order += [p for p in s["by_profile"] if p not in order]
    print(f"\n  {'profile':<13}{'n':>4}{'WER':>8}{'DNER':>8}{'twin':>8}"
          f"{'near':>7}{'other':>7}{'miss':>7}{'silent%':>9}")
    for p in order:
        d = s["by_profile"][p]
        c = d["outcome_counts"]
        print(f"  {p:<13}{d['n_renders']:>4}{d['wer']:>8.3f}"
              f"{d['drug_name_error_rate']:>8.3f}{d['twin_confusion_rate']:>8.3f}"
              f"{c['near_miss']:>7}{c['other_drug']:>7}{c['missing']:>7}"
              f"{d['silent_error_share']*100:>8.1f}%")
    print("  DNER = drug-name error rate. twin = fraction transcribed as the LASA twin.")
    print("  silent% = of the drug-name errors, the share that produced a plausible "
          "wrong drug rather than a garble.")

    pairs = sorted(s["by_pair"].items(), key=lambda kv: -kv[1]["twin_confusion_rate"])
    if pairs:
        print(f"\n  worst LASA pairs by twin-confusion rate:")
        for pid, d in pairs[:top_pairs]:
            print(f"    {pid:<32} {d['twin_confusion']}/{d['n']} "
                  f"({d['twin_confusion_rate']:.2f})  WER {d['wer']:.3f}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Score the Tier-A audio corpus.")
    ap.add_argument("--manifest", default=str(M.MANIFEST_PATH))
    ap.add_argument("--mock", action="store_true",
                    help="offline fabricated ASR; proves the path without a key")
    ap.add_argument("--deepgram", action="store_true", help="use Deepgram (needs DEEPGRAM_API_KEY)")
    ap.add_argument("--model", default="nova-2-medical")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--profiles", nargs="*", default=None)
    ap.add_argument("--kinds", nargs="*", default=None, choices=["lasa", "distress"])
    ap.add_argument("--out", help="write full per-render results JSON here")
    a = ap.parse_args(argv)

    man = M.load_manifest(a.manifest)

    if a.deepgram:
        transcribe = deepgram_transcriber(a.model)
        label = f"deepgram:{a.model}"
    elif a.mock:
        transcribe = make_mock_transcriber(man, a.seed)
        label = f"mock(seed={a.seed})"
    else:
        ap.error("pick one of --mock or --deepgram")
        return 2

    res = evaluate(man, transcribe, a.profiles, a.kinds, verbose=True)
    res["asr"] = label
    print(f"\nASR: {label}")
    if a.mock:
        print("!! MOCK MODE: these numbers are fabricated from a hand-written "
              "confusion table.\n   They demonstrate the scoring path. They say "
              "nothing about any real ASR.")
    print_report(res)
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=2))
        print(f"\n  wrote {a.out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
