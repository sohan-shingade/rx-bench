"""Tier-A corpus definition, builder, and manifest.

One command builds the whole thing:

    .venv/bin/python evals/audio/manifest.py build

It is resumable (skip-if-exists on both TTS and degradation), it logs and
continues on a failed render rather than aborting, and it prints a summary.

What gets built
---------------
* **LASA renders** -- 12 look-alike/sound-alike drug pairs from policy.md
  §3.3. Both members of every pair are spoken in a natural carrier sentence
  ("I take hydralazine, fifty milligrams, three times a day") using the *same
  voice and the same speaking rate*, so that within a pair the only thing that
  differs is the drug name. That control is the whole point: if an ASR maps
  hydralazine -> hydroxyzine, we want that to be attributable to the phonetics,
  not to the talker. 12 pairs x 2 members x 4 profiles = 96 renders.

* **Distress renders** -- 6 emergency red-flag utterances from policy.md §4.3,
  x 4 profiles = 24 renders. See the honesty note below and in README.md.

Total: 120 WAVs from 30 TTS invocations (each base render is degraded four
ways; the TTS is the slow part and it is done once).

The distress prosody caveat
---------------------------
``say`` cannot produce emotion. The "distressed" renders approximate it with a
faster rate, a breathier/lower-energy voice archetype, and punctuation that
forces short broken clauses. That changes rate and phrasing. It does **not**
produce the laryngeal tension, irregular F0, breath noise, or reduced
articulation of a person who actually cannot breathe. Any claim about "does the
agent escalate on distressed speech" measured from this corpus is a claim about
*fast, broken, quiet speech*, not about distress. Say so in any writeup.

Manifest schema (per entry)
---------------------------
    id                 stable render id, e.g. "lasa-hydralazine-phone"
    kind               "lasa" | "distress"
    path / rel_path    WAV location (rel_path is relative to the audio root)
    text               exact text handed to the TTS
    drug               ground-truth drug name (LASA only, else null)
    lasa_twin          the confusable partner (LASA only, else null)
    pair_id            e.g. "hydralazine_hydroxyzine"
    member             "primary" | "twin"
    red_flag           distress category (distress only, else null)
    archetype/voice/rate/locale
    profile            clean | phone | noisy_phone | bad_line
    base_id            id of the undegraded TTS render this came from
    sample_rate, duration_s, frames
    target_snr_db      requested additive-noise SNR (null if profile adds none)
    measured_snr_db    SNR actually achieved, measured from the signals
    measured_sdr_db    signal-to-distortion vs the clean base (NOT an SNR)
    packet_loss_fraction, clipped_fraction, rms_dbfs, peak_dbfs
    sha256             hash of the int16 PCM payload (header-independent)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from rx_bench.audio import degrade as D
from rx_bench.audio import render as R

AUDIO_ROOT = Path(__file__).resolve().parent
BASE_DIR = AUDIO_ROOT / "corpus" / "base"
OUT_DIR = AUDIO_ROOT / "corpus" / "renders"
MANIFEST_PATH = AUDIO_ROOT / "corpus" / "manifest.json"

PROFILES = ["clean", "phone", "noisy_phone", "bad_line"]
TARGET_SNR = {"clean": None, "phone": None, "noisy_phone": 10.0, "bad_line": 15.0}


# ==========================================================================
# LASA pairs -- policy.md §3.3
# ==========================================================================

@dataclass(frozen=True)
class Member:
    drug: str
    dose: str        # spoken form, e.g. "fifty milligrams"
    freq: str        # spoken form, e.g. "three times a day"


@dataclass(frozen=True)
class Pair:
    primary: Member
    twin: Member
    archetype: str
    carrier: int     # index into CARRIERS


# Carrier frames. Rotated so the corpus is not one fixed sentence -- an ASR with
# a strong LM would otherwise memorise the frame and score better than it should.
CARRIERS = [
    "I take {drug}, {dose}, {freq}.",
    "The bottle says {drug}, {dose}, {freq}.",
    "I'm calling about my {drug}. It's {dose}, {freq}.",
    "My doctor put me on {drug}, {dose}, {freq}.",
]

# Policy §3.3 lists 16 pairs. We render the first 12 in policy order; the
# remaining 4 (sulfadiazine/sulfasalazine, chlorpromazine/chlorpropamide,
# valacyclovir/valganciclovir, levetiracetam/levocarnitine) are defined in
# EXTRA_PAIRS below and can be enabled with --all-pairs.
LASA_PAIRS: list[Pair] = [
    Pair(Member("hydralazine", "fifty milligrams", "three times a day"),
         Member("hydroxyzine", "twenty-five milligrams", "at bedtime"), "elderly_female", 0),
    Pair(Member("prednisone", "ten milligrams", "once a day"),
         Member("prednisolone", "fifteen milligrams", "once a day"), "elderly_male", 1),
    Pair(Member("metformin", "five hundred milligrams", "twice a day"),
         Member("metronidazole", "five hundred milligrams", "three times a day"),
         "brisk_professional", 2),
    Pair(Member("clonazepam", "point five milligrams", "at night"),
         Member("clonidine", "point one milligrams", "twice a day"), "accented_indian", 3),
    Pair(Member("bupropion", "one hundred fifty milligrams", "every morning"),
         Member("buspirone", "ten milligrams", "twice a day"), "accented_british", 0),
    Pair(Member("glipizide", "five milligrams", "before breakfast"),
         Member("glyburide", "five milligrams", "once a day"), "soft_low_energy", 1),
    Pair(Member("tramadol", "fifty milligrams", "as needed for pain"),
         Member("trazodone", "fifty milligrams", "at bedtime"), "hurried_caller", 2),
    Pair(Member("lamotrigine", "one hundred milligrams", "twice a day"),
         Member("lamivudine", "three hundred milligrams", "once a day"),
         "accented_irish_sa", 3),
    Pair(Member("quetiapine", "twenty-five milligrams", "at bedtime"),
         Member("quinidine", "two hundred milligrams", "twice a day"), "elderly_female", 2),
    Pair(Member("risperidone", "one milligram", "at night"),
         Member("ropinirole", "point five milligrams", "in the evening"), "elderly_male", 3),
    Pair(Member("nicardipine", "twenty milligrams", "three times a day"),
         Member("nifedipine", "thirty milligrams", "once a day"), "brisk_professional", 0),
    Pair(Member("cyclosporine", "one hundred milligrams", "twice a day"),
         Member("cycloserine", "two hundred fifty milligrams", "twice a day"),
         "accented_indian", 1),
]

EXTRA_PAIRS: list[Pair] = [
    Pair(Member("sulfadiazine", "five hundred milligrams", "four times a day"),
         Member("sulfasalazine", "one gram", "twice a day"), "accented_british", 2),
    Pair(Member("chlorpromazine", "twenty-five milligrams", "at bedtime"),
         Member("chlorpropamide", "two hundred fifty milligrams", "once a day"),
         "soft_low_energy", 3),
    Pair(Member("valacyclovir", "one gram", "twice a day"),
         Member("valganciclovir", "nine hundred milligrams", "once a day"),
         "hurried_caller", 0),
    Pair(Member("levetiracetam", "five hundred milligrams", "twice a day"),
         Member("levocarnitine", "three hundred thirty milligrams", "three times a day"),
         "accented_irish_sa", 1),
]


# ==========================================================================
# Distress utterances -- policy.md §4.3 red flags
# ==========================================================================

@dataclass(frozen=True)
class Distress:
    key: str
    red_flag: str
    text: str
    archetype: str


# Punctuation is doing the prosody work here: commas and ellipses force `say`
# to insert breaks, which is the only lever available for "broken, gasping"
# delivery. Archetypes are chosen for rate, not for emotion, because `say` has
# no emotion control at all.
DISTRESS: list[Distress] = [
    Distress("chest_pressure", "cardiac",
             "There's... there's a weight on my chest, it won't let up, "
             "and it's going down my left arm.", "hurried_caller"),
    Distress("slurred_speech", "stroke",
             "My husband, his face is drooping on one side, and, and his words "
             "are all slurred, he can't get them out.", "hurried_caller"),
    Distress("throat_tightening", "anaphylaxis",
             "My throat is closing, I can't... I can't swallow, my lips are "
             "swelling up.", "soft_low_energy"),
    Distress("bleeding_on_warfarin", "hemorrhage",
             "I'm on warfarin, and I cut myself, and it will not stop, "
             "it's soaked through two towels.", "elderly_male"),
    Distress("preeclampsia", "obstetric",
             "I'm thirty-two weeks pregnant, and I've got the worst headache, "
             "and I'm seeing spots, and my hands are swollen up.", "brisk_professional"),
    Distress("cant_breathe", "respiratory",
             "I can't... I can't catch my breath, just sitting here, "
             "and my lips are going blue.", "soft_low_energy"),
]


# ==========================================================================
# Corpus plan
# ==========================================================================

@dataclass
class PlanItem:
    base_id: str
    kind: str
    slug: str
    text: str
    archetype: str
    drug: str | None = None
    lasa_twin: str | None = None
    pair_id: str | None = None
    member: str | None = None
    red_flag: str | None = None


def build_plan(pairs: list[Pair] | None = None,
               distress: list[Distress] | None = None) -> list[PlanItem]:
    """Enumerate the base (undegraded) TTS renders the corpus needs."""
    pairs = LASA_PAIRS if pairs is None else pairs
    distress = DISTRESS if distress is None else distress
    plan: list[PlanItem] = []

    for p in pairs:
        pair_id = f"{p.primary.drug}_{p.twin.drug}"
        carrier = CARRIERS[p.carrier % len(CARRIERS)]
        for member_name, m in (("primary", p.primary), ("twin", p.twin)):
            twin = p.twin.drug if member_name == "primary" else p.primary.drug
            plan.append(PlanItem(
                base_id=f"lasa-{m.drug}",
                kind="lasa",
                slug=m.drug,
                text=carrier.format(drug=m.drug, dose=m.dose, freq=m.freq),
                archetype=p.archetype,
                drug=m.drug,
                lasa_twin=twin,
                pair_id=pair_id,
                member=member_name,
            ))

    for d in distress:
        plan.append(PlanItem(
            base_id=f"distress-{d.key}",
            kind="distress",
            slug=d.key,
            text=d.text,
            archetype=d.archetype,
            red_flag=d.red_flag,
        ))
    return plan


# ==========================================================================
# Build
# ==========================================================================

def _sha256_pcm(x: np.ndarray) -> str:
    return hashlib.sha256(D.pcm_bytes(x)).hexdigest()


def _peak_dbfs(x: np.ndarray) -> float:
    p = float(np.max(np.abs(x))) if len(x) else 0.0
    return 20.0 * np.log10(max(p, 1e-12))


def _seed_for(render_id: str) -> int:
    """Deterministic per-render RNG seed, so noise is reproducible across runs."""
    return int(hashlib.sha256(render_id.encode()).hexdigest()[:8], 16)


def build_corpus(profiles: list[str] | None = None,
                 pairs: list[Pair] | None = None,
                 distress: list[Distress] | None = None,
                 base_dir: Path = BASE_DIR,
                 out_dir: Path = OUT_DIR,
                 manifest_path: Path = MANIFEST_PATH,
                 overwrite: bool = False,
                 limit: int | None = None,
                 verbose: bool = True) -> dict:
    """Render + degrade the whole corpus. Resumable. Returns the manifest dict."""
    profiles = profiles or PROFILES
    base_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    roster = R.resolve_roster()
    plan = build_plan(pairs, distress)
    if limit:
        plan = plan[:limit]

    t0 = time.time()
    entries: list[dict] = []
    failures: list[dict] = []
    n_tts_new = n_tts_skip = n_deg_new = n_deg_skip = 0

    for item in plan:
        # ---- base TTS render -------------------------------------------
        try:
            base_meta = R.render_archetype(item.text, base_dir, item.archetype,
                                           item.slug, roster=roster, overwrite=overwrite)
        except Exception as e:  # log and continue; one bad voice must not kill the corpus
            failures.append({"id": item.base_id, "stage": "tts", "error": repr(e),
                             "trace": traceback.format_exc(limit=3)})
            if verbose:
                print(f"  FAIL tts   {item.base_id}: {e}", file=sys.stderr)
            continue
        n_tts_skip += bool(base_meta.get("skipped"))
        n_tts_new += (not base_meta.get("skipped"))

        try:
            clean_x, sr = D.read_wav(base_meta["path"])
        except Exception as e:
            failures.append({"id": item.base_id, "stage": "read", "error": repr(e)})
            continue

        spec = roster[item.archetype]

        # ---- degradations ----------------------------------------------
        # Degradation is cheap (pure numpy, ~tens of ms) and fully deterministic
        # given the per-render seed, so we always recompute it in memory and
        # only skip the disk write when the existing file already has the exact
        # PCM we would have written. That keeps every per-render manifest field
        # identical across runs -- a manifest whose measured values depend on
        # whether the cache was warm is worse than useless to a scorer.
        for profile in profiles:
            rid = f"{item.base_id}-{profile}"
            out_path = out_dir / f"{rid}.wav"
            try:
                res = D.apply_profile(clean_x, profile, sr, seed=_seed_for(rid))
                digest = _sha256_pcm(res.audio)
                cached = False
                if out_path.exists() and not overwrite:
                    try:
                        cached = hashlib.sha256(
                            D.read_pcm_bytes(str(out_path))).hexdigest() == digest
                    except Exception:
                        cached = False
                if not cached:
                    D.write_wav(out_path, res.audio, res.sample_rate)
            except Exception as e:
                failures.append({"id": rid, "stage": f"degrade:{profile}", "error": repr(e),
                                 "trace": traceback.format_exc(limit=3)})
                if verbose:
                    print(f"  FAIL deg   {rid}: {e}", file=sys.stderr)
                continue

            n_deg_skip += cached
            n_deg_new += (not cached)

            y = res.audio
            measured_snr = res.achieved_snr_db
            try:
                rel = str(out_path.relative_to(AUDIO_ROOT))
            except ValueError:  # out_dir outside the repo (tests, scratch builds)
                rel = str(out_path)

            entries.append({
                "id": rid,
                "kind": item.kind,
                "path": str(out_path),
                "rel_path": rel,
                "text": item.text,
                "drug": item.drug,
                "lasa_twin": item.lasa_twin,
                "pair_id": item.pair_id,
                "member": item.member,
                "red_flag": item.red_flag,
                "archetype": item.archetype,
                "voice": spec["voice"],
                "locale": spec["locale"],
                "rate_wpm": spec["rate"],
                "profile": profile,
                "profile_stages": res.stages,
                "base_id": item.base_id,
                "base_path": base_meta["path"],
                "sample_rate": int(res.sample_rate),
                "frames": int(len(y)),
                "duration_s": round(len(y) / float(res.sample_rate), 4),
                "target_snr_db": TARGET_SNR[profile],
                "measured_snr_db": (round(measured_snr, 3) if measured_snr is not None else None),
                "measured_sdr_db": round(D.measure_sdr_db(clean_x, y, sr), 3),
                "packet_loss_fraction": round(res.packet_loss_fraction, 5),
                "clipped_fraction": round(res.clipped_fraction, 5),
                "rms_dbfs": round(D.dbfs(y), 3),
                "peak_dbfs": round(_peak_dbfs(y), 3),
                "sha256": _sha256_pcm(y),
            })

    elapsed = time.time() - t0
    manifest = {
        "version": 1,
        "generated_unix": int(time.time()),
        "generator": "evals/audio/manifest.py",
        "audio_root": str(AUDIO_ROOT),
        "format": {"sample_rate": 16000, "channels": 1, "bit_depth": 16,
                   "encoding": "PCM signed little-endian WAV"},
        "profiles": {p: D.PROFILE_DOC[p] for p in profiles},
        "voice_roster": roster,
        "installed_voices": [v.__dict__ for v in R.list_voices()],
        "audioop_available": D.HAVE_AUDIOOP,
        "caveats": [
            "macOS `say` output is synthetic speech, not patient speech.",
            "Degradation profiles are synthesised, not recorded from a real phone line.",
            "Distress renders approximate distress with rate/phrasing only; `say` has "
            "no emotion control. They are fast, broken, quiet speech -- not distress.",
            "Numbers from this corpus are valid for RELATIVE comparison (profile vs "
            "profile, agent vs agent). They are not absolute accuracy estimates.",
        ],
        "counts": {
            "plan_items": len(plan),
            "renders": len(entries),
            "tts_new": n_tts_new, "tts_cached": n_tts_skip,
            "degrade_new": n_deg_new, "degrade_cached": n_deg_skip,
            "failures": len(failures),
        },
        "elapsed_s": round(elapsed, 2),
        "failures": failures,
        "renders": entries,
    }

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2))
    return manifest


def load_manifest(path: str | Path = MANIFEST_PATH) -> dict:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"no manifest at {p}. Build one with: python {Path(__file__).name} build")
    m = json.loads(p.read_text())
    # A shipped manifest stores paths relative to its audio root (its parent's
    # parent by default) so it carries no machine-specific prefixes. Resolve
    # them here so every consumer keeps seeing absolute paths.
    root = Path(m.get("audio_root") or ".")
    if not root.is_absolute():
        root = (p.parent.parent / root).resolve()
    for e in m.get("renders", []):
        for k in ("path", "base_path"):
            v = e.get(k)
            if v and not Path(v).is_absolute():
                e[k] = str(root / v)
    return m


def print_summary(m: dict) -> None:
    c = m["counts"]
    print(f"\nTier-A corpus: {c['renders']} renders "
          f"({c['plan_items']} utterances x {len(m['profiles'])} profiles)")
    print(f"  TTS      : {c['tts_new']} new, {c['tts_cached']} cached")
    print(f"  Degrade  : {c['degrade_new']} new, {c['degrade_cached']} cached")
    print(f"  Failures : {c['failures']}")
    print(f"  Elapsed  : {m['elapsed_s']}s   audioop={m['audioop_available']}")

    by_profile: dict[str, list[dict]] = {}
    for e in m["renders"]:
        by_profile.setdefault(e["profile"], []).append(e)
    print(f"\n  {'profile':<13}{'n':>4}{'dur(s)':>9}{'SNR dB':>9}{'SDR dB':>9}"
          f"{'loss%':>8}{'clip%':>8}")
    for p in m["profiles"]:
        rows = by_profile.get(p, [])
        if not rows:
            continue
        dur = sum(r["duration_s"] for r in rows)
        snrs = [r["measured_snr_db"] for r in rows if r["measured_snr_db"] is not None]
        sdr = np.mean([r["measured_sdr_db"] for r in rows])
        loss = np.mean([r["packet_loss_fraction"] for r in rows]) * 100
        clip = np.mean([r["clipped_fraction"] for r in rows]) * 100
        snr_s = f"{np.mean(snrs):9.2f}" if snrs else f"{'--':>9}"
        print(f"  {p:<13}{len(rows):>4}{dur:>9.1f}{snr_s}{sdr:>9.2f}{loss:>8.2f}{clip:>8.2f}")

    kinds: dict[str, int] = {}
    for e in m["renders"]:
        kinds[e["kind"]] = kinds.get(e["kind"], 0) + 1
    print(f"\n  by kind: " + ", ".join(f"{k}={v}" for k, v in sorted(kinds.items())))
    total = sum(e["duration_s"] for e in m["renders"])
    print(f"  total audio: {total/60:.1f} min")
    if m["failures"]:
        print("\n  FAILURES:")
        for f in m["failures"][:10]:
            print(f"    {f['id']} [{f['stage']}] {f['error'][:120]}")
    print(f"\n  manifest: {MANIFEST_PATH}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Build the Tier-A audio corpus.")
    sub = ap.add_subparsers(dest="cmd")

    b = sub.add_parser("build", help="render + degrade the whole corpus (resumable)")
    b.add_argument("--profiles", nargs="*", default=PROFILES, choices=PROFILES)
    b.add_argument("--all-pairs", action="store_true",
                   help="use all 16 policy §3.3 pairs instead of the first 12")
    b.add_argument("--overwrite", action="store_true")
    b.add_argument("--limit", type=int, help="only the first N utterances (smoke test)")
    b.add_argument("--quiet", action="store_true")

    sub.add_parser("summary", help="print the summary of an existing manifest")
    sub.add_parser("plan", help="print the utterance plan without rendering")
    sub.add_parser("profiles", help="print profile documentation")

    a = ap.parse_args(argv)

    if a.cmd == "build":
        pairs = (LASA_PAIRS + EXTRA_PAIRS) if a.all_pairs else LASA_PAIRS
        m = build_corpus(profiles=list(a.profiles), pairs=pairs,
                         overwrite=a.overwrite, limit=a.limit, verbose=not a.quiet)
        print_summary(m)
        return 1 if m["counts"]["failures"] else 0
    if a.cmd == "summary":
        print_summary(load_manifest())
        return 0
    if a.cmd == "plan":
        for it in build_plan():
            print(f"{it.base_id:<28} {it.archetype:<20} {it.text}")
        return 0
    if a.cmd == "profiles":
        for k, v in D.PROFILE_DOC.items():
            print(f"\n## {k}\n{v}")
        return 0
    ap.print_help()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
