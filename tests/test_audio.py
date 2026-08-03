"""pytest suite for the Tier-A audio pipeline.

Run:
    uv run pytest tests/test_audio.py -q

Everything except the two ``say``-dependent tests runs with no macOS, no
network, and no API key.
"""

from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import pytest

from rx_bench.audio import asr_eval as A  # noqa: E402
from rx_bench.audio import degrade as D  # noqa: E402
from rx_bench.audio import manifest as M  # noqa: E402
from rx_bench.audio import render as R  # noqa: E402

SR = D.SAMPLE_RATE
HAVE_SAY = os.path.exists(R.SAY)


def tone(freq: float, dur: float = 1.0, sr: int = SR, amp: float = 0.5) -> np.ndarray:
    t = np.arange(int(sr * dur), dtype=np.float64) / sr
    return (amp * np.sin(2 * math.pi * freq * t)).astype(np.float32)


# ==========================================================================
# WAV I/O
# ==========================================================================

def test_wav_roundtrip_preserves_signal(tmp_path):
    x = tone(440.0, 0.5)
    p = tmp_path / "t.wav"
    D.write_wav(str(p), x, SR)
    y, sr = D.read_wav(str(p))
    assert sr == SR
    assert len(y) == len(x)
    assert np.max(np.abs(y - x)) < 1e-3  # 16-bit quantisation only


def test_read_pcm_bytes_matches_written_payload(tmp_path):
    x = tone(300.0, 0.2)
    p = tmp_path / "t.wav"
    D.write_wav(str(p), x, SR)
    assert D.read_pcm_bytes(str(p)) == D.pcm_bytes(x)


# ==========================================================================
# SNR
# ==========================================================================

@pytest.mark.parametrize("target", [0.0, 3.0, 10.0, 15.0, 20.0, 30.0])
def test_snr_db_exact_on_synthetic_signal(target):
    """Build noise at a known SNR by construction; snr_db must recover it."""
    rng = np.random.default_rng(7)
    sig = tone(500.0, 2.0, amp=0.4)
    noise = rng.standard_normal(len(sig)).astype(np.float32)
    noise = noise * (D.rms(sig) / D.rms(noise)) / (10.0 ** (target / 20.0))
    assert D.snr_db(sig, noise, SR) == pytest.approx(target, abs=0.05)


def test_add_noise_hits_requested_snr_within_tolerance():
    """A full-duration tone has active RMS == global RMS, so the target is exact."""
    sig = tone(700.0, 2.0, amp=0.3)
    for target in (5.0, 10.0, 15.0, 25.0):
        noisy, achieved = D.add_noise(sig, target, SR, kinds=("white",),
                                      rng=np.random.default_rng(1))
        assert achieved == pytest.approx(target, abs=0.1)
        # and verify independently from the produced signals
        assert D.snr_db(sig, noisy - sig, SR) == pytest.approx(target, abs=0.1)


def test_add_noise_snr_holds_with_leading_silence():
    """Active-RMS convention: padding with silence must not change the SNR."""
    sig = tone(700.0, 1.0, amp=0.3)
    padded = np.concatenate([np.zeros(SR, dtype=np.float32), sig,
                             np.zeros(SR, dtype=np.float32)])
    _, achieved = D.add_noise(padded, 12.0, SR, kinds=("white",),
                              rng=np.random.default_rng(2))
    assert achieved == pytest.approx(12.0, abs=0.3)
    # Naive whole-file RMS would have been ~4.8 dB off; assert we did not do that.
    naive = 20 * math.log10(D.rms(padded) / D.active_rms(padded, SR))
    assert naive < -2.0


def test_active_rms_ignores_silence():
    sig = tone(400.0, 1.0, amp=0.5)
    padded = np.concatenate([np.zeros(3 * SR, dtype=np.float32), sig])
    assert D.active_rms(padded, SR) == pytest.approx(D.rms(sig), rel=0.05)
    assert D.rms(padded) < 0.6 * D.rms(sig)


def test_measure_sdr_is_gain_invariant():
    x = tone(600.0, 1.0, amp=0.3)
    assert D.measure_sdr_db(x, D.apply_gain_db(x, -9.0), SR) > 60.0


# ==========================================================================
# Band-pass
# ==========================================================================

def _band_energy(x: np.ndarray, sr: int, lo: float, hi: float) -> float:
    spec = np.abs(np.fft.rfft(np.asarray(x, dtype=np.float64))) ** 2
    f = np.fft.rfftfreq(len(x), 1.0 / sr)
    return float(spec[(f >= lo) & (f < hi)].sum())


def test_telephony_bandpass_attenuates_out_of_band_energy():
    """White noise in -> passband energy retained, stopband energy destroyed."""
    rng = np.random.default_rng(3)
    x = rng.standard_normal(SR * 2).astype(np.float32) * 0.2
    y = D.telephony_bandpass(x, SR)

    in_before = _band_energy(x, SR, 400, 3300)
    in_after = _band_energy(y, SR, 400, 3300)
    assert in_after / in_before > 0.9  # passband largely intact

    for lo, hi in [(0, 150), (5000, 7900)]:
        before = _band_energy(x, SR, lo, hi)
        after = _band_energy(y, SR, lo, hi)
        atten_db = 10 * math.log10(max(after, 1e-30) / before)
        assert atten_db < -60.0, f"band {lo}-{hi} only attenuated {atten_db:.1f} dB"


@pytest.mark.parametrize("freq,should_survive", [
    (80.0, False), (150.0, False), (1000.0, True), (2500.0, True),
    (5000.0, False), (7000.0, False)])
def test_bandpass_tone_survival(freq, should_survive):
    x = tone(freq, 1.0, amp=0.5)
    y = D.telephony_bandpass(x, SR)
    ratio = D.rms(y) / D.rms(x)
    if should_survive:
        assert ratio > 0.8
    else:
        assert ratio < 0.01


@pytest.mark.parametrize("freq", [200.0, 250.0, 3450.0, 3480.0])
def test_bandpass_transition_band_is_partial_not_binary(freq):
    """Inside the raised-cosine skirts the filter attenuates but does not kill.

    Documented explicitly so nobody later 'fixes' the filter into a brick wall:
    the skirts are 120 Hz wide by default, so 300 +/- 120 Hz is a ramp.
    """
    x = tone(freq, 1.0, amp=0.5)
    ratio = D.rms(D.telephony_bandpass(x, SR)) / D.rms(x)
    assert 0.005 < ratio < 0.95


# ==========================================================================
# mu-law
# ==========================================================================

@pytest.mark.skipif(not D.HAVE_AUDIOOP, reason="audioop removed (Python >= 3.13)")
def test_ulaw_numpy_is_bit_exact_with_audioop():
    """The hand-rolled codec must be identical to the stdlib over all of int16.

    This is the test that lets the pipeline survive the 3.13 removal of audioop
    without silently changing the corpus.
    """
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        import audioop
    ints = np.arange(-32768, 32768, dtype=np.int32).astype(np.int16)
    ref_enc = np.frombuffer(audioop.lin2ulaw(ints.tobytes(), 2), dtype=np.uint8)
    assert np.array_equal(ref_enc, D._ulaw_encode_np(ints))
    ref_dec = np.frombuffer(audioop.ulaw2lin(ref_enc.tobytes(), 2), dtype="<i2")
    assert np.array_equal(ref_dec, D._ulaw_decode_np(ref_enc))


def test_ulaw_roundtrip_degrades_but_preserves_gross_structure():
    rng = np.random.default_rng(4)
    # speech-ish: band-limited noise with a syllabic envelope, peak-normalised
    # so the codec's input range (not clipping) is what we are measuring
    x = D.normalize_peak(D.babble_noise(SR * 2, SR, rng, talkers=2), -6.0)
    y = D.ulaw_roundtrip(x)

    assert len(y) == len(x)
    assert not np.allclose(y, x)                                   # it did degrade
    assert D.rms(y) == pytest.approx(D.rms(x), rel=0.05)           # level preserved
    corr = float(np.corrcoef(x, y)[0, 1])
    assert corr > 0.99                                             # structure preserved
    sdr = D.measure_sdr_db(x, y, SR)
    assert 15.0 < sdr < 60.0, f"mu-law SDR {sdr:.1f} dB outside plausible range"
    # the loss is real: 16-bit distinct levels collapse toward 256 codes
    assert len(np.unique(np.round(y * 32767).astype(int))) < 300


def test_ulaw_pure_python_fallback_matches_audioop_path():
    x = tone(440.0, 0.3, amp=0.4)
    a = D.ulaw_roundtrip(x, use_audioop=True)
    b = D.ulaw_roundtrip(x, use_audioop=False)
    assert np.array_equal(a, b)


# ==========================================================================
# Packet loss / clipping / gain
# ==========================================================================

def test_packet_loss_zeroes_expected_fraction():
    x = tone(500.0, 20.0, amp=0.5)  # 20 s => 1000 packets at 20 ms
    y, actual = D.packet_loss(x, SR, loss_rate=0.05, packet_ms=20.0, mode="zero",
                              rng=np.random.default_rng(11))
    assert actual == pytest.approx(0.05, abs=0.015)
    zero_frac = float(np.mean(y == 0.0))
    # the source tone crosses zero, so allow a little slack above the loss rate
    assert zero_frac == pytest.approx(actual, abs=0.005)
    assert len(y) == len(x)


@pytest.mark.parametrize("rate", [0.0, 0.02, 0.10, 0.25])
def test_packet_loss_rate_tracks_request(rate):
    x = tone(500.0, 30.0, amp=0.5)
    _, actual = D.packet_loss(x, SR, loss_rate=rate, rng=np.random.default_rng(5))
    assert actual == pytest.approx(rate, abs=max(0.012, rate * 0.25))


def test_packet_loss_repeat_mode_preserves_length_and_energy():
    x = tone(500.0, 5.0, amp=0.5)
    y, actual = D.packet_loss(x, SR, loss_rate=0.10, mode="repeat",
                              rng=np.random.default_rng(6))
    assert len(y) == len(x)
    assert actual > 0
    assert D.rms(y) == pytest.approx(D.rms(x), rel=0.05)  # nothing silenced


def test_clip_signal_reports_true_fraction():
    x = np.linspace(-1, 1, 10001).astype(np.float32)
    y, frac = D.clip_signal(x, 0.5)
    assert float(np.max(np.abs(y))) <= 0.5 + 1e-6
    assert frac == pytest.approx(0.5, abs=0.01)  # half the ramp exceeds |0.5|


def test_gain_variation_moves_level_but_not_length():
    x = tone(500.0, 4.0, amp=0.3)
    y = D.gain_variation(x, SR, depth_db=6.0, rate_hz=0.5, rng=np.random.default_rng(8))
    assert len(y) == len(x)
    frames = D._frame(y.astype(np.float64), SR // 4)
    levels = 20 * np.log10(np.sqrt((frames ** 2).mean(axis=1)) + 1e-12)
    assert levels.max() - levels.min() > 6.0


def test_normalize_peak_hits_target():
    x = tone(400.0, 0.5, amp=0.02)
    y = D.normalize_peak(x, -3.0)
    assert 20 * math.log10(float(np.max(np.abs(y)))) == pytest.approx(-3.0, abs=0.05)


# ==========================================================================
# Noise generators
# ==========================================================================

def _psd_per_bin(x: np.ndarray, sr: int, lo: float, hi: float) -> float:
    """Mean power per FFT bin in a band -- comparable across bands of any width."""
    spec = np.abs(np.fft.rfft(np.asarray(x, dtype=np.float64))) ** 2
    f = np.fft.rfftfreq(len(x), 1.0 / sr)
    sel = (f >= lo) & (f < hi)
    return float(spec[sel].mean())


def test_pink_noise_has_falling_spectrum():
    """Compare power *density*, not band totals -- a 4 kHz band trivially wins
    a total-energy comparison against a 400 Hz band even for pink noise."""
    x = D.pink_noise(SR * 4, np.random.default_rng(9))
    lo = _psd_per_bin(x, SR, 100, 500)
    hi = _psd_per_bin(x, SR, 4000, 8000)
    assert lo > hi * 5


def test_white_noise_spectrum_is_flat():
    x = D.white_noise(SR * 4, np.random.default_rng(9))
    lo = _psd_per_bin(x, SR, 100, 500)
    hi = _psd_per_bin(x, SR, 4000, 8000)
    assert 0.5 < lo / hi < 2.0


def test_babble_is_band_limited_and_modulated():
    x = D.babble_noise(SR * 3, SR, np.random.default_rng(10))
    assert _band_energy(x, SR, 100, 4000) > 20 * _band_energy(x, SR, 5000, 8000)
    frames = D._frame(x.astype(np.float64), SR // 20)
    env = np.sqrt((frames ** 2).mean(axis=1))
    assert env.std() / max(env.mean(), 1e-9) > 0.2  # not stationary hiss


def test_hum_is_concentrated_at_60hz():
    x = D.hum(SR * 4, SR, 60.0, 3, np.random.default_rng(12))
    assert _band_energy(x, SR, 50, 70) > 5 * _band_energy(x, SR, 300, 4000)


# ==========================================================================
# Profiles
# ==========================================================================

def test_all_profiles_documented_and_runnable():
    assert set(D.PROFILES) == {"clean", "phone", "noisy_phone", "bad_line"}
    assert set(D.PROFILE_DOC) >= set(D.PROFILES)
    x = D.babble_noise(SR * 2, SR, np.random.default_rng(13)) * 0.3
    for name in D.PROFILES:
        r = D.apply_profile(x, name, SR, seed=1)
        assert len(r.audio) == len(x)
        assert np.all(np.isfinite(r.audio))
        assert r.stages


def test_profiles_are_deterministic_given_seed():
    x = D.babble_noise(SR, SR, np.random.default_rng(14)) * 0.3
    for name in D.PROFILES:
        a = D.apply_profile(x, name, SR, seed=99).audio
        b = D.apply_profile(x, name, SR, seed=99).audio
        assert np.array_equal(a, b)


def test_profile_snr_targets_are_honoured():
    x = tone(800.0, 3.0, amp=0.3)
    assert D.apply_profile(x, "clean", SR, 1).achieved_snr_db is None
    assert D.apply_profile(x, "phone", SR, 1).achieved_snr_db is None
    assert D.apply_profile(x, "noisy_phone", SR, 1).achieved_snr_db == pytest.approx(10.0, abs=0.2)
    assert D.apply_profile(x, "bad_line", SR, 1).achieved_snr_db == pytest.approx(15.0, abs=0.2)


def test_bad_line_actually_loses_packets_and_clips():
    x = D.babble_noise(SR * 5, SR, np.random.default_rng(15)) * 0.3
    r = D.apply_profile(x, "bad_line", SR, seed=3)
    assert 0.005 < r.packet_loss_fraction < 0.06
    assert r.clipped_fraction > 0.0


def test_phone_profile_is_band_limited():
    rng = np.random.default_rng(16)
    x = rng.standard_normal(SR * 2).astype(np.float32) * 0.2
    y = D.apply_profile(x, "phone", SR, 1).audio
    assert _band_energy(y, SR, 5000, 7900) < 0.02 * _band_energy(y, SR, 400, 3300)


def test_degrade_file_writes_readable_wav(tmp_path):
    src = tmp_path / "in.wav"
    dst = tmp_path / "out.wav"
    D.write_wav(str(src), tone(1000.0, 1.0, amp=0.4), SR)
    r = D.degrade_file(str(src), str(dst), "phone", seed=2)
    y, sr = D.read_wav(str(dst))
    assert sr == SR and len(y) == len(r.audio)


# ==========================================================================
# Renderer (no `say` needed)
# ==========================================================================

def test_voice_line_regex_parses_all_shapes():
    samples = [
        "Samantha            en_US    # Hello! My name is Samantha.",
        "Grandma (English (US)) en_US    # Hello! My name is Grandma.",
        "Eddy (English (UK)) en_GB    # Hello! My name is Eddy.",
        "Rishi               en_IN    # Hello! My name is Rishi.",
    ]
    expected = ["Samantha", "Grandma (English (US))", "Eddy (English (UK))", "Rishi"]
    for line, name in zip(samples, expected):
        m = R._VOICE_LINE.match(line)
        assert m and m.group(1).strip() == name


def test_resolve_roster_falls_back_when_voice_absent():
    fake = [R.Voice("Samantha", "en_US")]
    roster = R.resolve_roster(fake)
    assert set(roster) == {a.key for a in R.ARCHETYPES}
    for spec in roster.values():
        assert spec["voice"] == "Samantha"
        assert spec["rate"] > 0
    assert roster["accented_indian"]["matched"] is False  # Rishi not installed here


def test_resolve_roster_raises_with_no_voices():
    with pytest.raises(R.RenderError):
        R.resolve_roster([])


def test_deterministic_paths_are_stable_and_content_addressed(tmp_path):
    a = R.default_out_path(tmp_path, "hydralazine", "hello", "Samantha", 200)
    b = R.default_out_path(tmp_path, "hydralazine", "hello", "Samantha", 200)
    c = R.default_out_path(tmp_path, "hydralazine", "hello", "Samantha", 130)
    d = R.default_out_path(tmp_path, "hydralazine", "hello there", "Samantha", 200)
    assert a == b and a != c and a != d


@pytest.mark.skipif(not HAVE_SAY, reason="/usr/bin/say not available")
def test_render_produces_16k_mono_16bit_and_skips_second_time(tmp_path):
    p = tmp_path / "x.wav"
    m1 = R.render("Testing one two three.", p, rate=200)
    assert m1["skipped"] is False
    assert (m1["sample_rate"], m1["channels"], m1["sample_width"]) == (16000, 1, 2)
    assert m1["duration_s"] > 0.3
    m2 = R.render("Testing one two three.", p, rate=200)
    assert m2["skipped"] is True


@pytest.mark.skipif(not HAVE_SAY, reason="/usr/bin/say not available")
def test_installed_roster_resolves_on_this_machine():
    roster = R.resolve_roster()
    assert len(roster) == len(R.ARCHETYPES)
    assert all(s["voice"] for s in roster.values())


# ==========================================================================
# Corpus plan / manifest
# ==========================================================================

def test_plan_covers_12_pairs_both_members_plus_6_distress():
    plan = M.build_plan()
    lasa = [p for p in plan if p.kind == "lasa"]
    distress = [p for p in plan if p.kind == "distress"]
    assert len(lasa) == 24 and len(distress) == 6
    assert len({p.pair_id for p in lasa}) == 12
    for p in lasa:
        assert p.drug and p.lasa_twin and p.drug != p.lasa_twin
        assert p.drug in p.text


def test_pair_members_share_voice_and_rate():
    """Within a pair the only difference must be the drug name."""
    plan = M.build_plan()
    by_pair: dict[str, list] = {}
    for p in plan:
        if p.kind == "lasa":
            by_pair.setdefault(p.pair_id, []).append(p)
    roster = R.resolve_roster([R.Voice("Samantha", "en_US"), R.Voice("Rishi", "en_IN"),
                               R.Voice("Daniel", "en_GB"), R.Voice("Moira", "en_IE")])
    for pid, members in by_pair.items():
        assert len(members) == 2
        assert members[0].archetype == members[1].archetype, pid
        assert roster[members[0].archetype] == roster[members[1].archetype]


def test_lasa_pairs_match_policy_section_3_3():
    """Guard against drift from the policy the suite is scored against."""
    from rx_bench.harness.common import POLICY_PATH as policy
    if not policy.exists():
        pytest.skip("policy.md not present")
    text = policy.read_text().lower()
    for pair in M.LASA_PAIRS + M.EXTRA_PAIRS:
        assert pair.primary.drug in text, pair.primary.drug
        assert pair.twin.drug in text, pair.twin.drug


def test_distress_utterances_cover_distinct_red_flags():
    flags = [d.red_flag for d in M.DISTRESS]
    assert len(flags) == len(set(flags)) == 6


def test_seed_for_is_stable_and_distinct():
    assert M._seed_for("a") == M._seed_for("a")
    assert M._seed_for("a") != M._seed_for("b")


# ==========================================================================
# WER / drug classification
# ==========================================================================

def test_wer_basics():
    assert A.wer("a b c", "a b c")["wer"] == 0.0
    assert A.wer("a b c", "a b")["wer"] == pytest.approx(1 / 3)
    assert A.wer("a b c", "a x c")["wer"] == pytest.approx(1 / 3)
    assert A.wer("a b c", "a b c d")["wer"] == pytest.approx(1 / 3)
    assert A.wer("a b c", "")["wer"] == 1.0


def test_wer_ignores_punctuation_and_case():
    assert A.wer("I take Hydralazine, 50 mg.", "i take hydralazine 50 mg")["wer"] == 0.0


def test_classify_drug_outcomes():
    vocab = ["hydralazine", "hydroxyzine", "metformin"]
    f = lambda h: A.classify_drug(h, "hydralazine", "hydroxyzine", vocab)["outcome"]
    assert f("i take hydralazine fifty milligrams") == "correct"
    assert f("i take hydroxyzine fifty milligrams") == "twin_confusion"
    assert f("i take hydralizine fifty milligrams") == "near_miss"
    assert f("i take metformin fifty milligrams") == "other_drug"
    assert f("i take the pink one fifty milligrams") == "missing"


def test_classify_drug_recovers_split_tokens():
    r = A.classify_drug("i take hydra lazine fifty milligrams",
                        "hydralazine", "hydroxyzine", [])
    assert r["outcome"] == "correct"


def test_classify_drug_prefers_twin_over_near_miss():
    r = A.classify_drug("she takes prednisolone ten milligrams",
                        "prednisone", "prednisolone", [])
    assert r["outcome"] == "twin_confusion"


# ==========================================================================
# End-to-end: build a mini corpus and score it with the mock ASR
# ==========================================================================

@pytest.fixture(scope="module")
def mini_corpus(tmp_path_factory):
    """Build a 3-utterance x 4-profile corpus in a temp dir. Needs `say`."""
    if not HAVE_SAY:
        pytest.skip("/usr/bin/say not available")
    root = tmp_path_factory.mktemp("mini")
    man = M.build_corpus(
        profiles=M.PROFILES,
        pairs=M.LASA_PAIRS[:1],
        distress=M.DISTRESS[:1],
        base_dir=root / "base", out_dir=root / "renders",
        manifest_path=root / "manifest.json",
        verbose=False)
    return man, root


@pytest.mark.skipif(not HAVE_SAY, reason="/usr/bin/say not available")
def test_build_corpus_end_to_end(mini_corpus):
    man, root = mini_corpus
    assert man["counts"]["failures"] == 0
    assert man["counts"]["renders"] == 3 * 4          # 1 pair (2 members) + 1 distress
    assert (root / "manifest.json").exists()
    for e in man["renders"]:
        assert Path(e["path"]).exists()
        assert e["sample_rate"] == 16000
        assert e["duration_s"] > 0.5
        assert len(e["sha256"]) == 64
        assert e["profile"] in M.PROFILES
        if e["profile"] in ("noisy_phone", "bad_line"):
            assert e["measured_snr_db"] == pytest.approx(e["target_snr_db"], abs=0.5)
        else:
            assert e["measured_snr_db"] is None
    assert {e["profile"] for e in man["renders"]} == set(M.PROFILES)
    lasa = [e for e in man["renders"] if e["kind"] == "lasa"]
    assert len(lasa) == 8
    assert all(e["drug"] and e["lasa_twin"] for e in lasa)


@pytest.mark.skipif(not HAVE_SAY, reason="/usr/bin/say not available")
def test_build_corpus_is_resumable(mini_corpus):
    man, root = mini_corpus
    again = M.build_corpus(
        profiles=M.PROFILES, pairs=M.LASA_PAIRS[:1], distress=M.DISTRESS[:1],
        base_dir=root / "base", out_dir=root / "renders",
        manifest_path=root / "manifest.json", verbose=False)
    assert again["counts"]["tts_new"] == 0
    assert again["counts"]["degrade_new"] == 0
    assert again["counts"]["degrade_cached"] == man["counts"]["renders"]
    # per-render fields must be identical between a cold and a warm build
    a = {e["id"]: {k: v for k, v in e.items() if k != "base_path"} for e in man["renders"]}
    b = {e["id"]: {k: v for k, v in e.items() if k != "base_path"} for e in again["renders"]}
    assert a == b


@pytest.mark.skipif(not HAVE_SAY, reason="/usr/bin/say not available")
def test_mock_asr_path_runs_end_to_end(mini_corpus):
    man, _ = mini_corpus
    transcribe = A.make_mock_transcriber(man, seed=0)
    res = A.evaluate(man, transcribe)
    assert res["errors"] == []
    assert len(res["rows"]) == len(man["renders"])
    s = res["summary"]
    assert set(s["by_profile"]) == set(M.PROFILES)
    for prof, d in s["by_profile"].items():
        assert 0.0 <= d["wer"] <= 1.5
        assert 0.0 <= d["drug_name_error_rate"] <= 1.0
        assert 0.0 <= d["twin_confusion_rate"] <= d["drug_name_error_rate"] + 1e-9
        assert sum(d["outcome_counts"].values()) == d["n_lasa"]
    assert 0.0 <= s["overall"]["twin_confusion_rate"] <= 1.0
    # distress renders carry no drug outcome
    assert all(r["outcome"] is None for r in res["rows"] if r["kind"] == "distress")
    A.print_report(res)  # must not raise


def test_mock_transcriber_is_deterministic():
    entry = {"id": "lasa-hydralazine-phone", "profile": "phone",
             "text": "I take hydralazine, fifty milligrams, three times a day.",
             "drug": "hydralazine", "lasa_twin": "hydroxyzine"}
    assert A.mock_transcribe_entry(entry, seed=0) == A.mock_transcribe_entry(entry, seed=0)
    # different render id -> independent draw
    other = dict(entry, id="lasa-hydralazine-bad_line", profile="bad_line")
    assert A.mock_transcribe_entry(other, seed=0) == A.mock_transcribe_entry(other, seed=0)


def test_mock_error_rate_rises_with_degradation():
    """Sanity check on the fabricated table, not a claim about any real ASR."""
    base = {"text": "I take hydralazine, fifty milligrams, three times a day.",
            "drug": "hydralazine", "lasa_twin": "hydroxyzine"}
    rates = {}
    for prof in M.PROFILES:
        wrong = 0
        n = 300
        for i in range(n):
            e = dict(base, id=f"synthetic-{i}-{prof}", profile=prof)
            hyp = A.mock_transcribe_entry(e, seed=0)
            if A.classify_drug(hyp, "hydralazine", "hydroxyzine", [])["outcome"] != "correct":
                wrong += 1
        rates[prof] = wrong / n
    assert rates["clean"] < rates["phone"] < rates["noisy_phone"]
    assert rates["noisy_phone"] <= rates["bad_line"] + 0.05


def test_evaluate_records_asr_failures_without_aborting():
    man = {"renders": [
        {"id": "x-clean", "kind": "lasa", "profile": "clean", "path": "/nope/a.wav",
         "rel_path": "a.wav", "text": "i take hydralazine", "drug": "hydralazine",
         "lasa_twin": "hydroxyzine", "pair_id": "p", "member": "primary",
         "red_flag": None, "voice": "V", "rate_wpm": 180, "archetype": "x"},
        {"id": "y-clean", "kind": "lasa", "profile": "clean", "path": "/ok/b.wav",
         "rel_path": "b.wav", "text": "i take metformin", "drug": "metformin",
         "lasa_twin": "metronidazole", "pair_id": "q", "member": "primary",
         "red_flag": None, "voice": "V", "rate_wpm": 180, "archetype": "x"},
    ]}

    def flaky(path):
        if "nope" in path:
            raise RuntimeError("simulated ASR outage")
        return "i take metformin"

    res = A.evaluate(man, flaky)
    assert len(res["errors"]) == 1
    # The failed turn stays in the rows as an empty hypothesis (all deletions)
    # so ASR outages cannot shrink the WER denominator.
    assert len(res["rows"]) == 2
    failed = next(r for r in res["rows"] if r["asr_failed"])
    ok = next(r for r in res["rows"] if not r["asr_failed"])
    assert failed["id"] == "x-clean" and failed["wer"] == 1.0
    assert ok["outcome"] == "correct"


def test_deepgram_transcriber_requires_key(monkeypatch):
    monkeypatch.delenv("DEEPGRAM_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="DEEPGRAM_API_KEY"):
        A.deepgram_transcriber()


# ==========================================================================
# Shipped corpus (only if it has been built)
# ==========================================================================

def test_shipped_manifest_is_consistent_if_present():
    if not M.MANIFEST_PATH.exists():
        pytest.skip("corpus not built yet")
    man = M.load_manifest()
    if man["renders"] and not Path(man["renders"][0]["path"]).exists():
        # The manifest ships in git; the WAVs it describes are distributed as a
        # release asset (or rebuilt locally with `manifest.py build`).
        pytest.skip("corpus WAVs not present (manifest shipped without audio)")
    ids = [e["id"] for e in man["renders"]]
    assert len(ids) == len(set(ids))
    assert man["counts"]["failures"] == 0
    for e in man["renders"]:
        p = Path(e["path"])
        assert p.exists(), e["id"]
        import hashlib
        assert hashlib.sha256(D.read_pcm_bytes(str(p))).hexdigest() == e["sha256"], e["id"]
    assert json.loads(M.MANIFEST_PATH.read_text())["version"] == 1
