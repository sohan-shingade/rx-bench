"""Pure-Python audio degradation for the Tier-A (audio) eval corpus.

No ffmpeg, no sox, no cloud. Everything here runs on numpy + the stdlib so it
works on a laptop with nothing installed.

Everything operates on float32 mono signals in [-1.0, 1.0] at a known sample
rate (the corpus is 16 kHz mono 16-bit -- see ``render.py``).

Signal-chain philosophy
-----------------------
Degradations are applied in the order they would happen physically on a real
call, because order changes the result:

    speech -> (room noise added) -> (telephone band-pass) -> (codec)
           -> (packet loss / jitter) -> (AGC gain wobble, clipping)

Adding babble *before* the band-pass means the noise is band-limited by the
same channel that limits the speech, which is what a real phone does. Adding it
after would produce out-of-band noise that no phone line could carry, and would
flatter or punish an ASR for the wrong reason.

SNR convention
--------------
SNR is computed against the **speech-active RMS** of the signal, not the
whole-file RMS. TTS output has leading and trailing silence; including it would
drag the "signal" power down and silently make every file noisier than the
label claims. ``active_rms`` keeps only frames within ``ACTIVE_FLOOR_DB`` of the
loudest frame. For a full-duration synthetic tone active RMS == global RMS, so
the convention is a no-op on test signals.
"""

from __future__ import annotations

import math
import struct
import wave
from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np

# --------------------------------------------------------------------------
# stdlib mu-law if available, hand-rolled if not.
#
# `audioop` is present in CPython 3.12 (this venv) but was removed in 3.13.
# The numpy fallback below is bit-exact with G.711 mu-law, so behaviour does
# not change when the stdlib module disappears.
# --------------------------------------------------------------------------
try:  # pragma: no cover - environment dependent
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        import audioop as _audioop  # type: ignore
    HAVE_AUDIOOP = True
except Exception:  # pragma: no cover
    _audioop = None  # type: ignore
    HAVE_AUDIOOP = False


SAMPLE_RATE = 16000
ACTIVE_FLOOR_DB = -35.0  # frames this far below the peak frame are "silence"
_EPS = 1e-12


# ==========================================================================
# WAV I/O
# ==========================================================================

def read_wav(path: str) -> tuple[np.ndarray, int]:
    """Read a mono 16-bit PCM WAV into float32 in [-1, 1]. Returns (x, sr).

    Stereo input is downmixed to mono. 8-bit and 32-bit PCM are accepted so
    that the module survives being pointed at something odd.
    """
    with wave.open(str(path), "rb") as w:
        n_ch = w.getnchannels()
        width = w.getsampwidth()
        sr = w.getframerate()
        raw = w.readframes(w.getnframes())

    if width == 2:
        x = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    elif width == 1:
        x = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    elif width == 4:
        x = np.frombuffer(raw, dtype="<i4").astype(np.float32) / 2147483648.0
    else:
        raise ValueError(f"unsupported sample width {width} in {path}")

    if n_ch > 1:
        usable = (len(x) // n_ch) * n_ch
        x = x[:usable].reshape(-1, n_ch).mean(axis=1)
    return np.ascontiguousarray(x, dtype=np.float32), sr


def write_wav(path: str, x: np.ndarray, sr: int = SAMPLE_RATE) -> None:
    """Write float32 [-1, 1] mono to a 16-bit PCM WAV."""
    pcm = np.clip(np.asarray(x, dtype=np.float64), -1.0, 1.0)
    # symmetric scaling, round-half-away-from-zero, then clamp to int16 range
    ints = np.clip(np.round(pcm * 32767.0), -32768, 32767).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(int(sr))
        w.writeframes(ints.tobytes())


def pcm_bytes(x: np.ndarray) -> bytes:
    """The exact int16 payload that ``write_wav`` would emit. Used for hashing."""
    pcm = np.clip(np.asarray(x, dtype=np.float64), -1.0, 1.0)
    return np.clip(np.round(pcm * 32767.0), -32768, 32767).astype("<i2").tobytes()


def read_pcm_bytes(path: str) -> bytes:
    """The raw int16 payload already on disk, for cache comparison.

    Note ``read_wav`` divides by 32768 while ``write_wav`` multiplies by 32767,
    so a read/write round trip is not the identity on the float representation.
    Comparing the int16 payloads directly avoids that trap.
    """
    with wave.open(str(path), "rb") as w:
        if w.getsampwidth() != 2 or w.getnchannels() != 1:
            raise ValueError(f"{path} is not mono 16-bit")
        return w.readframes(w.getnframes())


# ==========================================================================
# Level / SNR measurement
# ==========================================================================

def rms(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(x * x)))


def dbfs(x: np.ndarray) -> float:
    return 20.0 * math.log10(max(rms(x), _EPS))


def _frame(x: np.ndarray, frame_len: int) -> np.ndarray:
    n = (len(x) // frame_len) * frame_len
    if n == 0:
        return x.reshape(1, -1)
    return x[:n].reshape(-1, frame_len)


def active_mask(x: np.ndarray, sr: int = SAMPLE_RATE,
                frame_ms: float = 20.0, floor_db: float = ACTIVE_FLOOR_DB) -> np.ndarray:
    """Boolean per-sample mask of speech-active regions (simple energy VAD)."""
    frame_len = max(1, int(sr * frame_ms / 1000.0))
    frames = _frame(np.asarray(x, dtype=np.float64), frame_len)
    fr = np.sqrt(np.mean(frames * frames, axis=1))
    peak = float(fr.max()) if fr.size else 0.0
    if peak <= _EPS:
        return np.ones(len(x), dtype=bool)
    thresh = peak * (10.0 ** (floor_db / 20.0))
    keep = fr >= thresh
    mask = np.repeat(keep, frame_len)
    if len(mask) < len(x):  # tail partial frame: inherit last decision
        mask = np.concatenate([mask, np.full(len(x) - len(mask), keep[-1])])
    return mask[: len(x)]


def active_rms(x: np.ndarray, sr: int = SAMPLE_RATE) -> float:
    """RMS over speech-active frames only. Falls back to global RMS if silent."""
    m = active_mask(x, sr)
    if not m.any():
        return rms(x)
    return rms(np.asarray(x)[m])


def snr_db(signal: np.ndarray, noise: np.ndarray, sr: int = SAMPLE_RATE,
           use_active: bool = True) -> float:
    """SNR in dB of ``signal`` against ``noise``, both time-domain arrays."""
    s = active_rms(signal, sr) if use_active else rms(signal)
    n = rms(noise)
    return 20.0 * math.log10(max(s, _EPS) / max(n, _EPS))


def measure_sdr_db(clean: np.ndarray, degraded: np.ndarray,
                   sr: int = SAMPLE_RATE) -> float:
    """Signal-to-distortion ratio of a degraded render against its clean base.

    Gain-matched least-squares projection first, so a pure level change scores
    as ~no distortion. For purely additive noise this reduces to the SNR; for
    band-pass / codec / dropout it is a *distortion* figure, not an SNR -- the
    manifest reports the two separately for exactly this reason.
    """
    a = np.asarray(clean, dtype=np.float64)
    b = np.asarray(degraded, dtype=np.float64)
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    denom = float(np.dot(a, a))
    alpha = float(np.dot(a, b) / denom) if denom > _EPS else 0.0
    resid = b - alpha * a
    return 20.0 * math.log10(max(active_rms(alpha * a, sr), _EPS) / max(rms(resid), _EPS))


# ==========================================================================
# Noise generators
# ==========================================================================

def white_noise(n: int, rng: np.random.Generator) -> np.ndarray:
    return rng.standard_normal(n).astype(np.float32)


def pink_noise(n: int, rng: np.random.Generator) -> np.ndarray:
    """1/f noise via spectral shaping of white noise."""
    if n <= 1:
        return np.zeros(n, dtype=np.float32)
    spec = np.fft.rfft(rng.standard_normal(n))
    f = np.arange(len(spec), dtype=np.float64)
    f[0] = 1.0
    spec = spec / np.sqrt(f)
    out = np.fft.irfft(spec, n)
    return _unit(out).astype(np.float32)


def hum(n: int, sr: int = SAMPLE_RATE, freq: float = 60.0,
        harmonics: int = 3, rng: np.random.Generator | None = None) -> np.ndarray:
    """Mains hum: fundamental plus decaying harmonics, slight phase jitter."""
    t = np.arange(n, dtype=np.float64) / sr
    rng = rng or np.random.default_rng(0)
    out = np.zeros(n, dtype=np.float64)
    for k in range(1, harmonics + 1):
        phase = rng.uniform(0, 2 * math.pi)
        out += (1.0 / k) * np.sin(2 * math.pi * freq * k * t + phase)
    return _unit(out).astype(np.float32)


def babble_noise(n: int, sr: int = SAMPLE_RATE, rng: np.random.Generator | None = None,
                 talkers: int = 6) -> np.ndarray:
    """Synthetic 'several people talking in the background'.

    Each pseudo-talker is band-limited noise in a formant-ish band, amplitude
    modulated by a slow random envelope (syllable rate ~2-6 Hz) so the result
    has speech-like temporal structure rather than being stationary hiss. This
    is *not* recorded babble -- it is a stand-in with the right gross spectral
    and temporal statistics.
    """
    rng = rng or np.random.default_rng(0)
    if n <= 1:
        return np.zeros(n, dtype=np.float32)
    out = np.zeros(n, dtype=np.float64)
    bands = [(120, 500), (300, 900), (500, 1600), (900, 2600), (1600, 3600), (250, 3200)]
    for i in range(talkers):
        lo, hi = bands[i % len(bands)]
        src = _fft_bandpass(rng.standard_normal(n), sr, lo, hi, transition=0.4 * (hi - lo))
        # syllabic envelope: one slow rate plus a slower "pause" modulation
        t = np.arange(n, dtype=np.float64) / sr
        syll = rng.uniform(2.0, 6.0)
        pause = rng.uniform(0.2, 0.6)
        env = (0.55 + 0.45 * np.sin(2 * math.pi * syll * t + rng.uniform(0, 6.28)))
        env *= (0.5 + 0.5 * np.sin(2 * math.pi * pause * t + rng.uniform(0, 6.28))) ** 2
        out += _unit(src) * env
    return _unit(out).astype(np.float32)


def _unit(x: np.ndarray) -> np.ndarray:
    r = rms(x)
    return np.asarray(x, dtype=np.float64) / max(r, _EPS)


NOISE_KINDS: dict[str, Callable[..., np.ndarray]] = {
    "white": lambda n, sr, rng: white_noise(n, rng),
    "pink": lambda n, sr, rng: pink_noise(n, rng),
    "babble": lambda n, sr, rng: babble_noise(n, sr, rng),
    "hum": lambda n, sr, rng: hum(n, sr, 60.0, 3, rng),
}


def make_noise(kinds: Sequence[str], n: int, sr: int, rng: np.random.Generator,
               weights: Sequence[float] | None = None) -> np.ndarray:
    """Sum several noise kinds with the given RMS weights, return unit-RMS."""
    weights = list(weights or [1.0] * len(kinds))
    acc = np.zeros(n, dtype=np.float64)
    for kind, w in zip(kinds, weights):
        acc += w * _unit(NOISE_KINDS[kind](n, sr, rng))
    return _unit(acc).astype(np.float32)


def add_noise(x: np.ndarray, target_snr_db: float, sr: int = SAMPLE_RATE,
              kinds: Sequence[str] = ("babble",), rng: np.random.Generator | None = None,
              weights: Sequence[float] | None = None) -> tuple[np.ndarray, float]:
    """Add noise scaled to hit ``target_snr_db`` against speech-active RMS.

    Returns ``(noisy, achieved_snr_db)``. The achieved value is *measured* from
    the produced signals, not assumed, so the manifest can record the truth.
    """
    rng = rng or np.random.default_rng(0)
    n = len(x)
    noise = make_noise(kinds, n, sr, rng, weights)
    sig = active_rms(x, sr)
    target_noise_rms = sig / (10.0 ** (target_snr_db / 20.0))
    noise = noise * (target_noise_rms / max(rms(noise), _EPS))
    out = np.asarray(x, dtype=np.float64) + noise
    return out.astype(np.float32), snr_db(x, noise, sr)


# ==========================================================================
# Filters
# ==========================================================================

def _fft_bandpass(x: np.ndarray, sr: int, low: float, high: float,
                  transition: float = 100.0) -> np.ndarray:
    """Zero-phase FFT band-pass with raised-cosine transition bands."""
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    if n == 0:
        return x
    spec = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(n, 1.0 / sr)
    gain = np.ones_like(freqs)

    tr = max(transition, 1.0)
    # low edge
    lo_start, lo_end = max(low - tr, 0.0), low
    gain = np.where(freqs <= lo_start, 0.0, gain)
    band = (freqs > lo_start) & (freqs < lo_end)
    if band.any():
        gain[band] = 0.5 * (1 - np.cos(math.pi * (freqs[band] - lo_start) / max(lo_end - lo_start, _EPS)))
    # high edge
    hi_start, hi_end = high, high + tr
    gain = np.where(freqs >= hi_end, 0.0, gain)
    band = (freqs > hi_start) & (freqs < hi_end)
    if band.any():
        gain[band] = 0.5 * (1 + np.cos(math.pi * (freqs[band] - hi_start) / max(hi_end - hi_start, _EPS)))

    return np.fft.irfft(spec * gain, n)


def telephony_bandpass(x: np.ndarray, sr: int = SAMPLE_RATE,
                       low: float = 300.0, high: float = 3400.0,
                       transition: float = 120.0) -> np.ndarray:
    """~300-3400 Hz band-pass: the POTS / G.711 passband.

    This is the single most consequential degradation for a phone agent. It
    removes the low-frequency energy that carries voicing strength and the
    high-frequency energy that distinguishes fricatives -- which is precisely
    why '-azine' and '-yzine' collapse into each other on a phone.
    """
    return _fft_bandpass(x, sr, low, high, transition).astype(np.float32)


# ==========================================================================
# Codec
# ==========================================================================

_ULAW_BIAS = 0x84
_ULAW_CLIP = 8159  # in the 14-bit domain G.711 actually companding operates in
_ULAW_SEG_END = np.array([0x3F, 0x7F, 0xFF, 0x1FF, 0x3FF, 0x7FF, 0xFFF, 0x1FFF], dtype=np.int32)


def _ulaw_encode_np(pcm: np.ndarray) -> np.ndarray:
    """G.711 mu-law encode int16 -> uint8, vectorised.

    Bit-exact with ``audioop.lin2ulaw(..., 2)``: it companded in the 14-bit
    domain (input >> 2) with BIAS 0x84 >> 2, which is the detail every
    hand-rolled implementation gets wrong. Verified against audioop in
    ``test_audio.py`` whenever the stdlib module is present.
    """
    v = np.asarray(pcm, dtype=np.int32) >> 2  # arithmetic shift to 14-bit
    neg = v < 0
    mag = np.where(neg, -v, v)
    mask = np.where(neg, 0x7F, 0xFF).astype(np.int32)
    mag = np.minimum(mag, _ULAW_CLIP) + (_ULAW_BIAS >> 2)
    seg = np.searchsorted(_ULAW_SEG_END, mag, side="left").astype(np.int32)
    uval = np.where(
        seg >= 8,
        0x7F,
        ((seg << 4) | ((mag >> np.minimum(seg + 1, 30)) & 0x0F)),
    )
    return ((uval ^ mask) & 0xFF).astype(np.uint8)


def _ulaw_decode_np(code: np.ndarray) -> np.ndarray:
    """G.711 mu-law decode uint8 -> int16, vectorised. Matches ``audioop.ulaw2lin``."""
    u = (~np.asarray(code, dtype=np.int32)) & 0xFF
    sign = u & 0x80
    exponent = (u >> 4) & 0x07
    mantissa = u & 0x0F
    mag = (((mantissa << 3) + _ULAW_BIAS) << exponent) - _ULAW_BIAS
    out = np.where(sign != 0, -mag, mag)
    return np.clip(out, -32768, 32767).astype(np.int16)


def ulaw_roundtrip(x: np.ndarray, use_audioop: bool | None = None) -> np.ndarray:
    """G.711 mu-law encode/decode round trip: 16-bit -> 8-bit log -> 16-bit.

    Costs roughly 8 bits of resolution with logarithmic companding, so quiet
    passages lose relatively more than loud ones -- the classic "sounds like a
    phone" quantisation floor.

    Note: input outside [-1, 1] is clipped on the way into the codec, exactly as
    a real encoder would. Every profile in this module peak-normalises before
    calling this, so that path is not exercised in the corpus; if you call it
    directly on an un-normalised signal, normalise first or you will measure
    clipping and call it companding.
    """
    ints = np.clip(np.round(np.asarray(x, dtype=np.float64) * 32767.0), -32768, 32767).astype(np.int16)
    prefer = HAVE_AUDIOOP if use_audioop is None else (use_audioop and HAVE_AUDIOOP)
    if prefer:
        enc = _audioop.lin2ulaw(ints.tobytes(), 2)
        dec = _audioop.ulaw2lin(enc, 2)
        out = np.frombuffer(dec, dtype="<i2")
    else:
        out = _ulaw_decode_np(_ulaw_encode_np(ints))
    return (out.astype(np.float32) / 32768.0)


# ==========================================================================
# Channel impairments
# ==========================================================================

def packet_loss(x: np.ndarray, sr: int = SAMPLE_RATE, loss_rate: float = 0.02,
                packet_ms: float = 20.0, mode: str = "zero",
                rng: np.random.Generator | None = None) -> tuple[np.ndarray, float]:
    """Drop or repeat whole packets at ``loss_rate``.

    ``mode='zero'``   -> silence the packet (naive jitter buffer)
    ``mode='repeat'`` -> repeat the previous packet (packet-loss concealment)

    Length is preserved so degraded renders stay sample-aligned with the clean
    base; that alignment is what makes the manifest's distortion figure and any
    later forced-alignment work possible.

    Returns ``(y, actual_loss_fraction)`` where the fraction is measured, not
    requested.
    """
    rng = rng or np.random.default_rng(0)
    y = np.array(x, dtype=np.float32, copy=True)
    plen = max(1, int(sr * packet_ms / 1000.0))
    n_pkt = max(1, len(y) // plen)
    drop = rng.random(n_pkt) < loss_rate
    lost = 0
    for i in np.nonzero(drop)[0]:
        a, b = i * plen, (i + 1) * plen
        if mode == "repeat" and i > 0:
            y[a:b] = y[a - plen:a][: b - a]
        else:
            y[a:b] = 0.0
        lost += b - a
    return y, (lost / max(len(y), 1))


def clip_signal(x: np.ndarray, threshold: float = 0.85) -> tuple[np.ndarray, float]:
    """Hard-clip at ``threshold``. Returns (y, fraction of samples clipped)."""
    x = np.asarray(x, dtype=np.float32)
    hit = float(np.mean(np.abs(x) > threshold))
    return np.clip(x, -threshold, threshold).astype(np.float32), hit


def gain_variation(x: np.ndarray, sr: int = SAMPLE_RATE, depth_db: float = 6.0,
                   rate_hz: float = 0.35, rng: np.random.Generator | None = None) -> np.ndarray:
    """Slow level drift, as from AGC hunting or a handset moving off the mouth."""
    rng = rng or np.random.default_rng(0)
    t = np.arange(len(x), dtype=np.float64) / sr
    lfo = np.sin(2 * math.pi * rate_hz * t + rng.uniform(0, 6.28))
    return (np.asarray(x, dtype=np.float64) * 10.0 ** (depth_db * lfo / 20.0)).astype(np.float32)


def apply_gain_db(x: np.ndarray, db: float) -> np.ndarray:
    return (np.asarray(x, dtype=np.float32) * (10.0 ** (db / 20.0))).astype(np.float32)


def normalize_peak(x: np.ndarray, target_dbfs: float = -3.0) -> np.ndarray:
    """Peak-normalise. Used so profiles differ in channel, not in loudness."""
    peak = float(np.max(np.abs(x))) if len(x) else 0.0
    if peak <= _EPS:
        return np.asarray(x, dtype=np.float32)
    target = 10.0 ** (target_dbfs / 20.0)
    return (np.asarray(x, dtype=np.float64) * (target / peak)).astype(np.float32)


# ==========================================================================
# Named profiles
# ==========================================================================

@dataclass
class ProfileResult:
    audio: np.ndarray
    sample_rate: int
    achieved_snr_db: float | None = None      # additive-noise SNR, None if no noise
    packet_loss_fraction: float = 0.0
    clipped_fraction: float = 0.0
    stages: list[str] = field(default_factory=list)


def _p_clean(x, sr, rng) -> ProfileResult:
    return ProfileResult(normalize_peak(x), sr, None, 0.0, 0.0, ["normalize(-3 dBFS)"])


def _p_phone(x, sr, rng) -> ProfileResult:
    y = telephony_bandpass(x, sr)
    y = normalize_peak(y)
    y = ulaw_roundtrip(y)
    return ProfileResult(y, sr, None, 0.0, 0.0,
                         ["bandpass(300-3400 Hz)", "normalize(-3 dBFS)", "mu-law G.711 round trip"])


def _p_noisy_phone(x, sr, rng) -> ProfileResult:
    y, achieved = add_noise(x, 10.0, sr, kinds=("babble", "pink"), weights=(1.0, 0.3), rng=rng)
    y = telephony_bandpass(y, sr)
    y = normalize_peak(y)
    y = ulaw_roundtrip(y)
    return ProfileResult(y, sr, achieved, 0.0, 0.0,
                         ["+babble/pink noise @ 10 dB SNR", "bandpass(300-3400 Hz)",
                          "normalize(-3 dBFS)", "mu-law G.711 round trip"])


def _p_bad_line(x, sr, rng) -> ProfileResult:
    y, achieved = add_noise(x, 15.0, sr, kinds=("babble", "white", "hum"),
                            weights=(1.0, 0.5, 0.35), rng=rng)
    y = telephony_bandpass(y, sr)
    y = gain_variation(y, sr, depth_db=5.0, rate_hz=0.35, rng=rng)
    y = normalize_peak(y, -1.0)
    y = apply_gain_db(y, 8.0)               # drive it into the limiter
    y, clipped = clip_signal(y, 0.95)
    y = normalize_peak(y, -3.0)
    y = ulaw_roundtrip(y)
    y, lost = packet_loss(y, sr, loss_rate=0.02, packet_ms=20.0, mode="zero", rng=rng)
    return ProfileResult(y, sr, achieved, lost, clipped,
                         ["+babble/white/hum noise @ 15 dB SNR", "bandpass(300-3400 Hz)",
                          "gain wobble +/-5 dB @ 0.35 Hz", "normalize(-1 dBFS)",
                          "+8 dB drive", "hard clip @ 0.95", "normalize(-3 dBFS)",
                          "mu-law G.711 round trip",
                          "2% packet loss (20 ms packets, zeroed)"])


PROFILES: dict[str, Callable[[np.ndarray, int, np.random.Generator], ProfileResult]] = {
    "clean": _p_clean,
    "phone": _p_phone,
    "noisy_phone": _p_noisy_phone,
    "bad_line": _p_bad_line,
}

PROFILE_DOC = {
    "clean": (
        "Unmodified TTS, peak-normalised to -3 dBFS. Full 16 kHz band, no noise, "
        "no codec. The ceiling: any error here is a TTS or ASR failure, not a "
        "channel failure."
    ),
    "phone": (
        "300-3400 Hz band-pass -> peak-normalise -3 dBFS -> G.711 mu-law encode/"
        "decode. No added noise. This is the default and the most realistic "
        "single degradation for a telephone agent."
    ),
    "noisy_phone": (
        "Synthetic 6-talker babble + pink noise added at 10 dB SNR (measured "
        "against speech-active RMS) BEFORE the 300-3400 Hz band-pass, then "
        "peak-normalise and G.711 mu-law. Models a caller in a waiting room or "
        "a car."
    ),
    "bad_line": (
        "Babble + white + 60 Hz mains hum at 15 dB SNR -> 300-3400 Hz band-pass "
        "-> +/-5 dB gain wobble at 0.35 Hz (AGC hunting) -> normalise -1 dBFS -> "
        "+8 dB drive -> hard clip at 0.95 -> renormalise -3 dBFS -> G.711 mu-law "
        "-> 2% packet loss in 20 ms packets, zeroed. Note the additive noise is "
        "MILDER than noisy_phone (15 dB vs 10 dB) by design: bad_line's damage is "
        "temporal and non-linear (dropouts, clipping, level hunting), not "
        "spectral. The two profiles probe different failure modes and neither "
        "strictly dominates."
    ),
}


def apply_profile(x: np.ndarray, profile: str, sr: int = SAMPLE_RATE,
                  seed: int = 0) -> ProfileResult:
    """Apply a named degradation profile. Deterministic given ``seed``."""
    if profile not in PROFILES:
        raise KeyError(f"unknown profile {profile!r}; have {sorted(PROFILES)}")
    rng = np.random.default_rng(seed)
    return PROFILES[profile](np.asarray(x, dtype=np.float32), sr, rng)


def degrade_file(in_path: str, out_path: str, profile: str, seed: int = 0) -> ProfileResult:
    """Read a WAV, apply a profile, write a WAV. Returns the ProfileResult."""
    x, sr = read_wav(in_path)
    res = apply_profile(x, profile, sr, seed)
    write_wav(out_path, res.audio, sr)
    return res


# ==========================================================================
# Small CLI for eyeballing a single file
# ==========================================================================

if __name__ == "__main__":  # pragma: no cover
    import argparse

    ap = argparse.ArgumentParser(description="Apply a degradation profile to a WAV.")
    ap.add_argument("input")
    ap.add_argument("output")
    ap.add_argument("--profile", default="phone", choices=sorted(PROFILES))
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    r = degrade_file(a.input, a.output, a.profile, a.seed)
    clean, sr = read_wav(a.input)
    print(f"{a.profile}: {' -> '.join(r.stages)}")
    print(f"  snr={r.achieved_snr_db}  loss={r.packet_loss_fraction:.4f} "
          f"clipped={r.clipped_fraction:.4f} sdr={measure_sdr_db(clean, r.audio, sr):.2f} dB")
