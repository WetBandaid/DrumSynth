import numpy as np

from .config import SAMPLE_RATE
from .effects import (
    apply_bitcrush,
    apply_filter,
    apply_patch_delay,
    apply_patch_reverb,
    apply_saturation,
    apply_transient,
    modulation_gain,
    modulation_signal,
    one_pole_lowpass,
)
from .model import TrackState


def envelope(length: int, seconds: float, curve: float = 6.0) -> np.ndarray:
    t = np.linspace(0.0, 1.0, length, endpoint=False)
    return np.exp(-curve * t / max(seconds, 0.001)).astype(np.float32)


def sine_sweep(duration: float, start_hz: float, end_hz: float) -> np.ndarray:
    length = max(1, int(duration * SAMPLE_RATE))
    sweep = np.geomspace(max(15.0, start_hz), max(15.0, end_hz), length).astype(np.float32)
    phase = 2.0 * np.pi * np.cumsum(sweep) / SAMPLE_RATE
    return np.sin(phase).astype(np.float32)


def sine_sweep_with_lfo(
    duration: float,
    start_hz: float,
    end_hz: float,
    lfo: np.ndarray,
    amount: float,
) -> np.ndarray:
    length = max(1, int(duration * SAMPLE_RATE))
    sweep = np.geomspace(max(15.0, start_hz), max(15.0, end_hz), length).astype(np.float32)
    semitones = np.clip(lfo[:length], -1.0, 1.0) * amount * 36.0
    sweep *= (2.0 ** (semitones / 12.0)).astype(np.float32)
    phase = 2.0 * np.pi * np.cumsum(sweep) / SAMPLE_RATE
    return np.sin(phase).astype(np.float32)


def noise(duration: float) -> np.ndarray:
    return np.random.uniform(-1.0, 1.0, max(1, int(duration * SAMPLE_RATE))).astype(np.float32)


def highpass_like(x: np.ndarray) -> np.ndarray:
    return np.concatenate(([x[0]], np.diff(x))).astype(np.float32)


def normalize_peak(x: np.ndarray, peak: float = 1.0) -> np.ndarray:
    current = float(np.max(np.abs(x))) if len(x) else 0.0
    if current <= 0.0001:
        return x.astype(np.float32, copy=False)
    return (x * (peak / current)).astype(np.float32)


def smooth_noise_envelope(length: int, decay_seconds: float, attack_seconds: float = 0.004) -> np.ndarray:
    env = envelope(length, decay_seconds, 6.0)
    attack_len = min(length, max(2, int(attack_seconds * SAMPLE_RATE)))
    if attack_len > 1:
        env[:attack_len] *= np.sin(np.linspace(0.0, np.pi / 2.0, attack_len, dtype=np.float32)) ** 2
    return env.astype(np.float32)


def timed_envelope(length: int, decay_seconds: float, attack_seconds: float = 0.001) -> np.ndarray:
    t = np.arange(length, dtype=np.float32) / SAMPLE_RATE
    env = np.exp(-t / max(decay_seconds, 0.001)).astype(np.float32)
    attack_len = min(length, max(1, int(attack_seconds * SAMPLE_RATE)))
    if attack_len > 1:
        env[:attack_len] *= np.linspace(0.0, 1.0, attack_len, dtype=np.float32)
    return env


def band_limited_noise(length: int, low_hz: float, high_hz: float) -> np.ndarray:
    x = np.random.uniform(-1.0, 1.0, max(1, length)).astype(np.float32)
    high = x - one_pole_lowpass(x, low_hz, 0.05)
    return one_pole_lowpass(high, high_hz, 0.04).astype(np.float32)


def spectral_noise(length: int, low_hz: float, high_hz: float) -> np.ndarray:
    length = max(1, length)
    x = np.random.uniform(-1.0, 1.0, length).astype(np.float32)
    spectrum = np.fft.rfft(x)
    frequencies = np.fft.rfftfreq(length, 1.0 / SAMPLE_RATE)
    mask = (frequencies >= low_hz) & (frequencies <= high_hz)
    spectrum *= mask
    return normalize_peak(np.fft.irfft(spectrum, n=length).astype(np.float32), 1.0)


def kick_body(
    length: int,
    start_hz: float,
    end_hz: float,
    pitch: float,
    tone_seconds: float,
    mute: float = 0.0,
    pitch_mod: np.ndarray | None = None,
    bass_mode: bool = False,
) -> np.ndarray:
    mute = float(np.clip(mute, 0.0, 1.0))
    t = np.arange(length, dtype=np.float32) / SAMPLE_RATE
    if bass_mode:
        target_hz = float(np.clip(end_hz * pitch, 24.0, 320.0))
        snap_hz = float(np.clip(start_hz * pitch, target_hz * 1.15, target_hz * 3.2))
        bend = np.exp(-t / 0.018).astype(np.float32)
        frequency = target_hz + (snap_hz - target_hz) * bend
        if pitch_mod is not None:
            semitones = np.clip(pitch_mod[:length], -1.0, 1.0) * 7.0
            frequency *= (2.0 ** (semitones / 12.0)).astype(np.float32)

        phase = 2.0 * np.pi * np.cumsum(frequency) / SAMPLE_RATE
        bass = np.sin(phase)
        bass *= timed_envelope(length, max(0.22, tone_seconds * (0.95 - mute * 0.25)), 0.0025)
        warm = np.sin(phase * 2.0 + 0.18)
        warm *= timed_envelope(length, max(0.090, tone_seconds * 0.34), 0.0012)
        click = np.sin(phase * 5.0 + 0.7)
        click *= timed_envelope(length, 0.018, 0.00035)
        body = bass * 0.96 + warm * 0.13 + click * 0.035
        body = np.tanh(body * (1.08 + mute * 0.18)).astype(np.float32)
        return normalize_peak(one_pole_lowpass(body, 1450.0 - mute * 520.0, 0.035), 0.96)

    fast_bend = np.exp(-t / (0.020 + 0.010 * (1.0 - mute))).astype(np.float32)
    slow_bend = np.exp(-t / (0.060 + 0.080 * (1.0 - mute))).astype(np.float32)
    bend = np.clip(fast_bend * (0.86 - mute * 0.10) + slow_bend * (0.14 + mute * 0.10), 0.0, 1.0)
    frequency = (end_hz + (start_hz - end_hz) * bend) * pitch
    if pitch_mod is not None:
        semitones = np.clip(pitch_mod[:length], -1.0, 1.0) * 16.0
        frequency *= (2.0 ** (semitones / 12.0)).astype(np.float32)

    phase = 2.0 * np.pi * np.cumsum(frequency) / SAMPLE_RATE
    punch = np.sin(phase + 0.10 * np.sin(phase * 2.0))
    punch *= timed_envelope(length, max(0.060, tone_seconds * (0.40 - mute * 0.18)), 0.0012)
    sub = np.sin(phase * 0.5 + 0.35)
    sub *= timed_envelope(length, max(0.075, tone_seconds * (0.60 - mute * 0.30)), 0.003)
    membrane = np.sin(phase * 1.72 + 0.55)
    membrane *= timed_envelope(length, max(0.020, tone_seconds * (0.15 - mute * 0.08)), 0.0008)
    beater_tone = np.sin(phase * 3.15 + 0.9)
    beater_tone *= timed_envelope(length, max(0.020, tone_seconds * 0.055), 0.00035)
    thud = np.sin(2.0 * np.pi * max(32.0, end_hz * pitch * 0.78) * t + 0.2)
    thud *= timed_envelope(length, 0.035 + 0.070 * mute, 0.0015)

    body = punch * 0.84 + sub * (0.24 + mute * 0.18)
    body += thud * (0.10 + mute * 0.18)
    body += membrane * (0.14 * (1.0 - mute * 0.75))
    body += beater_tone * (0.07 * (1.0 - mute * 0.45))
    body = np.tanh(body * (1.14 + mute * 0.34)).astype(np.float32)
    cutoff = 1650.0 - mute * 760.0
    return normalize_peak(one_pole_lowpass(body, cutoff, 0.05), 0.96)


def kick_noise(length: int, noise_seconds: float, mute: float = 0.0) -> np.ndarray:
    mute = float(np.clip(mute, 0.0, 1.0))
    n = np.zeros(length, dtype=np.float32)
    beater_len = min(length, int(max(0.020, noise_seconds * 0.60) * SAMPLE_RATE))
    if beater_len > 0:
        beater = band_limited_noise(beater_len, 120.0, 2600.0 - mute * 900.0)
        beater *= timed_envelope(beater_len, max(0.010, noise_seconds * (0.34 - mute * 0.10)), 0.00045)
        n[:beater_len] += beater * (0.62 + mute * 0.12)

    air_len = min(length, int(max(0.045, noise_seconds * 1.25) * SAMPLE_RATE))
    if air_len > 0:
        air = band_limited_noise(air_len, 35.0, 520.0 - mute * 170.0)
        air *= timed_envelope(air_len, max(0.025, noise_seconds * (1.0 - mute * 0.45)), 0.003)
        n[:air_len] += air * (0.30 + mute * 0.20)
    return normalize_peak(n, 0.72)


def snare_noise(length: int, noise_seconds: float) -> np.ndarray:
    shell = band_limited_noise(length, 620.0, 5200.0)
    shell *= timed_envelope(length, max(0.070, noise_seconds * 0.34), 0.0006)

    wire = band_limited_noise(length, 1850.0, 9800.0)
    wire_env = timed_envelope(length, max(0.18, noise_seconds * 0.70), 0.0015)
    delay = min(length, int(0.0035 * SAMPLE_RATE))
    if delay > 0:
        wire_env[:delay] *= np.linspace(0.12, 1.0, delay, dtype=np.float32)
    wire *= wire_env

    tail = band_limited_noise(length, 2600.0, 8500.0)
    tail *= timed_envelope(length, max(0.12, noise_seconds * 0.48), 0.010)
    return normalize_peak(shell * 0.44 + wire * 0.70 + tail * 0.24, 0.86)


def snare_body(
    length: int,
    start_hz: float,
    end_hz: float,
    pitch: float,
    tone_seconds: float,
    pitch_mod: np.ndarray | None = None,
) -> np.ndarray:
    t = np.arange(length, dtype=np.float32) / SAMPLE_RATE
    bend = np.exp(-t / 0.030).astype(np.float32)
    frequency = (end_hz + (start_hz - end_hz) * bend) * pitch
    if pitch_mod is not None:
        semitones = np.clip(pitch_mod[:length], -1.0, 1.0) * 20.0
        frequency *= (2.0 ** (semitones / 12.0)).astype(np.float32)

    phase = 2.0 * np.pi * np.cumsum(frequency) / SAMPLE_RATE
    head = np.sin(phase) * timed_envelope(length, max(0.055, tone_seconds * 0.34), 0.0005)
    shell = np.sin(phase * 1.58 + 0.55) * timed_envelope(length, max(0.035, tone_seconds * 0.20), 0.0004)
    ring = np.sin(phase * 2.18 + 1.1) * timed_envelope(length, max(0.026, tone_seconds * 0.14), 0.0004)
    snap = band_limited_noise(length, 900.0, 4300.0)
    snap *= timed_envelope(length, max(0.018, tone_seconds * 0.10), 0.0003)

    body = head * 0.74 + shell * 0.26 + ring * 0.12 + snap * 0.16
    return normalize_peak(one_pole_lowpass(body.astype(np.float32), 4800.0, 0.06), 0.90)


def scaled_noise_band(base_low: float, base_high: float, pitch: float) -> tuple[float, float]:
    low = max(20.0, base_low * pitch)
    high = min(SAMPLE_RATE * 0.45, base_high * pitch)
    return low, max(low + 40.0, high)


def hat_noise(
    length: int,
    open_hat: bool,
    noise_seconds: float,
    crash: bool = False,
    pitch: float = 1.0,
) -> np.ndarray:
    t = np.arange(length, dtype=np.float32) / SAMPLE_RATE
    metallic = np.zeros(length, dtype=np.float32)
    if crash:
        crash_pitch = float(np.clip(pitch, 0.55, 1.8))
        tail_scale = float(np.clip(noise_seconds / 1.32, 0.06, 1.35))
        flex = 0.55 * np.sin(2.0 * np.pi * 1.4 * t + 0.3)
        flex += 0.30 * np.sin(2.0 * np.pi * 2.7 * t + 1.6)
        flex += 0.15 * np.sin(2.0 * np.pi * 4.1 * t + 2.4)
        flex = flex.astype(np.float32)
        wobble_depth = timed_envelope(length, max(0.08, 0.95 * tail_scale), 0.006)
        flex_amount = 0.006 * wobble_depth
        resonant = np.zeros(length, dtype=np.float32)
        for freq, gain, decay in (
            (520.0, 0.08, 2.20),
            (820.0, 0.12, 1.95),
            (1390.0, 0.17, 1.70),
            (2140.0, 0.22, 1.42),
            (3160.0, 0.27, 1.12),
            (4520.0, 0.23, 0.88),
            (6380.0, 0.16, 0.62),
            (8900.0, 0.08, 0.38),
        ):
            freq *= crash_pitch
            slow_wobble = np.sin(2.0 * np.pi * (1.3 + gain * 8.0) * t + gain * 9.0)
            fast_wobble = np.sin(2.0 * np.pi * (5.8 + gain * 13.0) * t + freq * 0.001)
            wobble = 1.0 + flex_amount * flex + 0.004 * wobble_depth * slow_wobble
            wobble += 0.0018 * wobble_depth * fast_wobble
            phase = 2.0 * np.pi * np.cumsum(freq * wobble.astype(np.float32)) / SAMPLE_RATE
            partial = np.sin(phase + gain * 3.0)
            partial += 0.28 * np.sin(phase * 1.017 + 0.8 + flex * 0.18)
            partial += 0.12 * np.sin(phase * 0.997 + 1.9)
            amp_wobble = 0.78 + 0.22 * np.sin(2.0 * np.pi * (2.0 + gain * 7.5) * t + gain)
            resonant += (
                partial.astype(np.float32)
                * timed_envelope(length, max(0.035, decay * tail_scale), 0.006)
                * amp_wobble.astype(np.float32)
                * gain
            )

        stick = spectral_noise(length, *scaled_noise_band(750.0, 16500.0, crash_pitch))
        stick *= timed_envelope(length, 0.018, 0.00015)
        bell_ping = np.sin(2.0 * np.pi * 3850.0 * crash_pitch * t + 0.7)
        bell_ping += 0.45 * np.sin(2.0 * np.pi * 6120.0 * crash_pitch * t + 1.4)
        bell_ping *= timed_envelope(length, 0.030, 0.00025)

        low, high = scaled_noise_band(650.0, 15500.0, crash_pitch)
        impact = spectral_noise(length, low, high)
        impact *= timed_envelope(length, max(0.075, noise_seconds * 0.12), 0.00035)
        low, high = scaled_noise_band(780.0, 9800.0, crash_pitch)
        body = spectral_noise(length, low, high)
        body *= timed_envelope(length, max(0.045, noise_seconds * 0.82), 0.006)
        low, high = scaled_noise_band(420.0, 4200.0, crash_pitch)
        low_wash = spectral_noise(length, low, high)
        low_wash *= timed_envelope(length, max(0.060, noise_seconds * 1.28), 0.018)
        low, high = scaled_noise_band(4300.0, 13500.0, crash_pitch)
        shimmer = spectral_noise(length, low, high)
        shimmer *= timed_envelope(length, max(0.035, noise_seconds * 0.48), 0.028)

        tail_bloom = 1.0 - np.exp(-t / 0.035)
        high_tail_duck = 0.10 + 0.90 * np.exp(-t / max(0.045, 0.42 * tail_scale))
        shimmer *= (tail_bloom * high_tail_duck).astype(np.float32)
        low_wash *= tail_bloom.astype(np.float32)
        moving_body = body + resonant * 0.28 + low_wash * 0.35
        flex_gain = 0.82 + 0.18 * flex * wobble_depth
        cymbal = stick * 0.42 + bell_ping.astype(np.float32) * 0.13
        cymbal += impact * 0.78 + moving_body * 0.78 + resonant * 0.54
        cymbal += low_wash * 0.24 + shimmer * 0.16
        cymbal *= flex_gain.astype(np.float32)
        dark_tail = one_pole_lowpass(cymbal.astype(np.float32), 5200.0, 0.04)
        dark_start = max(0.025, 0.24 * tail_scale)
        dark_mix = np.clip((t - dark_start) / max(0.10, 1.45 * tail_scale), 0.0, 0.72)
        cymbal = cymbal * (1.0 - dark_mix) + dark_tail * dark_mix
        return normalize_peak(np.tanh(cymbal * 1.18).astype(np.float32), 0.92)

    for freq, gain in (
        (5150.0, 0.28),
        (6910.0, 0.25),
        (8420.0, 0.21),
        (10150.0, 0.16),
        (12200.0, 0.11),
    ):
        metallic += np.sin(2.0 * np.pi * freq * t) * gain
    wash = band_limited_noise(length, 5600.0 if open_hat else 6200.0, 15500.0)
    decay = max(0.30 if open_hat else 0.045, noise_seconds * (0.70 if open_hat else 0.55))
    env = timed_envelope(length, decay, 0.0005)
    tail = timed_envelope(length, max(0.34, noise_seconds * 0.78), 0.010) if open_hat else 0.0
    cymbal = (wash * (0.86 if open_hat else 0.72) + metallic * (0.34 if open_hat else 0.22)) * env
    if open_hat:
        cymbal += band_limited_noise(length, 8200.0, 15800.0) * tail * 0.08
    return normalize_peak(cymbal, 0.86 if open_hat else 0.82)


def clap_noise(length: int, duration: float, noise_seconds: float) -> np.ndarray:
    claps = np.zeros(length, dtype=np.float32)
    burst_specs = (
        (0.000, 0.020, 1.75),
        (0.006, 0.024, 0.34),
        (0.014, 0.028, 0.20),
        (0.023, 0.034, 0.12),
    )

    for offset_seconds, burst_seconds, gain in burst_specs:
        offset = int(offset_seconds * SAMPLE_RATE)
        if offset >= length:
            continue
        burst_len = min(length - offset, int(burst_seconds * SAMPLE_RATE))
        burst = band_limited_noise(burst_len, 850.0, 8800.0)
        burst_count = min(burst_len, len(burst))
        burst = burst[:burst_count] * smooth_noise_envelope(
            burst_count, burst_seconds, 0.0016
        )
        claps[offset : offset + burst_count] += burst * gain

    tail_start = min(length, int(0.012 * SAMPLE_RATE))
    tail_len = length - tail_start
    if tail_len > 0:
        tail = band_limited_noise(tail_len, 850.0, 5400.0)
        tail_count = min(tail_len, len(tail))
        tail = tail[:tail_count] * smooth_noise_envelope(
            tail_count, max(0.18, noise_seconds), 0.006
        )
        fade_in = min(tail_count, int(0.012 * SAMPLE_RATE))
        if fade_in > 1:
            tail[:fade_in] *= np.linspace(0.0, 1.0, fade_in, dtype=np.float32)
        claps[tail_start : tail_start + tail_count] += tail * 0.42

    body = band_limited_noise(length, 550.0, 3400.0)
    body_count = min(length, len(body))
    body = body[:body_count] * smooth_noise_envelope(
        body_count, max(0.15, noise_seconds * 0.75), 0.003
    )
    claps[:body_count] += body * 0.20
    claps *= smooth_noise_envelope(length, max(0.22, noise_seconds * 0.8), 0.0012)
    return normalize_peak(claps, 0.78)


def rim_clack(length: int, pitch: float, noise_seconds: float) -> np.ndarray:
    t = np.arange(length, dtype=np.float32) / SAMPLE_RATE
    clack = np.zeros(length, dtype=np.float32)

    for offset_seconds, freq, gain, decay_seconds in (
        (0.000, 1850.0, 1.00, 0.055),
        (0.006, 2450.0, 0.62, 0.045),
    ):
        offset = int(offset_seconds * SAMPLE_RATE)
        if offset >= length:
            continue
        count = length - offset
        local_t = t[:count]
        partial = np.sin(2.0 * np.pi * freq * pitch * local_t)
        partial += 0.42 * np.sin(2.0 * np.pi * freq * 1.47 * pitch * local_t)
        partial *= timed_envelope(count, decay_seconds, 0.0008)
        clack[offset : offset + count] += partial.astype(np.float32) * gain

    noise_len = min(length, int(max(0.060, noise_seconds * 1.2) * SAMPLE_RATE))
    woody = band_limited_noise(noise_len, 1200.0, 5200.0)
    woody *= timed_envelope(noise_len, max(0.050, noise_seconds * 1.2), 0.001)
    clack[:noise_len] += woody * 0.46
    return normalize_peak(clack, 0.9)


def low_tom_body(
    length: int,
    start_hz: float,
    end_hz: float,
    pitch: float,
    tone_seconds: float,
    pitch_mod: np.ndarray | None = None,
) -> np.ndarray:
    t = np.arange(length, dtype=np.float32) / SAMPLE_RATE
    fast_bend = np.exp(-t / 0.045).astype(np.float32)
    slow_bend = np.exp(-t / 0.120).astype(np.float32)
    bend = np.clip(fast_bend * 0.74 + slow_bend * 0.26, 0.0, 1.0)
    frequency = end_hz + (start_hz - end_hz) * bend
    frequency *= pitch
    if pitch_mod is not None:
        semitones = np.clip(pitch_mod[:length], -1.0, 1.0) * 18.0
        frequency *= (2.0 ** (semitones / 12.0)).astype(np.float32)

    phase = 2.0 * np.pi * np.cumsum(frequency) / SAMPLE_RATE
    thump = np.sin(phase + 0.08 * np.sin(phase * 1.7))
    thump *= timed_envelope(length, max(0.095, tone_seconds * 0.30), 0.0012)
    shell = np.sin(phase * 1.47 + 0.35)
    shell *= timed_envelope(length, max(0.050, tone_seconds * 0.15), 0.0007)
    membrane = np.sin(phase * 2.18 + 0.9)
    membrane *= timed_envelope(length, max(0.030, tone_seconds * 0.09), 0.00045)
    mallet = band_limited_noise(length, 95.0, 2600.0)
    mallet *= timed_envelope(length, max(0.020, tone_seconds * 0.055), 0.00035)
    body = thump * 0.92 + shell * 0.25 + membrane * 0.13 + mallet * 0.18
    body = np.tanh(body * 1.10).astype(np.float32)
    body = one_pole_lowpass(body, 2900.0, 0.07)
    return normalize_peak(body, 0.95)


def perc_body(length: int, start_hz: float, end_hz: float, pitch: float, tone_seconds: float) -> np.ndarray:
    t = np.arange(length, dtype=np.float32) / SAMPLE_RATE
    bend = np.exp(-t / 0.035).astype(np.float32)
    frequency = (end_hz + (start_hz - end_hz) * bend) * pitch
    phase = 2.0 * np.pi * np.cumsum(frequency) / SAMPLE_RATE
    tone = np.sin(phase) * timed_envelope(length, tone_seconds * 0.34, 0.001)
    tone += 0.28 * np.sin(phase * 1.82 + 0.4) * timed_envelope(length, tone_seconds * 0.18, 0.0008)
    return normalize_peak(tone.astype(np.float32), 0.88)


def step_note_resonance(
    length: int,
    note_hz: float,
    instrument: str,
    tone_seconds: float,
    noise_seconds: float,
) -> np.ndarray:
    note_hz = float(np.clip(note_hz, 24.0, 320.0))
    t = np.arange(length, dtype=np.float32) / SAMPLE_RATE
    layer = np.zeros(length, dtype=np.float32)

    if instrument in {"Closed Hat", "Open Hat", "Crash"}:
        multipliers = (16.0, 23.0, 31.0) if instrument != "Crash" else (8.0, 13.0, 21.0)
        decay = max(0.040, noise_seconds * (0.42 if instrument == "Closed Hat" else 0.62))
        attack = 0.0005
        gains = (0.26, 0.18, 0.11)
    elif instrument in {"Snare", "Clap", "Rim"}:
        multipliers = (2.0, 3.0, 5.0)
        decay = max(0.035, tone_seconds * 0.42)
        attack = 0.00045
        gains = (0.34, 0.18, 0.08)
    else:
        multipliers = (1.0, 2.0, 3.0)
        decay = max(0.055, tone_seconds * 0.62)
        attack = 0.0008
        gains = (0.42, 0.16, 0.06)

    note_env = timed_envelope(length, decay, attack)
    for multiplier, gain in zip(multipliers, gains):
        freq = float(np.clip(note_hz * multiplier, 30.0, SAMPLE_RATE * 0.42))
        phase = 2.0 * np.pi * freq * t
        layer += np.sin(phase).astype(np.float32) * gain

    layer *= note_env
    return normalize_peak(layer, 0.78)


def make_hit(
    track: TrackState,
    velocity: float = 1.0,
    lfo_phase_override: float | tuple[float | None, float | None] | None = None,
) -> np.ndarray:
    velocity = float(np.clip(velocity, 0.0, 1.5))
    volume = float(np.clip(track.volume, 0.0, 1.2)) * velocity
    uses_step_notes = bool(getattr(track, "bass_enabled", False))
    is_tuned_bass = uses_step_notes and track.instrument == "Kick"
    pitch = float(np.clip(track.pitch, 0.35, 7.0 if uses_step_notes else 2.5))
    decay_scale = 0.35 + float(np.clip(track.decay, 0.05, 1.0)) * 1.3
    tone_seconds = max(0.005, track.tone_decay * decay_scale)
    noise_seconds = max(0.005, track.noise_decay * decay_scale)
    duration = max(0.035, tone_seconds, noise_seconds, 0.05)
    if track.instrument == "Crash":
        duration = max(duration, noise_seconds * 2.15, 0.18)
    duration = min(3.6 if track.instrument == "Crash" else 2.0, duration)
    length = max(1, int(duration * SAMPLE_RATE))
    hit = np.zeros(length, dtype=np.float32)
    phase1, phase2 = split_lfo_phase_override(lfo_phase_override)
    lfo1 = make_lfo(track, length, phase1, prefix="lfo")
    lfo2 = make_lfo(track, length, phase2, prefix="lfo2")
    env_mod = make_env_mod(track, length)
    kick_mute = float(np.clip(getattr(track, "kick_mute", 0.0), 0.0, 1.0))

    tone_level = float(np.clip(track.tone_level, 0.0, 1.5))
    if tone_level > 0.0:
        pitch_env = 2.0 ** (float(np.clip(track.pitch_env_amount, -48.0, 48.0)) / 12.0)
        env_mix = min(1.0, max(0.0, track.pitch_env_decay / max(duration, 0.001)))
        sweep_start = track.tone_start * pitch * pitch_env
        sweep_end = track.tone_end * pitch
        pitch_mod = modulation_signal(track, "Pitch", lfo1, lfo2, env_mod)
        if track.instrument == "Kick":
            tone = kick_body(
                length,
                track.tone_start,
                track.tone_end,
                pitch * pitch_env,
                tone_seconds,
                kick_mute,
                pitch_mod,
                is_tuned_bass,
            )
            tone_env = 1.0
        elif track.instrument == "Snare":
            tone = snare_body(length, track.tone_start, track.tone_end, pitch * pitch_env, tone_seconds, pitch_mod)
            tone_env = 1.0
        elif track.instrument == "Low Tom":
            tone = low_tom_body(length, track.tone_start, track.tone_end, pitch * pitch_env, tone_seconds, pitch_mod)
            tone_env = 1.0
        elif track.instrument == "Perc":
            tone = perc_body(length, track.tone_start, track.tone_end, pitch * pitch_env, tone_seconds)
            tone_env = 1.0
        elif pitch_mod is not None:
            tone = sine_sweep_with_lfo(duration, sweep_start, sweep_end, pitch_mod, 1.0)
            tone_env = envelope(length, tone_seconds * (0.5 + env_mix), 7.0)
        else:
            tone = sine_sweep(duration, sweep_start, sweep_end)
            tone_env = envelope(length, tone_seconds * (0.5 + env_mix), 7.0)
        tone_gain = modulation_gain(track, "Tone Level", lfo1, lfo2, env_mod, 0.0, 2.0)
        hit += tone[:length] * tone_env * tone_level * tone_gain

    noise_level = float(np.clip(track.noise_level, 0.0, 1.5))
    if noise_level > 0.0:
        if track.instrument == "Clap":
            n = clap_noise(length, duration, noise_seconds)
        elif track.instrument == "Rim":
            n = rim_clack(length, pitch, noise_seconds)
        elif track.instrument == "Low Tom":
            n = np.zeros(length, dtype=np.float32)
            noise_len = min(length, int(max(0.055, noise_seconds) * SAMPLE_RATE))
            head = band_limited_noise(noise_len, 85.0, 3200.0)
            head *= timed_envelope(noise_len, max(0.026, noise_seconds * 0.48), 0.00055)
            n[:noise_len] = head
        elif track.instrument == "Kick":
            n = kick_noise(length, noise_seconds, kick_mute)
        elif track.instrument == "Snare":
            n = snare_noise(length, noise_seconds)
        elif track.instrument in {"Closed Hat", "Open Hat", "Crash"}:
            n = hat_noise(
                length,
                track.instrument in {"Open Hat", "Crash"},
                noise_seconds,
                track.instrument == "Crash",
                pitch,
            )
        elif track.instrument == "Perc":
            n = band_limited_noise(length, 500.0, 4200.0)
            n *= timed_envelope(length, max(0.050, noise_seconds * 0.55), 0.0009)
        else:
            n = highpass_like(noise(duration))
            n *= envelope(length, noise_seconds, 8.5)
        noise_gain = modulation_gain(track, "Noise Level", lfo1, lfo2, env_mod, 0.0, 2.0)
        hit += n * noise_level * noise_gain

    click_level = float(np.clip(track.click_level, 0.0, 1.5))
    if click_level > 0.0:
        click_len = min(length, int((0.010 if track.instrument == "Clap" else 0.018) * SAMPLE_RATE))
        if track.instrument == "Clap":
            click = band_limited_noise(click_len, 1200.0, 7600.0)
            click *= smooth_noise_envelope(click_len, 0.010, 0.0015)
        elif track.instrument == "Rim":
            click = band_limited_noise(click_len, 1600.0, 6200.0)
            click *= smooth_noise_envelope(click_len, 0.012, 0.0007)
        elif track.instrument == "Low Tom":
            click = band_limited_noise(click_len, 110.0, 3600.0)
            click *= timed_envelope(click_len, 0.010, 0.00045)
        elif track.instrument == "Kick":
            click = band_limited_noise(click_len, 90.0, 3600.0)
            click *= timed_envelope(click_len, 0.008, 0.00035)
        elif track.instrument == "Snare":
            click = band_limited_noise(click_len, 850.0, 7600.0)
            click *= timed_envelope(click_len, 0.011, 0.00035)
        elif track.instrument in {"Closed Hat", "Open Hat", "Crash"}:
            click = band_limited_noise(click_len, 5200.0 if track.instrument == "Crash" else 6500.0, 14000.0)
            click *= timed_envelope(click_len, 0.018 if track.instrument == "Crash" else 0.009, 0.0004)
        else:
            click = highpass_like(noise(click_len / SAMPLE_RATE))
            click *= envelope(click_len, 0.018, 16.0)
        click_gain = click_level * (
            0.32
            if track.instrument == "Clap"
            else 0.7
            if track.instrument == "Low Tom"
            else 0.52
            if track.instrument == "Kick"
            else 0.62
            if track.instrument == "Snare"
            else 0.7
            if track.instrument in {"Closed Hat", "Open Hat", "Crash"}
            else 1.0
        )
        hit[:click_len] += click * click_gain

    if uses_step_notes and track.instrument != "Kick":
        note_hz = getattr(track, "step_note_hz", None)
        if note_hz is not None:
            note_layer = step_note_resonance(
                length,
                note_hz,
                track.instrument,
                tone_seconds,
                noise_seconds,
            )
            note_amount = 0.18 if track.instrument in {"Closed Hat", "Open Hat", "Crash"} else 0.24
            hit += note_layer * note_amount

    hit = apply_transient(hit, track)
    hit = apply_filter(hit, track, lfo1, lfo2, env_mod)
    hit = apply_bitcrush(hit, track)
    hit = apply_saturation(hit, track, lfo1, lfo2, env_mod)

    if track.instrument == "Kick" and kick_mute > 0.0:
        damping_env = timed_envelope(length, 0.11 + 0.22 * (1.0 - kick_mute), 0.0008)
        hit *= (1.0 - kick_mute * 0.35) + kick_mute * 0.35 * damping_env

    attack_len = min(length, max(1, int(track.attack_ms * SAMPLE_RATE / 1000.0)))
    hit[:attack_len] *= np.linspace(0.0, 1.0, attack_len, dtype=np.float32)
    fade_seconds = max(0.012, min(0.180, noise_seconds * 0.14)) if track.instrument == "Crash" else 96 / SAMPLE_RATE
    fade = min(int(fade_seconds * SAMPLE_RATE), len(hit))
    if fade > 1:
        hit[-fade:] *= np.linspace(1.0, 0.0, fade, dtype=np.float32)

    mono = hit * volume * modulation_gain(track, "Volume", lfo1, lfo2, env_mod, 0.0, 2.0)
    pan = float(np.clip(track.pan, -1.0, 1.0))
    pan_mod = modulation_signal(track, "Pan", lfo1, lfo2, env_mod)
    if pan_mod is not None:
        pan_values = np.clip(pan + pan_mod, -1.0, 1.0)
        angle = (pan_values + 1.0) * np.pi / 4.0
        left = mono * np.cos(angle)
        right = mono * np.sin(angle)
    else:
        angle = (pan + 1.0) * np.pi / 4.0
        left = mono * np.cos(angle)
        right = mono * np.sin(angle)
    stereo = np.column_stack((left, right)).astype(np.float32)
    stereo = apply_patch_delay(stereo, track, lfo1, lfo2, env_mod)
    stereo = apply_patch_reverb(stereo, track, lfo1, lfo2, env_mod)
    return np.tanh(stereo).astype(np.float32)


def split_lfo_phase_override(phase_override):
    if isinstance(phase_override, tuple):
        first = phase_override[0] if len(phase_override) > 0 else None
        second = phase_override[1] if len(phase_override) > 1 else None
        return first, second
    return phase_override, None


def make_lfo(
    track: TrackState, length: int, phase_override: float | None = None, prefix: str = "lfo"
) -> np.ndarray | None:
    if not getattr(track, f"{prefix}_enabled") or getattr(track, f"{prefix}_amount") <= 0.0:
        return None

    rate = float(np.clip(getattr(track, f"{prefix}_rate"), 0.05, 80.0))
    start_phase = getattr(track, f"{prefix}_phase") if phase_override is None else phase_override
    phase = (np.arange(length, dtype=np.float32) / SAMPLE_RATE) * rate + start_phase
    phase = phase % 1.0

    shape = getattr(track, f"{prefix}_shape")
    if shape == "Triangle":
        lfo = 4.0 * np.abs(phase - 0.5) - 1.0
    elif shape == "Square":
        lfo = np.where(phase < 0.5, 1.0, -1.0)
    elif shape == "Random":
        steps = max(1, int(SAMPLE_RATE / rate))
        values = np.random.uniform(-1.0, 1.0, max(2, int(np.ceil(length / steps)) + 1))
        lfo = np.repeat(values, steps)[:length]
    else:
        lfo = np.sin(2.0 * np.pi * phase)
    return lfo.astype(np.float32)


def make_env_mod(track: TrackState, length: int) -> np.ndarray | None:
    if not track.env_mod_enabled or track.env_mod_amount <= 0.0:
        return None
    points = sorted(track.env_mod_points, key=lambda item: item[0])
    x = np.array([point[0] for point in points], dtype=np.float32)
    y = np.array([point[1] for point in points], dtype=np.float32)
    position = np.linspace(0.0, 1.0, length, endpoint=False, dtype=np.float32)
    return np.interp(position, x, y).astype(np.float32)
