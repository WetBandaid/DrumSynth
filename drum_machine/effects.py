import numpy as np

from .config import SAMPLE_RATE
from .model import TrackState


class BusCompressor:
    def __init__(self):
        self.envelope = 0.0

    def process(self, block: np.ndarray, amount: float) -> np.ndarray:
        amount = float(np.clip(amount, 0.0, 1.0))
        if amount <= 0.0:
            return np.tanh(block * 0.98)

        threshold = 0.88 - amount * 0.45
        ratio = 1.5 + amount * 6.0
        out = block.copy()
        for index, sample in enumerate(out):
            level = float(np.max(np.abs(sample)))
            self.envelope = max(level, self.envelope * 0.995)
            if self.envelope > threshold:
                over = self.envelope - threshold
                gain = threshold + over / ratio
                out[index] *= gain / max(self.envelope, 0.001)

        makeup = 1.0 + amount * 0.25
        return np.tanh(out * makeup).astype(np.float32)


def modulation_signal(
    track: TrackState,
    destination: str,
    lfo1: np.ndarray | None,
    lfo2: np.ndarray | None,
    env_mod: np.ndarray | None,
) -> np.ndarray | None:
    signal = None
    if track.lfo_destination == destination and lfo1 is not None:
        signal = lfo1 * float(np.clip(track.lfo_amount, 0.0, 1.0))
    if track.lfo2_destination == destination and lfo2 is not None:
        contribution = lfo2 * float(np.clip(track.lfo2_amount, 0.0, 1.0))
        signal = contribution if signal is None else signal + contribution
    if track.env_mod_destination == destination and env_mod is not None:
        contribution = env_mod * float(np.clip(track.env_mod_amount, -1.0, 1.0))
        signal = contribution if signal is None else signal + contribution
    if signal is None:
        return None
    return np.clip(signal, -1.5, 1.5).astype(np.float32)


def modulation_gain(
    track: TrackState,
    destination: str,
    lfo1: np.ndarray | None,
    lfo2: np.ndarray | None,
    env_mod: np.ndarray | None,
    minimum: float,
    maximum: float,
) -> float | np.ndarray:
    signal = modulation_signal(track, destination, lfo1, lfo2, env_mod)
    if signal is None:
        return 1.0
    return np.clip(1.0 + signal, minimum, maximum).astype(np.float32)


def apply_transient(hit: np.ndarray, track: TrackState) -> np.ndarray:
    attack = float(np.clip(track.transient_attack, -1.0, 1.0))
    body = float(np.clip(track.transient_body, 0.0, 2.0))
    split = min(len(hit), int(0.025 * SAMPLE_RATE))
    out = hit.copy()
    if split > 0:
        out[:split] *= 1.0 + attack
        out[split:] *= body
    return out


def apply_filter(
    hit: np.ndarray,
    track: TrackState,
    lfo1: np.ndarray | None = None,
    lfo2: np.ndarray | None = None,
    env_mod: np.ndarray | None = None,
) -> np.ndarray:
    filter_type = track.filter_type
    filter_mod = modulation_signal(track, "Filter Cutoff", lfo1, lfo2, env_mod)
    filter_lfo_active = filter_mod is not None
    if filter_type == "Off" and not filter_lfo_active:
        return hit
    if filter_type == "Off" and filter_lfo_active:
        filter_type = "Low-pass"
        cutoff = min(float(track.filter_cutoff), 6000.0)
    else:
        cutoff = float(track.filter_cutoff)
    cutoff = float(np.clip(cutoff, 20.0, SAMPLE_RATE * 0.45))
    resonance = float(np.clip(track.filter_resonance, 0.0, 0.95))
    if filter_lfo_active:
        cutoff = np.clip(cutoff * (2.0 ** (filter_mod * 4.0)), 20.0, SAMPLE_RATE * 0.45)
        if filter_type == "Low-pass":
            return one_pole_lowpass_variable(hit, cutoff, resonance)
        if filter_type == "High-pass":
            return hit - one_pole_lowpass_variable(hit, cutoff, resonance)
        if filter_type == "Band-pass":
            low_cut = np.clip(cutoff * 0.55, 20.0, SAMPLE_RATE * 0.45)
            high_cut = np.clip(cutoff * 1.8, 20.0, SAMPLE_RATE * 0.45)
            return one_pole_lowpass_variable(
                hit - one_pole_lowpass_variable(hit, low_cut, resonance), high_cut, resonance
            )
    if filter_type == "Low-pass":
        return one_pole_lowpass(hit, cutoff, resonance)
    if filter_type == "High-pass":
        return hit - one_pole_lowpass(hit, cutoff, resonance)
    if filter_type == "Band-pass":
        low_cut = max(20.0, cutoff * 0.55)
        high_cut = min(SAMPLE_RATE * 0.45, cutoff * 1.8)
        return one_pole_lowpass(hit - one_pole_lowpass(hit, low_cut, resonance), high_cut, resonance)
    return hit


def one_pole_lowpass(x: np.ndarray, cutoff: float, resonance: float) -> np.ndarray:
    alpha = np.exp(-2.0 * np.pi * cutoff / SAMPLE_RATE)
    out = np.empty_like(x)
    y = 0.0
    feedback = resonance * 0.12
    for index, sample in enumerate(x):
        y = (1.0 - alpha) * (sample - y * feedback) + alpha * y
        out[index] = y
    return out


def one_pole_lowpass_variable(
    x: np.ndarray, cutoff: np.ndarray, resonance: float
) -> np.ndarray:
    alpha = np.exp(-2.0 * np.pi * cutoff / SAMPLE_RATE)
    out = np.empty_like(x)
    y = 0.0
    feedback = resonance * 0.12
    for index, sample in enumerate(x):
        y = (1.0 - alpha[index]) * (sample - y * feedback) + alpha[index] * y
        out[index] = y
    return out


def apply_bitcrush(hit: np.ndarray, track: TrackState) -> np.ndarray:
    bits = int(np.clip(track.bit_depth, 4, 16))
    reduction = int(np.clip(track.sample_rate_reduction, 1, 32))
    out = hit
    if bits < 16:
        levels = float(2 ** bits)
        out = np.round(out * levels) / levels
    if reduction > 1 and len(out) > reduction:
        out = out.copy()
        for start in range(0, len(out), reduction):
            out[start : start + reduction] = out[start]
    return out.astype(np.float32, copy=False)


def apply_saturation(
    hit: np.ndarray,
    track: TrackState,
    lfo1: np.ndarray | None = None,
    lfo2: np.ndarray | None = None,
    env_mod: np.ndarray | None = None,
) -> np.ndarray:
    drive = float(np.clip(track.drive, 0.0, 1.0))
    drive_mod = modulation_signal(track, "Drive", lfo1, lfo2, env_mod)
    if drive_mod is not None:
        drive = np.clip(drive + drive_mod, 0.0, 1.0)
    driven = hit * (1.0 + drive * 6.0)
    if track.saturation_mode == "Hard":
        return np.clip(driven, -0.9, 0.9).astype(np.float32)
    if track.saturation_mode == "Fold":
        folded = ((driven + 1.0) % 4.0) - 2.0
        folded = 1.0 - np.abs(folded)
        return folded.astype(np.float32)
    if track.saturation_mode == "Sine":
        return np.sin(np.clip(driven, -np.pi, np.pi)).astype(np.float32)
    return np.tanh(driven).astype(np.float32)


def apply_patch_delay(
    stereo: np.ndarray,
    track: TrackState,
    lfo1: np.ndarray | None = None,
    lfo2: np.ndarray | None = None,
    env_mod: np.ndarray | None = None,
) -> np.ndarray:
    mix = float(np.clip(track.delay_send, 0.0, 1.0))
    send_mod = modulation_signal(track, "Delay Send", lfo1, lfo2, env_mod)
    if send_mod is not None:
        mix = float(np.clip(mix + np.mean(send_mod) * 0.85, 0.0, 1.0))
    if mix <= 0.0:
        return stereo

    delay_samples = int(np.clip(track.delay_time, 0.03, 1.5) * SAMPLE_RATE)
    feedback = float(np.clip(track.delay_feedback, 0.0, 0.88))
    tone = float(np.clip(track.delay_tone, 0.0, 1.0))
    width = float(np.clip(track.delay_width, 0.0, 1.0))
    repeats = 1
    level = feedback
    while repeats < 8 and level * mix > 0.025:
        repeats += 1
        level *= feedback

    tail = delay_samples * repeats
    out = np.zeros((len(stereo) + tail, 2), dtype=np.float32)
    out[: len(stereo)] += stereo
    echo = stereo.copy() * mix * (0.35 + tone * 0.65)
    for repeat in range(1, repeats + 1):
        start = delay_samples * repeat
        end = start + len(stereo)
        widened = apply_width(echo, width if repeat % 2 else min(1.0, width + 0.2))
        if repeat % 2 == 0:
            widened = widened[:, ::-1]
        out[start:end] += widened
        echo *= feedback
    return out


def apply_patch_reverb(
    stereo: np.ndarray,
    track: TrackState,
    lfo1: np.ndarray | None = None,
    lfo2: np.ndarray | None = None,
    env_mod: np.ndarray | None = None,
) -> np.ndarray:
    mix = float(np.clip(track.reverb_send, 0.0, 1.0))
    send_mod = modulation_signal(track, "Reverb Send", lfo1, lfo2, env_mod)
    if send_mod is not None:
        mix = float(np.clip(mix + np.mean(send_mod) * 0.85, 0.0, 1.0))
    if mix <= 0.0:
        return stereo

    size = float(np.clip(track.reverb_size, 0.1, 1.0))
    decay = float(np.clip(track.reverb_decay, 0.0, 0.92))
    tone = float(np.clip(track.reverb_tone, 0.0, 1.0))
    delays = [0.029, 0.037, 0.053, 0.071]
    tail = int((0.18 + size * 1.4) * SAMPLE_RATE)
    out = np.zeros((len(stereo) + tail, 2), dtype=np.float32)
    out[: len(stereo)] += stereo
    source = stereo * mix * (0.35 + tone * 0.65)

    for index, delay in enumerate(delays):
        delay_samples = int(delay * (0.65 + size) * SAMPLE_RATE)
        repeats = max(2, int(3 + size * 5))
        echo = source * (0.45 / (index + 1))
        for repeat in range(1, repeats + 1):
            start = delay_samples * repeat
            end = min(len(out), start + len(source))
            if start >= len(out):
                break
            chunk = echo[: end - start]
            if (index + repeat) % 2:
                chunk = chunk[:, ::-1]
            out[start:end] += chunk
            echo *= decay
    return out


def apply_width(stereo: np.ndarray, width: float) -> np.ndarray:
    mid = (stereo[:, 0] + stereo[:, 1]) * 0.5
    side = (stereo[:, 0] - stereo[:, 1]) * 0.5 * width
    return np.column_stack((mid + side, mid - side)).astype(np.float32)
