import copy

from dataclasses import dataclass, field

from .config import (
    DEFAULT_PATTERN,
    DRUM_PRESETS,
    FILTER_TYPES,
    INSTRUMENTS,
    LFO_DESTINATIONS,
    LFO_SHAPES,
    SATURATION_MODES,
    STEPS,
)


TRACK_SOUND_EXCLUDED_KEYS = {
    "name",
    "pattern",
    "muted",
    "solo",
    "track_steps",
    "velocities",
    "probabilities",
    "ratchets",
    "bass_notes",
}


@dataclass
class TrackState:
    name: str
    instrument: str
    pattern: list[bool] = field(default_factory=lambda: [False] * STEPS)

    volume: float = 0.85
    decay: float = 0.5
    pitch: float = 1.0
    pan: float = 0.0
    muted: bool = False
    solo: bool = False
    track_steps: int = STEPS

    tone_start: float = 150.0
    tone_end: float = 42.0
    tone_level: float = 1.0
    tone_decay: float = 0.5
    pitch_env_amount: float = 0.0
    pitch_env_decay: float = 0.08

    noise_level: float = 0.1
    noise_decay: float = 0.2
    click_level: float = 0.1
    attack_ms: float = 1.0
    kick_mute: float = 0.0
    transient_attack: float = 0.0
    transient_body: float = 1.0

    drive: float = 0.2
    saturation_mode: str = "Soft"
    filter_type: str = "Off"
    filter_cutoff: float = 12_000.0
    filter_resonance: float = 0.0
    bit_depth: int = 16
    sample_rate_reduction: int = 1

    delay_send: float = 0.0
    delay_time: float = 0.25
    delay_feedback: float = 0.25
    delay_tone: float = 0.7
    delay_width: float = 0.75
    reverb_send: float = 0.0
    reverb_size: float = 0.55
    reverb_decay: float = 0.45
    reverb_tone: float = 0.65

    lfo_enabled: bool = False
    lfo_shape: str = "Sine"
    lfo_destination: str = "Filter Cutoff"
    lfo_rate: float = 5.0
    lfo_amount: float = 0.0
    lfo_phase: float = 0.25
    lfo2_enabled: bool = False
    lfo2_shape: str = "Triangle"
    lfo2_destination: str = "Pan"
    lfo2_rate: float = 2.0
    lfo2_amount: float = 0.0
    lfo2_phase: float = 0.0
    env_mod_enabled: bool = False
    env_mod_destination: str = "Filter Cutoff"
    env_mod_amount: float = 0.0
    env_mod_points: list[list[float]] = field(
        default_factory=lambda: [[0.0, 0.0], [0.08, 1.0], [0.35, 0.45], [1.0, 0.0]]
    )

    velocities: list[float] = field(default_factory=lambda: [1.0] * STEPS)
    probabilities: list[float] = field(default_factory=lambda: [1.0] * STEPS)
    ratchets: list[int] = field(default_factory=lambda: [1] * STEPS)
    bass_enabled: bool = False
    bass_notes: list[int] = field(default_factory=lambda: [36] * STEPS)

    def apply_preset(self, instrument: str):
        defaults = TrackState(name=self.name, instrument=instrument)
        for key, value in defaults.sound_preset_dict().items():
            setattr(self, key, copy.deepcopy(value))

        self.instrument = instrument
        for key, value in DRUM_PRESETS[instrument].items():
            setattr(self, key, copy.deepcopy(value))

    def to_dict(self) -> dict:
        return {
            key: getattr(self, key)
            for key in (
                "name",
                "instrument",
                "pattern",
                "volume",
                "decay",
                "pitch",
                "pan",
                "muted",
                "solo",
                "track_steps",
                "tone_start",
                "tone_end",
                "tone_level",
                "tone_decay",
                "pitch_env_amount",
                "pitch_env_decay",
                "noise_level",
                "noise_decay",
                "click_level",
                "attack_ms",
                "kick_mute",
                "transient_attack",
                "transient_body",
                "drive",
                "saturation_mode",
                "filter_type",
                "filter_cutoff",
                "filter_resonance",
                "bit_depth",
                "sample_rate_reduction",
                "delay_send",
                "delay_time",
                "delay_feedback",
                "delay_tone",
                "delay_width",
                "reverb_send",
                "reverb_size",
                "reverb_decay",
                "reverb_tone",
                "lfo_enabled",
                "lfo_shape",
                "lfo_destination",
                "lfo_rate",
                "lfo_amount",
                "lfo_phase",
                "lfo2_enabled",
                "lfo2_shape",
                "lfo2_destination",
                "lfo2_rate",
                "lfo2_amount",
                "lfo2_phase",
                "env_mod_enabled",
                "env_mod_destination",
                "env_mod_amount",
                "env_mod_points",
                "velocities",
                "probabilities",
                "ratchets",
                "bass_enabled",
                "bass_notes",
            )
        }

    def sound_preset_dict(self) -> dict:
        return {
            key: value
            for key, value in self.to_dict().items()
            if key not in TRACK_SOUND_EXCLUDED_KEYS
        }

    def update_sound_preset(self, data: dict):
        self.update_from_dict(
            {
                key: value
                for key, value in data.items()
                if key not in TRACK_SOUND_EXCLUDED_KEYS
            }
        )

    def update_from_dict(self, data: dict):
        for key, value in data.items():
            if not hasattr(self, key):
                continue
            if key == "pattern":
                value = _fit_list([bool(step) for step in value], False)
            elif key in {"velocities", "probabilities"}:
                value = _fit_list([float(step) for step in value], 1.0)
            elif key == "ratchets":
                value = _fit_list([int(step) for step in value], 1)
            elif key == "bass_notes":
                value = _fit_list([max(12, min(84, int(step))) for step in value], 36)
            elif key == "track_steps":
                value = max(1, min(STEPS, int(value)))
            elif key == "saturation_mode" and value not in SATURATION_MODES:
                value = "Soft"
            elif key == "filter_type" and value not in FILTER_TYPES:
                value = "Off"
            elif key == "lfo_shape" and value not in LFO_SHAPES:
                value = "Sine"
            elif key == "lfo_destination" and value not in LFO_DESTINATIONS:
                value = "Filter Cutoff"
            elif key == "lfo2_shape" and value not in LFO_SHAPES:
                value = "Triangle"
            elif key == "lfo2_destination" and value not in LFO_DESTINATIONS:
                value = "Pan"
            elif key == "env_mod_destination" and value not in LFO_DESTINATIONS:
                value = "Filter Cutoff"
            elif key == "env_mod_points":
                value = _fit_envelope_points(value)
            setattr(self, key, value)


def _fit_list(values: list, fill_value):
    values = values[:STEPS]
    return values + [fill_value] * (STEPS - len(values))


def _fit_envelope_points(values):
    points = []
    for point in values[:4]:
        if not isinstance(point, (list, tuple)) or len(point) != 2:
            continue
        x = max(0.0, min(1.0, float(point[0])))
        y = max(0.0, min(1.0, float(point[1])))
        points.append([x, y])
    if len(points) != 4:
        return [[0.0, 0.0], [0.08, 1.0], [0.35, 0.45], [1.0, 0.0]]
    points.sort(key=lambda item: item[0])
    points[0][0] = 0.0
    points[-1][0] = 1.0
    return points


def create_default_tracks() -> list[TrackState]:
    tracks = []
    for index, name in enumerate(INSTRUMENTS):
        track = TrackState(
            name=name,
            instrument=name,
            pattern=[step == "1" for step in DEFAULT_PATTERN[index]],
        )
        track.apply_preset(name)
        tracks.append(track)
    return tracks
