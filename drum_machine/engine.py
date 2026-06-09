import copy
import hashlib
import json
import threading

import numpy as np
import sounddevice as sd

from .config import CHANNELS, GENERATOR_STYLES, PATTERN_SCENES, SAMPLE_RATE, STEPS
from .effects import BusCompressor, one_pole_lowpass
from .model import create_default_tracks
from .synth import make_hit


class SampleVoice:
    def __init__(self, samples: np.ndarray, gain: float = 1.0):
        self.samples = samples.astype(np.float32, copy=False)
        self.gain = float(np.clip(gain, 0.0, 1.6))
        self.position = 0

    def render_into(
        self,
        dry: np.ndarray,
        start: int,
        count: int,
    ) -> bool:
        available = len(self.samples) - self.position
        if available <= 0:
            return False

        render_count = min(count, available)
        chunk = self.samples[self.position : self.position + render_count] * self.gain
        target = slice(start, start + render_count)
        dry[target] += chunk
        self.position += render_count
        return self.position < len(self.samples)


class DrumEngine:
    def __init__(self):
        self.lock = threading.RLock()
        self.tracks = create_default_tracks()
        self.current_scene = 0
        self.pattern_scenes = self._make_initial_scenes()
        self.scene_names = [f"Scene {index + 1}" for index in range(PATTERN_SCENES)]
        self.track_clipboard: dict | None = None
        self.scene_clipboard: dict | None = None
        self.song_chain = [{"scene": 0, "bars": 1}]
        self.song_loop = True
        self.song_playing = False
        self.song_position = 0
        self.song_bar_progress = 0
        self.bpm = 120.0
        self.swing = 0.0
        self.master_volume = 0.75
        self.compressor_amount = 0.16
        self.global_filter_cutoff = 18_000.0
        self.global_filter_resonance = 0.0
        self.global_drive = 0.0
        self.global_fx_amount = 1.0
        self.global_density = 1.0
        self.global_humanize = 0.0
        self.fill_enabled = False
        self.playing = False
        self.current_step = 0
        self.samples_until_step = 0
        self.transport_sample_position = 0
        self.voices: list[SampleVoice] = []
        self.scheduled: list[tuple[int, SampleVoice]] = []
        self.render_cache: dict[tuple[int, str, float], np.ndarray] = {}
        self.fallback_hits: dict[int, np.ndarray] = {}
        self.cache_request_version = 0
        self.cache_render_thread: threading.Thread | None = None
        self.render_random_lock = threading.Lock()
        self.stream: sd.OutputStream | None = None
        self.compressor = BusCompressor()
        self.spectrum_bands = np.zeros(24, dtype=np.float32)
        self.spectrum_channel_db = np.full((CHANNELS, 56), -72.0, dtype=np.float32)
        self.spectrum_buffer = np.zeros((4096, CHANNELS), dtype=np.float32)
        self.spectrum_write_index = 0
        self.startup_generation = {}
        self.generate_startup_session()

    def start_stream(self):
        if self.stream is None:
            self.stream = sd.OutputStream(
                samplerate=SAMPLE_RATE,
                channels=CHANNELS,
                dtype="float32",
                blocksize=256,
                callback=self._audio_callback,
            )
            self.stream.start()

    def close(self):
        if self.stream is not None:
            self.stream.stop()
            self.stream.close()
            self.stream = None

    def set_playing(self, playing: bool):
        with self.lock:
            self.playing = playing
            if playing:
                self.samples_until_step = 0
                self.transport_sample_position = 0
            else:
                self.song_playing = False
                self.voices.clear()
                self.scheduled.clear()

    def set_bpm(self, bpm: float):
        with self.lock:
            self.bpm = float(np.clip(bpm, 40.0, 240.0))

    def set_swing(self, swing: float):
        with self.lock:
            self.swing = float(np.clip(swing, 0.0, 0.45))

    def set_master_volume(self, volume: float):
        with self.lock:
            self.master_volume = float(np.clip(volume, 0.0, 1.0))

    def spectrum_levels(self) -> np.ndarray:
        with self.lock:
            write_index = self.spectrum_write_index
            buffer = self.spectrum_buffer.copy()
            previous = self.spectrum_bands.copy()
        ordered = np.concatenate((buffer[write_index:], buffer[:write_index]), axis=0)
        bands = output_spectrum_bands(ordered, len(previous))
        with self.lock:
            self.spectrum_bands = np.maximum(bands, previous * 0.82)
            return self.spectrum_bands.copy()

    def spectrum_channel_db_levels(self, band_count: int = 56) -> tuple[np.ndarray, np.ndarray]:
        with self.lock:
            write_index = self.spectrum_write_index
            buffer = self.spectrum_buffer.copy()
            previous = self.spectrum_channel_db.copy()
        ordered = np.concatenate((buffer[write_index:], buffer[:write_index]), axis=0)
        bands = np.zeros((CHANNELS, band_count), dtype=np.float32)
        totals = np.zeros(CHANNELS, dtype=np.float32)
        for channel in range(CHANNELS):
            channel_audio = ordered[:, channel]
            bands[channel] = output_spectrum_db_bands(channel_audio, band_count)
            rms = float(np.sqrt(np.mean(channel_audio * channel_audio)))
            totals[channel] = 20.0 * np.log10(max(rms, 1.0e-6))
        with self.lock:
            if previous.shape == bands.shape:
                self.spectrum_channel_db = np.maximum(bands, previous - 4.0)
            else:
                self.spectrum_channel_db = bands
            return self.spectrum_channel_db.copy(), totals

    def set_global_param(self, param: str, value):
        should_render = False
        with self.lock:
            setattr(self, param, value)
            if param == "global_fx_amount":
                self.render_cache.clear()
                should_render = self.playing
        if should_render:
            self.prepare_render_cache_async()

    def global_state(self) -> dict:
        return {
            "compressor_amount": self.compressor_amount,
            "global_filter_cutoff": self.global_filter_cutoff,
            "global_filter_resonance": self.global_filter_resonance,
            "global_drive": self.global_drive,
            "global_fx_amount": self.global_fx_amount,
            "global_density": self.global_density,
            "global_humanize": self.global_humanize,
            "fill_enabled": self.fill_enabled,
        }

    def update_global_state(self, data: dict):
        should_render = False
        with self.lock:
            for key, value in data.items():
                if hasattr(self, key):
                    setattr(self, key, value)
            self.render_cache.clear()
            should_render = self.playing
        if should_render:
            self.prepare_render_cache_async()

    def set_step(self, track: int, step: int, enabled: bool):
        with self.lock:
            self.tracks[track].pattern[step] = enabled
            self._store_current_scene_locked()

    def set_scene_name(self, index: int, name: str):
        with self.lock:
            index = int(np.clip(index, 0, PATTERN_SCENES - 1))
            cleaned = " ".join(str(name).strip().split())
            self.scene_names[index] = cleaned[:24] or f"Scene {index + 1}"

    def scene_label(self, index: int) -> str:
        with self.lock:
            index = int(np.clip(index, 0, PATTERN_SCENES - 1))
            return self.scene_names[index]

    def scene_labels(self) -> list[str]:
        with self.lock:
            return list(self.scene_names)

    def set_track_param(self, track: int, param: str, value, render_async: bool = True):
        should_render = False
        with self.lock:
            setattr(self.tracks[track], param, value)
            if param == "bass_enabled" and bool(value) and not any(self.tracks[track].bass_note_enabled):
                self.tracks[track].bass_note_enabled = [bool(step) for step in self.tracks[track].pattern]
                self._store_current_scene_locked()
            if param not in {"muted", "solo", "track_steps"}:
                self._clear_track_cache(track)
                should_render = self.playing
            elif param == "track_steps":
                self._store_current_scene_locked()
        if should_render and render_async:
            self.prepare_render_cache_async()

    def set_step_param(self, track: int, step: int, param: str, value):
        should_render = False
        with self.lock:
            if param == "bass_notes" and value is None:
                return
            values = getattr(self.tracks[track], param)
            values[step] = value
            if param == "bass_notes":
                self.tracks[track].bass_note_enabled[step] = int(value) != 36
                self.tracks[track].bass_enabled = any(self.tracks[track].bass_note_enabled)
                self._clear_track_cache(track)
                should_render = self.playing
            self._store_current_scene_locked()
        if should_render:
            self.prepare_render_cache_async()

    def apply_track_preset(self, track: int, instrument: str):
        should_render = False
        with self.lock:
            self.tracks[track].apply_preset(instrument)
            self._clear_track_cache(track)
            should_render = self.playing
        if should_render:
            self.prepare_render_cache_async()

    def track_sound_preset_state(self, track: int) -> dict:
        with self.lock:
            return copy.deepcopy(self.tracks[track].sound_preset_dict())

    def load_track_sound_preset(self, track: int, data: dict):
        should_render = False
        with self.lock:
            self.tracks[track].update_sound_preset(data)
            self._clear_track_cache(track)
            should_render = self.playing
        if should_render:
            self.prepare_render_cache_async()

    def clear_pattern(self):
        with self.lock:
            for track in self.tracks:
                self._reset_track_pattern_locked(track)
            self.voices.clear()
            self.scheduled.clear()
            self.current_step = 0
            self.samples_until_step = 0
            self.render_cache.clear()
            self.fallback_hits.clear()
            self._store_current_scene_locked()

    def load_default_pattern(self):
        with self.lock:
            for index, track in enumerate(create_default_tracks()):
                self.tracks[index].pattern = track.pattern
                self.tracks[index].velocities = track.velocities
                self.tracks[index].probabilities = track.probabilities
                self.tracks[index].ratchets = track.ratchets
                self.tracks[index].bass_notes = track.bass_notes
                self.tracks[index].bass_note_enabled = track.bass_note_enabled
                self.tracks[index].track_steps = track.track_steps
            self._store_current_scene_locked()

    def randomize_pattern(self):
        probabilities = [0.36, 0.18, 0.68, 0.12, 0.14, 0.12, 0.2, 0.18]
        with self.lock:
            for track, probability in zip(self.tracks, probabilities):
                track.pattern = [bool(step) for step in (np.random.random(STEPS) < probability)]
                track.velocities = [float(value) for value in np.random.uniform(0.55, 1.0, STEPS)]
                track.probabilities = [1.0] * STEPS
                track.ratchets = [1] * STEPS
            self.render_cache.clear()
            self._store_current_scene_locked()

    def generate_session(
        self,
        pattern_count: int,
        bars_per_pattern: int,
        style: str,
        complexity: float,
        fills: float,
        variation: float,
    ):
        pattern_count = int(np.clip(pattern_count, 1, PATTERN_SCENES))
        bars_per_pattern = int(np.clip(bars_per_pattern, 1, 16))
        complexity = float(np.clip(complexity, 0.0, 1.0))
        fills = float(np.clip(fills, 0.0, 1.0))
        variation = float(np.clip(variation, 0.0, 1.0))

        rng = np.random.default_rng()
        scenes = []
        for scene_index in range(pattern_count):
            scenes.append(
                self._generate_scene_pattern(
                    rng,
                    scene_index,
                    pattern_count,
                    style,
                    complexity,
                    fills,
                    variation,
                )
            )

        empty_scene = {
            "tracks": [
                {
                    "pattern": [False] * STEPS,
                    "velocities": [1.0] * STEPS,
                    "probabilities": [1.0] * STEPS,
                    "ratchets": [1] * STEPS,
                    "bass_notes": [36] * STEPS,
                    "bass_note_enabled": [False] * STEPS,
                    "track_steps": int(track.track_steps),
                }
                for track in self.tracks
            ]
        }

        should_render = False
        with self.lock:
            self.pattern_scenes = scenes + [copy.deepcopy(empty_scene) for _ in range(PATTERN_SCENES - pattern_count)]
            self.scene_names = [
                f"{style} {index + 1}" if index < pattern_count else f"Scene {index + 1}"
                for index in range(PATTERN_SCENES)
            ]
            self.song_chain = [{"scene": index, "bars": bars_per_pattern} for index in range(pattern_count)]
            self.song_loop = True
            self.song_playing = False
            self.song_position = 0
            self.song_bar_progress = 0
            self.current_scene = 0
            self._load_scene_locked(self.pattern_scenes[0])
            self.current_step = 0
            self.samples_until_step = 0
            self.transport_sample_position = 0
            self.render_cache.clear()
            should_render = self.playing
        if should_render:
            self.prepare_render_cache_async()

    def generate_startup_session(self):
        rng = np.random.default_rng()
        style = str(rng.choice(GENERATOR_STYLES))
        pattern_count = min(PATTERN_SCENES, 4)
        bars_per_pattern = 4
        complexity = float(rng.uniform(0.38, 0.78))
        fills = float(rng.uniform(0.24, 0.68))
        variation = float(rng.uniform(0.20, 0.62))
        self.startup_generation = {
            "style": style,
            "pattern_count": pattern_count,
            "bars_per_pattern": bars_per_pattern,
            "complexity": complexity,
            "fills": fills,
            "variation": variation,
        }
        self.generate_session(
            pattern_count,
            bars_per_pattern,
            style,
            complexity,
            fills,
            variation,
        )

    def _generate_scene_pattern(
        self,
        rng,
        scene_index: int,
        pattern_count: int,
        style: str,
        complexity: float,
        fills: float,
        variation: float,
    ) -> dict:
        normalized_style = style.lower()
        tracks = [self._empty_generated_track(track.track_steps) for track in self.tracks]
        last_scene = scene_index == pattern_count - 1
        final_turn = (scene_index + 1) % 4 == 0 or last_scene
        energy = np.clip(0.42 + complexity * 0.44 + scene_index * 0.05, 0.0, 1.0)

        if normalized_style == "breakbeat":
            kick_steps = [0, 6, 10]
            snare_steps = [4, 12]
            hat_steps = [0, 2, 4, 6, 8, 10, 12, 14]
            open_steps = [7, 15]
        elif normalized_style == "rock":
            kick_steps = [0, 6, 8, 14] if complexity > 0.45 else [0, 8]
            snare_steps = [4, 12]
            hat_steps = [0, 2, 4, 6, 8, 10, 12, 14]
            open_steps = [14] if complexity > 0.35 else []
        elif normalized_style == "hip hop":
            kick_steps = [0, 6, 10]
            snare_steps = [4, 12]
            hat_steps = [0, 2, 4, 6, 8, 10, 12, 14]
            open_steps = [15] if complexity > 0.45 else []
        elif normalized_style == "boom bap":
            kick_steps = [0, 7, 10]
            snare_steps = [4, 12]
            hat_steps = [0, 2, 4, 6, 8, 10, 12, 14]
            open_steps = [6, 14] if complexity > 0.55 else []
        elif normalized_style == "uk garage":
            kick_steps = [0, 5, 10]
            snare_steps = [4, 11]
            hat_steps = [1, 3, 5, 7, 9, 11, 13, 15]
            open_steps = [6, 14]
        elif normalized_style == "jungle":
            kick_steps = [0, 3, 10, 13]
            snare_steps = [4, 7, 12]
            hat_steps = list(range(STEPS))
            open_steps = [7, 11, 15]
        elif normalized_style == "reggaeton":
            kick_steps = [0, 3, 8, 11]
            snare_steps = [4, 7, 12, 15]
            hat_steps = [0, 2, 4, 6, 8, 10, 12, 14]
            open_steps = [6, 14]
        elif normalized_style == "latin":
            kick_steps = [0, 6, 10]
            snare_steps = [4, 12]
            hat_steps = [0, 3, 6, 8, 11, 14]
            open_steps = [6, 14]
        elif normalized_style == "disco":
            kick_steps = [0, 4, 8, 12]
            snare_steps = [4, 12]
            hat_steps = [2, 6, 10, 14] if complexity < 0.55 else list(range(0, STEPS, 2))
            open_steps = [2, 6, 10, 14]
        elif normalized_style == "synthwave":
            kick_steps = [0, 8, 10]
            snare_steps = [4, 12]
            hat_steps = [0, 2, 4, 6, 8, 10, 12, 14]
            open_steps = [6, 14]
        elif normalized_style == "dark synth":
            kick_steps = [0, 4, 8, 10, 12]
            snare_steps = [4, 12] if complexity > 0.35 else [12]
            hat_steps = [0, 2, 4, 6, 8, 10, 12, 14]
            open_steps = [7, 15] if complexity > 0.45 else [15]
        elif normalized_style == "industrial":
            kick_steps = [0, 4, 8, 10, 12]
            snare_steps = [4, 12]
            hat_steps = list(range(0, STEPS, 1 if complexity > 0.60 else 2))
            open_steps = [3, 7, 11, 15]
        elif normalized_style == "idm":
            kick_steps = [0, 5, 9, 14]
            snare_steps = [3, 11]
            hat_steps = [0, 2, 5, 7, 8, 10, 13, 15]
            open_steps = [6, 15]
        elif normalized_style == "drum & bass":
            kick_steps = [0, 3, 10]
            snare_steps = [4, 12]
            hat_steps = list(range(0, STEPS, 1 if complexity > 0.45 else 2))
            open_steps = [7, 11, 15] if complexity > 0.35 else [15]
        elif normalized_style == "funk":
            kick_steps = [0, 3, 7, 10]
            snare_steps = [4, 12]
            hat_steps = list(range(STEPS))
            open_steps = [6, 14]
        elif normalized_style == "trap":
            kick_steps = [0, 7, 10, 15]
            snare_steps = [8]
            hat_steps = list(range(0, STEPS, 2))
            open_steps = [15]
        elif normalized_style == "electro":
            kick_steps = [0, 7, 10, 12]
            snare_steps = [4, 12]
            hat_steps = [0, 2, 4, 6, 8, 10, 12, 14]
            open_steps = [3, 11]
        elif normalized_style == "minimal":
            kick_steps = [0, 8] if complexity < 0.35 else [0, 4, 8, 12]
            snare_steps = [12] if complexity < 0.55 else [4, 12]
            hat_steps = [2, 6, 10, 14]
            open_steps = [14] if complexity > 0.25 else []
        elif normalized_style == "dub":
            kick_steps = [0, 6, 11]
            snare_steps = [4, 12]
            hat_steps = [0, 3, 6, 9, 12, 15]
            open_steps = [10, 15]
        elif normalized_style == "half-time":
            kick_steps = [0, 6, 11]
            snare_steps = [8]
            hat_steps = [0, 2, 4, 6, 8, 10, 12, 14]
            open_steps = [14]
        elif normalized_style == "techno":
            kick_steps = [0, 4, 8, 12]
            snare_steps = [4, 12] if complexity > 0.35 else [12]
            hat_steps = list(range(0, STEPS, 2 if complexity < 0.58 else 1))
            open_steps = [2, 6, 10, 14]
        else:
            kick_steps = [0, 4, 8, 12]
            snare_steps = [4, 12]
            hat_steps = [0, 2, 4, 6, 8, 10, 12, 14] if complexity < 0.62 else list(range(STEPS))
            open_steps = [2, 6, 10, 14]

        if normalized_style == "trap" and complexity > 0.45:
            hat_steps.extend([1, 5, 9, 13])
        elif normalized_style in {"jungle", "idm"} and complexity > 0.45:
            self._maybe_add_steps(rng, hat_steps, list(range(STEPS)), complexity * 0.55)
        elif normalized_style == "rock" and final_turn:
            self._maybe_add_steps(rng, kick_steps, [13, 15], fills * 0.35)

        if variation > 0.2:
            self._maybe_add_steps(rng, kick_steps, [3, 7, 11, 14, 15], 0.10 + variation * 0.28)
            self._maybe_add_steps(rng, snare_steps, [3, 7, 10, 15], max(0.0, variation - 0.25) * 0.25)
        if complexity > 0.65:
            self._maybe_add_steps(rng, hat_steps, list(range(STEPS)), (complexity - 0.65) * 0.9)

        self._write_steps(tracks[0], kick_steps, rng, 0.84, 1.0, variation)
        self._write_steps(tracks[1], snare_steps, rng, 0.78, 0.98, variation)
        self._write_steps(tracks[2], hat_steps, rng, 0.42, 0.82, variation)
        self._write_steps(tracks[3], open_steps, rng, 0.44, 0.78, variation)

        if scene_index == 0 or final_turn:
            self._write_steps(tracks[3], [0], rng, 0.74, 1.0, variation)
        if complexity > 0.25:
            clap_steps = (
                snare_steps
                if normalized_style in {"house", "techno", "electro", "disco"}
                else [step for step in snare_steps if step % 8 == 4]
            )
            self._write_steps(tracks[4], clap_steps, rng, 0.42, 0.72, variation)

        tom_steps = [14] if final_turn else []
        rim_steps = [3, 11] if complexity > 0.40 else []
        perc_steps = [5, 13] if complexity > 0.30 else []
        if normalized_style == "funk":
            rim_steps.extend([6, 14])
            perc_steps.extend([1, 9])
        elif normalized_style == "rock":
            tom_steps.extend([12, 14] if final_turn else [])
            rim_steps = []
            perc_steps = []
        elif normalized_style in {"hip hop", "boom bap"}:
            rim_steps.extend([4, 12] if complexity > 0.45 else [])
            perc_steps.extend([15] if final_turn else [])
        elif normalized_style == "uk garage":
            rim_steps.extend([2, 10])
            perc_steps.extend([6, 13])
        elif normalized_style == "jungle":
            perc_steps.extend([3, 6, 10, 14])
            tom_steps.extend([13, 15] if final_turn else [])
        elif normalized_style == "reggaeton":
            rim_steps.extend([4, 12])
            perc_steps.extend([3, 7, 11, 15])
        elif normalized_style == "latin":
            rim_steps.extend([3, 8, 13])
            perc_steps.extend([2, 6, 10, 14])
        elif normalized_style == "disco":
            perc_steps.extend([6, 14])
        elif normalized_style == "synthwave":
            tom_steps.extend([14] if final_turn else [])
            perc_steps.extend([5, 13] if complexity > 0.45 else [])
        elif normalized_style == "dark synth":
            rim_steps.extend([3, 11] if complexity > 0.40 else [11])
            perc_steps.extend([5, 9, 13] if complexity > 0.45 else [13])
            tom_steps.extend([14, 15] if final_turn and fills > 0.35 else [])
        elif normalized_style == "industrial":
            rim_steps.extend([4, 7, 12, 15])
            perc_steps.extend([1, 5, 9, 13])
        elif normalized_style == "idm":
            rim_steps.extend([2, 9, 15])
            perc_steps.extend([1, 4, 6, 12, 14])
        elif normalized_style == "drum & bass":
            perc_steps.extend([6, 14])
        elif normalized_style == "trap":
            perc_steps.extend([3, 11] if complexity > 0.35 else [])
            if final_turn and fills > 0.4:
                self._set_generated_step(tracks[2], 15, float(rng.uniform(0.42, 0.68)), 3)
        elif normalized_style == "dub":
            rim_steps.extend([4, 12])
            perc_steps.extend([7])
        elif normalized_style == "minimal":
            rim_steps = rim_steps[:1]
            perc_steps = perc_steps[:1]
        if variation > 0.45:
            self._maybe_add_steps(rng, rim_steps, [1, 6, 9, 14], variation * 0.20)
            self._maybe_add_steps(rng, perc_steps, [2, 7, 10, 15], variation * 0.25)
        self._write_steps(tracks[5], tom_steps, rng, 0.52, 0.85, variation)
        self._write_steps(tracks[6], rim_steps, rng, 0.38, 0.66, variation)
        self._write_steps(tracks[7], perc_steps, rng, 0.36, 0.70, variation)

        if fills > 0.0 and final_turn:
            fill_steps = [12, 13, 14, 15]
            fill_tracks = [2, 5, 6, 7]
            for local_index, step in enumerate(fill_steps):
                if rng.random() <= fills * (0.45 + local_index * 0.16):
                    track_index = int(rng.choice(fill_tracks))
                    self._set_generated_step(
                        tracks[track_index],
                        step,
                        float(rng.uniform(0.42, 0.84)),
                        2 if step in {14, 15} and fills > 0.55 else 1,
                    )
            if fills > 0.70:
                self._set_generated_step(tracks[1], 15, float(rng.uniform(0.32, 0.55)), 2)

        if energy < 0.55:
            self._thin_track(rng, tracks[2], 0.12 + (0.55 - energy) * 0.30)
            self._thin_track(rng, tracks[7], 0.30)

        return {"tracks": tracks}

    def _empty_generated_track(self, track_steps: int) -> dict:
        return {
            "pattern": [False] * STEPS,
            "velocities": [1.0] * STEPS,
            "probabilities": [1.0] * STEPS,
            "ratchets": [1] * STEPS,
            "bass_notes": [36] * STEPS,
            "bass_note_enabled": [False] * STEPS,
            "track_steps": int(np.clip(track_steps, 1, STEPS)),
        }

    def _write_steps(self, track_data: dict, steps, rng, low_velocity: float, high_velocity: float, variation: float):
        for step in sorted({int(step) % STEPS for step in steps}):
            velocity = float(rng.uniform(low_velocity, high_velocity))
            if step % 4 == 0:
                velocity = min(1.15, velocity + 0.06)
            velocity *= float(rng.uniform(1.0 - variation * 0.18, 1.0 + variation * 0.12))
            self._set_generated_step(track_data, step, velocity, 1)

    def _set_generated_step(self, track_data: dict, step: int, velocity: float, ratchet: int):
        step %= STEPS
        track_data["pattern"][step] = True
        track_data["velocities"][step] = float(np.clip(velocity, 0.15, 1.25))
        track_data["probabilities"][step] = 1.0
        track_data["ratchets"][step] = int(np.clip(ratchet, 1, 4))

    def _maybe_add_steps(self, rng, steps: list[int], candidates: list[int], probability: float):
        for step in candidates:
            if step not in steps and rng.random() < probability:
                steps.append(step)

    def _thin_track(self, rng, track_data: dict, chance: float):
        for step, enabled in enumerate(track_data["pattern"]):
            if enabled and rng.random() < chance:
                track_data["pattern"][step] = False

    def select_scene(self, index: int):
        with self.lock:
            self.song_playing = False
            self._store_current_scene_locked()
            self.current_scene = int(np.clip(index, 0, PATTERN_SCENES - 1))
            self._load_scene_locked(self.pattern_scenes[self.current_scene])
            self.current_step = 0
            self.samples_until_step = 0
            self.transport_sample_position = 0

    def store_scene(self, index: int | None = None):
        with self.lock:
            target = self.current_scene if index is None else int(np.clip(index, 0, PATTERN_SCENES - 1))
            self.pattern_scenes[target] = self._snapshot_pattern_locked()

    def copy_scene(self):
        with self.lock:
            self.scene_clipboard = copy.deepcopy(self._snapshot_pattern_locked())

    def paste_scene(self):
        with self.lock:
            if self.scene_clipboard is None:
                return False
            self._load_scene_locked(self.scene_clipboard)
            self._store_current_scene_locked()
            return True

    def clear_scene(self):
        with self.lock:
            for track in self.tracks:
                self._reset_track_pattern_locked(track)
            self.voices.clear()
            self.scheduled.clear()
            self.render_cache.clear()
            self.fallback_hits.clear()
            self._store_current_scene_locked()

    def clear_all_patterns(self):
        with self.lock:
            empty_scene = {
                "tracks": [
                    self._snapshot_track_pattern_locked(track)
                    for track in self.tracks
                ]
            }
            for track_data in empty_scene["tracks"]:
                track_data["pattern"] = [False] * STEPS
                track_data["velocities"] = [1.0] * STEPS
                track_data["probabilities"] = [1.0] * STEPS
                track_data["ratchets"] = [1] * STEPS
                track_data["bass_notes"] = [36] * STEPS
                track_data["bass_note_enabled"] = [False] * STEPS
            self.pattern_scenes = [copy.deepcopy(empty_scene) for _ in range(PATTERN_SCENES)]
            self._load_scene_locked(self.pattern_scenes[self.current_scene])
            self.voices.clear()
            self.scheduled.clear()
            self.current_step = 0
            self.samples_until_step = 0
            self.transport_sample_position = 0
            self.render_cache.clear()
            self.fallback_hits.clear()

    def copy_track_pattern(self, track_index: int):
        with self.lock:
            self.track_clipboard = self._snapshot_track_pattern_locked(self.tracks[track_index])

    def paste_track_pattern(self, track_index: int):
        with self.lock:
            if self.track_clipboard is None:
                return False
            self._load_track_pattern_locked(self.tracks[track_index], self.track_clipboard)
            self._store_current_scene_locked()
            return True

    def clear_track_pattern(self, track_index: int):
        with self.lock:
            track = self.tracks[track_index]
            self._reset_track_pattern_locked(track)
            self._store_current_scene_locked()

    def rotate_track_pattern(self, track_index: int, amount: int):
        with self.lock:
            track = self.tracks[track_index]
            for key in ("pattern", "velocities", "probabilities", "ratchets", "bass_notes", "bass_note_enabled"):
                values = list(getattr(track, key))
                shift = amount % STEPS
                setattr(track, key, values[-shift:] + values[:-shift] if shift else values)
            self._store_current_scene_locked()

    def morph_scenes(self, source_index: int, target_index: int, amount: float, destination_index: int):
        amount = float(np.clip(amount, 0.0, 1.0))
        with self.lock:
            self.song_playing = False
            self._store_current_scene_locked()
            source_index = int(np.clip(source_index, 0, PATTERN_SCENES - 1))
            target_index = int(np.clip(target_index, 0, PATTERN_SCENES - 1))
            destination_index = int(np.clip(destination_index, 0, PATTERN_SCENES - 1))
            source = copy.deepcopy(self.pattern_scenes[source_index])
            target = copy.deepcopy(self.pattern_scenes[target_index])
            morphed = self._morph_scene_locked(source, target, amount)
            self.pattern_scenes[destination_index] = morphed
            self.current_scene = destination_index
            self._load_scene_locked(morphed)
            self.current_step = 0
            self.samples_until_step = 0
            self.transport_sample_position = 0

    def pattern_bank_state(self) -> dict:
        with self.lock:
            self._store_current_scene_locked()
            return {
                "current_scene": self.current_scene,
                "scenes": copy.deepcopy(self.pattern_scenes),
                "scene_names": copy.deepcopy(self.scene_names),
            }

    def update_pattern_bank_state(self, data: dict):
        with self.lock:
            scenes = data.get("scenes")
            if isinstance(scenes, list) and scenes:
                self.pattern_scenes = self._fit_scenes(scenes)
                self.current_scene = int(
                    np.clip(data.get("current_scene", 0), 0, PATTERN_SCENES - 1)
                )
                self.scene_names = self._fit_scene_names(data.get("scene_names"))
                self._load_scene_locked(self.pattern_scenes[self.current_scene])
            else:
                self.pattern_scenes = self._make_initial_scenes()
                self.scene_names = [f"Scene {index + 1}" for index in range(PATTERN_SCENES)]
                self.current_scene = 0

    def reset_pattern_bank_from_current(self):
        with self.lock:
            self.current_scene = 0
            self.pattern_scenes = self._make_initial_scenes()
            self.scene_names = [f"Scene {index + 1}" for index in range(PATTERN_SCENES)]

    def song_state(self) -> dict:
        with self.lock:
            return {
                "chain": copy.deepcopy(self.song_chain),
                "loop": self.song_loop,
                "playing": self.song_playing,
                "position": self.song_position,
                "bar_progress": self.song_bar_progress,
            }

    def update_song_state(self, data: dict):
        with self.lock:
            chain = data.get("chain") if isinstance(data, dict) else None
            self.song_chain = self._fit_song_chain(chain)
            self.song_loop = bool(data.get("loop", True)) if isinstance(data, dict) else True
            self.song_playing = False
            self.song_position = 0
            self.song_bar_progress = 0

    def add_song_slot(self, scene: int | None = None, bars: int = 1):
        with self.lock:
            slot = {
                "scene": int(np.clip(self.current_scene if scene is None else scene, 0, PATTERN_SCENES - 1)),
                "bars": int(np.clip(bars, 1, 16)),
            }
            self.song_chain.append(slot)

    def remove_song_slot(self, index: int):
        with self.lock:
            if len(self.song_chain) <= 1:
                return
            if 0 <= index < len(self.song_chain):
                del self.song_chain[index]
                self.song_position = int(np.clip(self.song_position, 0, len(self.song_chain) - 1))
                self.song_bar_progress = 0

    def move_song_slot(self, index: int, amount: int):
        with self.lock:
            target = index + amount
            if not (0 <= index < len(self.song_chain) and 0 <= target < len(self.song_chain)):
                return
            self.song_chain[index], self.song_chain[target] = self.song_chain[target], self.song_chain[index]
            if self.song_position == index:
                self.song_position = target
            elif self.song_position == target:
                self.song_position = index

    def set_song_slot(self, index: int, scene: int | None = None, bars: int | None = None):
        with self.lock:
            if not 0 <= index < len(self.song_chain):
                return
            if scene is not None:
                self.song_chain[index]["scene"] = int(np.clip(scene, 0, PATTERN_SCENES - 1))
            if bars is not None:
                self.song_chain[index]["bars"] = int(np.clip(bars, 1, 16))

    def set_song_loop(self, loop: bool):
        with self.lock:
            self.song_loop = bool(loop)

    def start_song(self):
        with self.lock:
            if not self.song_chain:
                self.song_chain = [{"scene": self.current_scene, "bars": 1}]
            self.song_position = int(np.clip(self.song_position, 0, len(self.song_chain) - 1))
            self.song_bar_progress = 0
            self.song_playing = True
            self.playing = False
            self.current_step = 0
            self.samples_until_step = 0
            self.transport_sample_position = 0
            self._load_song_slot_locked(self.song_position)
            jobs = self._cache_jobs_locked()

        rendered = []
        for track_index, key, track, phase_key, global_fx_amount in jobs:
            rendered.append(
                (
                    track_index,
                    key,
                    self._render_track_copy(track, phase_key, global_fx_amount),
                )
            )

        with self.lock:
            for track_index, key, audio in rendered:
                self.render_cache[key] = audio
                self.fallback_hits[track_index] = audio
            self.playing = True

    def stop_song(self):
        with self.lock:
            self.song_playing = False

    def audition_track(self, track_index: int):
        render_job = None
        with self.lock:
            track_snapshot = copy.copy(self.tracks[track_index])
            global_fx_amount = float(self.global_fx_amount)
            phase = self._lfo_phases_for_track(track_snapshot, self.transport_sample_position)
            key, phase_key = self._cache_key_for_track(
                track_index,
                track_snapshot,
                phase,
                global_fx_amount,
            )
            cached = self.render_cache.get(key)
            if cached is not None:
                self.voices.append(SampleVoice(cached))
                return
            if (
                self.playing
                and track_index in self.fallback_hits
                and not self._has_lfo_phase_variation(track_snapshot)
            ):
                self.voices.append(SampleVoice(self.fallback_hits[track_index]))
                return
            render_job = (track_index, key, track_snapshot, phase_key, global_fx_amount)

        threading.Thread(
            target=self._render_audition_job,
            args=render_job,
            daemon=True,
        ).start()

    def audition_step(self, track_index: int, step: int):
        render_job = None
        with self.lock:
            source_track = self.tracks[track_index]
            step = int(np.clip(step, 0, STEPS - 1))
            track_snapshot = self._render_track_for_step(source_track, step)
            global_fx_amount = float(self.global_fx_amount)
            phase = self._lfo_phases_for_track(track_snapshot, self.transport_sample_position)
            key, phase_key = self._cache_key_for_track(
                track_index,
                track_snapshot,
                phase,
                global_fx_amount,
            )
            cached = self.render_cache.get(key)
            if cached is not None:
                self.voices.append(SampleVoice(cached))
                return
            if (
                self.playing
                and track_index in self.fallback_hits
                and not self._has_note_overrides(source_track)
                and not self._has_lfo_phase_variation(track_snapshot)
            ):
                self.voices.append(SampleVoice(self.fallback_hits[track_index]))
                return
            render_job = (track_index, key, track_snapshot, phase_key, global_fx_amount)

        threading.Thread(
            target=self._render_audition_job,
            args=render_job,
            daemon=True,
        ).start()

    def prepare_render_cache(self):
        with self.lock:
            for track_index, track in enumerate(self.tracks):
                for step, enabled in enumerate(track.pattern):
                    if enabled:
                        phase = self._lfo_phases_for_track(track, self.transport_sample_position)
                        self._render_step_hit_locked(track_index, step, phase)

    def prepare_render_cache_async(self):
        start_thread = False
        with self.lock:
            self.cache_request_version += 1
            if self.cache_render_thread is None or not self.cache_render_thread.is_alive():
                self.cache_render_thread = threading.Thread(
                    target=self._background_render_cache,
                    daemon=True,
                )
                start_thread = True
        if start_thread:
            self.cache_render_thread.start()

    def render_current_pattern(self, bars: int = 4, tail_seconds: float = 3.0) -> np.ndarray:
        bars = int(np.clip(bars, 1, 64))
        with self.lock:
            tracks = copy.deepcopy(self.tracks)
            state = self._offline_render_state_locked()
        return self._render_offline_sections([(tracks, bars)], state, tail_seconds)

    def render_song_arrangement(self, tail_seconds: float = 3.0) -> np.ndarray:
        with self.lock:
            self._store_current_scene_locked()
            base_tracks = copy.deepcopy(self.tracks)
            scenes = copy.deepcopy(self.pattern_scenes)
            chain = self._fit_song_chain(copy.deepcopy(self.song_chain))
            state = self._offline_render_state_locked()

        sections = []
        for slot in chain:
            tracks = copy.deepcopy(base_tracks)
            scene = scenes[int(np.clip(slot["scene"], 0, PATTERN_SCENES - 1))]
            for track, data in zip(tracks, scene.get("tracks", [])):
                self._load_track_pattern_locked(track, data)
            sections.append((tracks, int(np.clip(slot["bars"], 1, 16))))
        return self._render_offline_sections(sections, state, tail_seconds)

    def _offline_render_state_locked(self) -> dict:
        return {
            "bpm": float(self.bpm),
            "swing": float(self.swing),
            "master_volume": float(self.master_volume),
            "compressor_amount": float(self.compressor_amount),
            "global_filter_cutoff": float(self.global_filter_cutoff),
            "global_filter_resonance": float(self.global_filter_resonance),
            "global_drive": float(self.global_drive),
            "global_fx_amount": float(self.global_fx_amount),
            "global_density": float(self.global_density),
            "global_humanize": float(self.global_humanize),
            "fill_enabled": bool(self.fill_enabled),
        }

    def _render_offline_sections(
        self,
        sections: list[tuple[list, int]],
        state: dict,
        tail_seconds: float,
    ) -> np.ndarray:
        step_lengths = []
        for _tracks, bars in sections:
            for step in range(max(1, bars) * STEPS):
                step_lengths.append(self._step_samples_for_state(step, state))

        tail_samples = max(0, int(float(np.clip(tail_seconds, 0.0, 12.0)) * SAMPLE_RATE))
        total_samples = max(1, sum(step_lengths) + tail_samples)
        dry = np.zeros((total_samples, CHANNELS), dtype=np.float32)
        rng = np.random.default_rng(42_409)
        render_cache: dict[tuple, np.ndarray] = {}
        transport_position = 0

        for tracks, bars in sections:
            section_steps = max(1, bars) * STEPS
            solo_active = any(track.solo for track in tracks)
            pattern_has_activity = self._patterns_have_activity(tracks)
            for step in range(section_steps):
                step_samples = self._step_samples_for_state(step, state)
                for track_index, track in enumerate(tracks):
                    local_step = step % max(1, track.track_steps)
                    audible = track.solo if solo_active else not track.muted
                    probability = float(np.clip(track.probabilities[local_step], 0.0, 1.0))
                    probability *= float(np.clip(state["global_density"], 0.0, 1.5))
                    ratchets = int(np.clip(track.ratchets[local_step], 1, 4))
                    active = track.pattern[local_step]
                    fill_velocity_scale = 1.0
                    if pattern_has_activity:
                        active, probability, ratchets, fill_velocity_scale = self._offline_fill_step_settings(
                            state,
                            track_index,
                            local_step,
                            active,
                            probability,
                            ratchets,
                        )
                    if not (
                        audible
                        and active
                        and track.volume > 0.0
                        and rng.random() <= probability
                    ):
                        continue

                    spacing = max(1, step_samples // ratchets)
                    for repeat in range(ratchets):
                        event_position = transport_position + repeat * spacing
                        phase = self._lfo_phases_for_track(track, event_position)
                        velocity = track.velocities[local_step] * fill_velocity_scale
                        if repeat > 0:
                            velocity *= 0.72**repeat
                        if state["global_humanize"] > 0.0:
                            spread = float(np.clip(state["global_humanize"], 0.0, 1.0)) * 0.35
                            velocity *= float(rng.uniform(1.0 - spread, 1.0 + spread))
                        hit = self._render_offline_step_hit(
                            track_index,
                            track,
                            local_step,
                            phase,
                            state["global_fx_amount"],
                            render_cache,
                        )
                        start = event_position
                        end = min(total_samples, start + len(hit))
                        if end > start:
                            dry[start:end] += hit[: end - start] * velocity
                transport_position += step_samples

        return self._process_master_offline(dry, state)

    def _render_offline_step_hit(
        self,
        track_index: int,
        track,
        step: int,
        lfo_phase,
        global_fx_amount: float,
        render_cache: dict[tuple, np.ndarray],
    ) -> np.ndarray:
        render_track = self._render_track_for_step(track, step)
        key, phase_key = self._cache_key_for_track(
            track_index,
            render_track,
            lfo_phase,
            global_fx_amount,
        )
        if key not in render_cache:
            render_cache[key] = self._render_track_copy(render_track, phase_key, global_fx_amount)
        return render_cache[key]

    def _offline_fill_step_settings(
        self,
        state: dict,
        track_index: int,
        local_step: int,
        active: bool,
        probability: float,
        ratchets: int,
    ) -> tuple[bool, float, int, float]:
        if not state["fill_enabled"]:
            return active, probability, ratchets, 1.0

        step_in_bar = local_step % STEPS
        last_beat = step_in_bar in {12, 13, 14, 15}
        final_turn = step_in_bar in {14, 15}
        ghost_active = active
        velocity_scale = 1.0

        if track_index in {0, 1, 4}:
            return active, probability, ratchets, velocity_scale

        if active:
            probability = max(probability, 0.9 if last_beat else probability)
            if final_turn and track_index in {2, 6, 7}:
                ratchets = max(ratchets, 2)
            return ghost_active, probability, ratchets, velocity_scale

        fill_chance = 0.0
        if track_index == 2 and step_in_bar in {13, 15}:
            fill_chance = 0.68
            velocity_scale = 0.52
            ratchets = 2 if step_in_bar == 15 else 1
        elif track_index == 3 and step_in_bar == 15:
            fill_chance = 0.32
            velocity_scale = 0.42
        elif track_index in {6, 7} and final_turn:
            fill_chance = 0.42
            velocity_scale = 0.5
            ratchets = 2 if step_in_bar == 15 else 1
        elif track_index == 5 and step_in_bar == 15:
            fill_chance = 0.28
            velocity_scale = 0.48

        if fill_chance <= 0.0:
            return active, probability, ratchets, velocity_scale
        return True, fill_chance, ratchets, velocity_scale

    def _process_master_offline(self, dry: np.ndarray, state: dict) -> np.ndarray:
        mix = dry * state["master_volume"]
        mix = apply_global_filter(
            mix,
            state["global_filter_cutoff"],
            state["global_filter_resonance"],
        )
        drive = float(np.clip(state["global_drive"], 0.0, 1.0))
        if drive > 0.0:
            mix = np.tanh(mix * (1.0 + drive * 5.0))
        return BusCompressor().process(mix, state["compressor_amount"])

    def _step_samples_for_state(self, step: int, state: dict) -> int:
        beat_seconds = 60.0 / max(float(state["bpm"]), 1.0)
        base = beat_seconds / 4.0
        swing_amount = float(np.clip(state["swing"], 0.0, 0.45)) * 0.45
        multiplier = 1.0 + swing_amount if step % 2 == 0 else 1.0 - swing_amount
        return max(1, int(base * multiplier * SAMPLE_RATE))

    def _patterns_have_activity(self, tracks) -> bool:
        for track in tracks:
            length = max(1, int(np.clip(track.track_steps, 1, STEPS)))
            if any(track.pattern[:length]):
                return True
        return False

    def _make_initial_scenes(self) -> list[dict]:
        first = self._snapshot_pattern_locked()
        scenes = [first]
        for _ in range(PATTERN_SCENES - 1):
            empty_tracks = []
            for track in self.tracks:
                empty_tracks.append(
                    {
                        "pattern": [False] * STEPS,
                        "velocities": [1.0] * STEPS,
                        "probabilities": [1.0] * STEPS,
                        "ratchets": [1] * STEPS,
                        "bass_notes": [36] * STEPS,
                        "bass_note_enabled": [False] * STEPS,
                        "track_steps": track.track_steps,
                    }
                )
            scenes.append({"tracks": empty_tracks})
        return scenes

    def _fit_scenes(self, scenes: list[dict]) -> list[dict]:
        fitted = copy.deepcopy(scenes[:PATTERN_SCENES])
        while len(fitted) < PATTERN_SCENES:
            fitted.append({"tracks": [self._snapshot_track_pattern_locked(track) for track in self.tracks]})
        return fitted

    def _fit_scene_names(self, names) -> list[str]:
        fitted = []
        if isinstance(names, list):
            for index, name in enumerate(names[:PATTERN_SCENES]):
                cleaned = " ".join(str(name).strip().split())
                fitted.append(cleaned[:24] or f"Scene {index + 1}")
        while len(fitted) < PATTERN_SCENES:
            fitted.append(f"Scene {len(fitted) + 1}")
        return fitted

    def _fit_song_chain(self, chain) -> list[dict]:
        fitted = []
        if isinstance(chain, list):
            for slot in chain[:64]:
                if not isinstance(slot, dict):
                    continue
                fitted.append(
                    {
                        "scene": int(np.clip(slot.get("scene", 0), 0, PATTERN_SCENES - 1)),
                        "bars": int(np.clip(slot.get("bars", 1), 1, 16)),
                    }
                )
        return fitted or [{"scene": self.current_scene, "bars": 1}]

    def _store_current_scene_locked(self):
        self.pattern_scenes[self.current_scene] = self._snapshot_pattern_locked()

    def _snapshot_pattern_locked(self) -> dict:
        return {"tracks": [self._snapshot_track_pattern_locked(track) for track in self.tracks]}

    def _snapshot_track_pattern_locked(self, track) -> dict:
        return {
            "pattern": [bool(value) for value in track.pattern],
            "velocities": [float(value) for value in track.velocities],
            "probabilities": [float(value) for value in track.probabilities],
            "ratchets": [int(value) for value in track.ratchets],
            "bass_notes": [int(value) for value in track.bass_notes],
            "bass_note_enabled": [bool(value) for value in track.bass_note_enabled],
            "track_steps": int(track.track_steps),
        }

    def _reset_track_pattern_locked(self, track):
        track.pattern = [False] * STEPS
        track.velocities = [1.0] * STEPS
        track.probabilities = [1.0] * STEPS
        track.ratchets = [1] * STEPS
        track.bass_notes = [36] * STEPS
        track.bass_note_enabled = [False] * STEPS

    def _load_scene_locked(self, scene: dict):
        for track, data in zip(self.tracks, scene.get("tracks", [])):
            self._load_track_pattern_locked(track, data)

    def _load_song_slot_locked(self, index: int):
        if not self.song_chain:
            return
        self._store_current_scene_locked()
        self.song_position = int(np.clip(index, 0, len(self.song_chain) - 1))
        self.current_scene = self.song_chain[self.song_position]["scene"]
        self._load_scene_locked(self.pattern_scenes[self.current_scene])

    def _advance_song_bar_locked(self):
        if not self.song_playing or not self.song_chain:
            return
        current_slot = self.song_chain[self.song_position]
        self.song_bar_progress += 1
        if self.song_bar_progress < current_slot["bars"]:
            return

        next_position = self.song_position + 1
        if next_position >= len(self.song_chain):
            if not self.song_loop:
                self.song_playing = False
                self.playing = False
                self.song_bar_progress = 0
                return
            next_position = 0

        self.song_bar_progress = 0
        self._load_song_slot_locked(next_position)

    def _load_track_pattern_locked(self, track, data: dict):
        track.pattern = self._fit_pattern_list(data.get("pattern", track.pattern), False, bool)
        track.velocities = self._fit_pattern_list(data.get("velocities", track.velocities), 1.0, float)
        track.probabilities = self._fit_pattern_list(
            data.get("probabilities", track.probabilities), 1.0, float
        )
        track.ratchets = self._fit_pattern_list(data.get("ratchets", track.ratchets), 1, int)
        track.bass_notes = self._fit_pattern_list(data.get("bass_notes", track.bass_notes), 36, int)
        if "bass_note_enabled" in data:
            track.bass_note_enabled = self._fit_pattern_list(
                data.get("bass_note_enabled", []), False, bool
            )
        else:
            track.bass_note_enabled = [int(note) != 36 for note in track.bass_notes]
        track.track_steps = int(np.clip(data.get("track_steps", track.track_steps), 1, STEPS))

    def _morph_scene_locked(self, source: dict, target: dict, amount: float) -> dict:
        source_tracks = source.get("tracks", [])
        target_tracks = target.get("tracks", [])
        tracks = []
        for track_index, track in enumerate(self.tracks):
            empty = self._empty_generated_track(track.track_steps)
            source_data = source_tracks[track_index] if track_index < len(source_tracks) else empty
            target_data = target_tracks[track_index] if track_index < len(target_tracks) else empty
            tracks.append(self._morph_track_pattern(source_data, target_data, amount, track_index))
        return {"tracks": tracks}

    def _morph_track_pattern(
        self,
        source_data: dict,
        target_data: dict,
        amount: float,
        track_index: int,
    ) -> dict:
        source_pattern = self._fit_pattern_list(source_data.get("pattern", []), False, bool)
        target_pattern = self._fit_pattern_list(target_data.get("pattern", []), False, bool)
        source_velocities = self._fit_pattern_list(source_data.get("velocities", []), 1.0, float)
        target_velocities = self._fit_pattern_list(target_data.get("velocities", []), 1.0, float)
        source_probabilities = self._fit_pattern_list(source_data.get("probabilities", []), 1.0, float)
        target_probabilities = self._fit_pattern_list(target_data.get("probabilities", []), 1.0, float)
        source_ratchets = self._fit_pattern_list(source_data.get("ratchets", []), 1, int)
        target_ratchets = self._fit_pattern_list(target_data.get("ratchets", []), 1, int)
        source_bass_notes = self._fit_pattern_list(source_data.get("bass_notes", []), 36, int)
        target_bass_notes = self._fit_pattern_list(target_data.get("bass_notes", []), 36, int)
        source_bass_note_enabled = self._fit_pattern_list(
            source_data.get("bass_note_enabled", []), False, bool
        )
        target_bass_note_enabled = self._fit_pattern_list(
            target_data.get("bass_note_enabled", []), False, bool
        )
        source_steps = int(np.clip(source_data.get("track_steps", STEPS), 1, STEPS))
        target_steps = int(np.clip(target_data.get("track_steps", STEPS), 1, STEPS))

        pattern = []
        velocities = []
        probabilities = []
        ratchets = []
        bass_notes = []
        bass_note_enabled = []
        for step in range(STEPS):
            source_active = bool(source_pattern[step])
            target_active = bool(target_pattern[step])
            pattern.append(
                self._morphed_step_active(source_active, target_active, amount, track_index, step)
            )
            velocities.append(
                float(
                    np.clip(
                        self._blend(
                            source_velocities[step] if source_active else 0.35,
                            target_velocities[step] if target_active else 0.35,
                            amount,
                        ),
                        0.0,
                        1.5,
                    )
                )
            )
            probabilities.append(
                float(
                    np.clip(
                        self._blend(
                            source_probabilities[step] if source_active else 0.0,
                            target_probabilities[step] if target_active else 0.0,
                            amount,
                        ),
                        0.0,
                        1.0,
                    )
                )
            )
            ratchets.append(
                int(
                    np.clip(
                        round(self._blend(source_ratchets[step], target_ratchets[step], amount)),
                        1,
                        4,
                    )
                )
            )
            bass_notes.append(
                int(
                    np.clip(
                        round(self._blend(source_bass_notes[step], target_bass_notes[step], amount)),
                        12,
                        84,
                    )
                )
            )
            bass_note_enabled.append(
                bool(target_bass_note_enabled[step] if amount >= 0.5 else source_bass_note_enabled[step])
            )

        return {
            "pattern": pattern,
            "velocities": velocities,
            "probabilities": probabilities,
            "ratchets": ratchets,
            "bass_notes": bass_notes,
            "bass_note_enabled": bass_note_enabled,
            "track_steps": int(np.clip(round(self._blend(source_steps, target_steps, amount)), 1, STEPS)),
        }

    def _morphed_step_active(
        self,
        source_active: bool,
        target_active: bool,
        amount: float,
        track_index: int,
        step: int,
    ) -> bool:
        if amount <= 0.0:
            return source_active
        if amount >= 1.0:
            return target_active
        if source_active and target_active:
            return True
        if not source_active and not target_active:
            return False

        threshold = self._morph_threshold(track_index, step)
        if target_active:
            return amount >= threshold
        return amount <= threshold

    def _morph_threshold(self, track_index: int, step: int) -> float:
        value = ((track_index + 1) * 37 + (step + 1) * 17) % 100
        return 0.15 + (value / 99.0) * 0.70

    def _blend(self, source: float, target: float, amount: float) -> float:
        return float(source) * (1.0 - amount) + float(target) * amount

    def _fit_pattern_list(self, values, fill_value, caster):
        fitted = [caster(value) for value in list(values)[:STEPS]]
        return fitted + [fill_value] * (STEPS - len(fitted))

    def _clear_track_cache(self, track_index: int):
        stale_keys = [key for key in self.render_cache if key[0] == track_index]
        if stale_keys:
            self.fallback_hits[track_index] = self.render_cache[stale_keys[-1]]
        for key in stale_keys:
            del self.render_cache[key]

    def _cache_key_for_track(
        self,
        track_index: int,
        track,
        lfo_phase: tuple[float | None, float | None] | float | None = None,
        global_fx_amount: float | None = None,
    ):
        data = track.to_dict()
        for sequencer_key in (
            "pattern",
            "velocities",
            "probabilities",
            "ratchets",
            "bass_notes",
            "bass_note_enabled",
            "muted",
            "solo",
            "track_steps",
        ):
            data.pop(sequencer_key, None)
        fx_amount = self.global_fx_amount if global_fx_amount is None else global_fx_amount
        data["_global_fx_amount"] = round(float(fx_amount), 3)
        if hasattr(track, "step_note_hz"):
            data["_step_note_hz"] = round(float(track.step_note_hz), 4)
        phase1, phase2 = self._split_phase(lfo_phase)
        phase_key = (
            round(float(phase1 if phase1 is not None else track.lfo_phase) % 1.0, 4)
            if track.lfo_enabled and track.lfo_amount > 0.0
            else 0.0,
            round(float(phase2 if phase2 is not None else track.lfo2_phase) % 1.0, 4)
            if track.lfo2_enabled and track.lfo2_amount > 0.0
            else 0.0,
        )
        patch_key = json.dumps(data, sort_keys=True, default=float)
        return (track_index, patch_key, phase_key), phase_key

    def _render_track_copy(self, track, phase_key, global_fx_amount: float) -> np.ndarray:
        render_track = copy.copy(track)
        fx_amount = float(np.clip(global_fx_amount, 0.0, 1.5))
        render_track.delay_send = track.delay_send * fx_amount
        render_track.reverb_send = track.reverb_send * fx_amount
        seed = self._render_seed(render_track, phase_key, fx_amount)
        with self.render_random_lock:
            random_state = np.random.get_state()
            np.random.seed(seed)
            try:
                return make_hit(render_track, 1.0, phase_key)
            finally:
                np.random.set_state(random_state)

    def _render_seed(self, track, phase_key, global_fx_amount: float) -> int:
        data = track.sound_preset_dict()
        data["_phase_key"] = phase_key
        data["_global_fx_amount"] = round(float(global_fx_amount), 3)
        if hasattr(track, "step_note_hz"):
            data["_step_note_hz"] = round(float(track.step_note_hz), 4)
        payload = json.dumps(data, sort_keys=True, default=float)
        digest = hashlib.blake2s(payload.encode("utf-8"), digest_size=4).digest()
        return int.from_bytes(digest, "little")

    def _cache_jobs_locked(self):
        jobs = {}
        global_fx_amount = float(self.global_fx_amount)
        transport_position = self.transport_sample_position
        self._collect_cache_jobs_for_tracks_locked(
            jobs,
            self.tracks,
            transport_position,
            global_fx_amount,
        )

        if self.song_playing and self.song_chain:
            base_tracks = copy.deepcopy(self.tracks)
            scene_start_position = transport_position
            for slot in self.song_chain:
                scene_index = int(np.clip(slot["scene"], 0, PATTERN_SCENES - 1))
                scene_tracks = copy.deepcopy(base_tracks)
                for track, data in zip(scene_tracks, self.pattern_scenes[scene_index].get("tracks", [])):
                    self._load_track_pattern_locked(track, data)
                self._collect_cache_jobs_for_tracks_locked(
                    jobs,
                    scene_tracks,
                    scene_start_position,
                    global_fx_amount,
                )
                scene_start_position += sum(
                    self._step_samples(step)
                    for step in range(int(np.clip(slot["bars"], 1, 16)) * STEPS)
                )
        return list(jobs.values())

    def _collect_cache_jobs_for_tracks_locked(
        self,
        jobs: dict[tuple, tuple],
        tracks,
        transport_position: int,
        global_fx_amount: float,
    ):
        for track_index, track in enumerate(tracks):
            if not any(track.pattern):
                continue
            active_steps = [
                step
                for step, enabled in enumerate(track.pattern)
                if enabled and step < max(1, track.track_steps)
            ]
            phase_variant = self._has_lfo_phase_variation(track)
            if not self._has_note_overrides(track) and not phase_variant:
                active_steps = active_steps[:1]
            for step in active_steps:
                track_snapshot = self._render_track_for_step(track, step)
                step_position = transport_position + sum(self._step_samples(index) for index in range(step))
                repeats = int(np.clip(track.ratchets[step % STEPS], 1, 4)) if phase_variant else 1
                spacing = max(1, self._step_samples(step) // repeats)
                for repeat in range(repeats):
                    phase = self._lfo_phases_for_track(
                        track_snapshot,
                        step_position + repeat * spacing,
                    )
                    key, phase_key = self._cache_key_for_track(
                        track_index,
                        track_snapshot,
                        phase,
                        global_fx_amount,
                    )
                    if key not in self.render_cache and key not in jobs:
                        jobs[key] = (track_index, key, track_snapshot, phase_key, global_fx_amount)

    def _background_render_cache(self):
        while True:
            with self.lock:
                request_version = self.cache_request_version
                jobs = self._cache_jobs_locked()

            rendered = []
            for track_index, key, track, phase_key, global_fx_amount in jobs:
                rendered.append(
                    (
                        track_index,
                        key,
                        self._render_track_copy(track, phase_key, global_fx_amount),
                    )
                )

            with self.lock:
                for track_index, key, audio in rendered:
                    self.render_cache[key] = audio
                    self.fallback_hits[track_index] = audio
                if request_version == self.cache_request_version:
                    self.cache_render_thread = None
                    return

    def _render_audition_job(self, track_index, key, track, phase_key, global_fx_amount):
        audio = self._render_track_copy(track, phase_key, global_fx_amount)
        with self.lock:
            self.render_cache[key] = audio
            self.fallback_hits[track_index] = audio
            self.voices.append(SampleVoice(audio))

    def _render_step_hit_locked(
        self,
        track_index: int,
        step: int,
        lfo_phase: tuple[float | None, float | None] | float | None = None,
    ) -> np.ndarray:
        source_track = self.tracks[track_index]
        render_track = self._render_track_for_step(source_track, step)
        key, phase_key = self._cache_key_for_track(track_index, render_track, lfo_phase)
        if key not in self.render_cache:
            if (
                self.playing
                and track_index in self.fallback_hits
                and not self._has_note_overrides(source_track)
                and not self._has_lfo_phase_variation(render_track)
            ):
                return self.fallback_hits[track_index]
            self.render_cache[key] = self._render_track_copy(
                render_track,
                phase_key,
                float(self.global_fx_amount),
            )
            self.fallback_hits[track_index] = self.render_cache[key]
        return self.render_cache[key]

    def _render_track_for_step(self, track, step: int):
        render_track = copy.copy(track)
        if not self._uses_note_mode_for_step(track, step):
            render_track.bass_enabled = False
            if hasattr(render_track, "step_note_hz"):
                delattr(render_track, "step_note_hz")
        else:
            note = int(np.clip(track.bass_notes[step % STEPS], 12, 84))
            target_hz = self._midi_to_frequency(note)
            render_track.step_note_hz = target_hz
            semitones = note - 36
            render_track.pitch = float(np.clip(track.pitch * (2.0 ** (semitones / 12.0)), 0.25, 7.0))
        return render_track

    def _uses_note_mode_for_step(self, track, step: int) -> bool:
        if not bool(getattr(track, "bass_enabled", False)):
            return False
        note_flags = getattr(track, "bass_note_enabled", [True] * STEPS)
        return bool(note_flags[step % STEPS])

    def _has_note_overrides(self, track) -> bool:
        return bool(getattr(track, "bass_enabled", False)) and any(
            bool(value) for value in getattr(track, "bass_note_enabled", [True] * STEPS)
        )

    def _has_lfo_phase_variation(self, track) -> bool:
        return (
            bool(getattr(track, "lfo_enabled", False))
            and float(getattr(track, "lfo_amount", 0.0)) > 0.0
        ) or (
            bool(getattr(track, "lfo2_enabled", False))
            and float(getattr(track, "lfo2_amount", 0.0)) > 0.0
        )

    def _midi_to_frequency(self, note: int) -> float:
        return 440.0 * (2.0 ** ((int(note) - 69) / 12.0))

    def _render_hit_locked(
        self,
        track_index: int,
        lfo_phase: tuple[float | None, float | None] | float | None = None,
    ) -> np.ndarray:
        track = self.tracks[track_index]
        key, phase_key = self._cache_key_for_track(track_index, track, lfo_phase)
        if key not in self.render_cache:
            if (
                self.playing
                and track_index in self.fallback_hits
                and not self._has_lfo_phase_variation(track)
            ):
                return self.fallback_hits[track_index]
            self.render_cache[key] = self._render_track_copy(
                track,
                phase_key,
                float(self.global_fx_amount),
            )
            self.fallback_hits[track_index] = self.render_cache[key]
        return self.render_cache[key]

    def _split_phase(self, phase):
        if isinstance(phase, tuple):
            first = phase[0] if len(phase) > 0 else None
            second = phase[1] if len(phase) > 1 else None
            return first, second
        return phase, None

    def _lfo_phases_for_track(self, track, sample_position: int) -> tuple[float | None, float | None]:
        elapsed = sample_position / SAMPLE_RATE
        phase1 = (
            (track.lfo_phase + elapsed * track.lfo_rate) % 1.0
            if track.lfo_enabled and track.lfo_amount > 0.0
            else None
        )
        phase2 = (
            (track.lfo2_phase + elapsed * track.lfo2_rate) % 1.0
            if track.lfo2_enabled and track.lfo2_amount > 0.0
            else None
        )
        return phase1, phase2

    def _step_samples(self, step: int) -> int:
        beat_seconds = 60.0 / max(self.bpm, 1.0)
        base = beat_seconds / 4.0
        swing_amount = self.swing * 0.45
        multiplier = 1.0 + swing_amount if step % 2 == 0 else 1.0 - swing_amount
        return max(1, int(base * multiplier * SAMPLE_RATE))

    def _fill_step_settings(
        self,
        track_index: int,
        local_step: int,
        active: bool,
        probability: float,
        ratchets: int,
    ) -> tuple[bool, float, int, float]:
        if not self.fill_enabled:
            return active, probability, ratchets, 1.0

        step_in_bar = local_step % STEPS
        last_beat = step_in_bar in {12, 13, 14, 15}
        final_turn = step_in_bar in {14, 15}
        ghost_active = active
        velocity_scale = 1.0

        if track_index in {0, 1, 4}:
            return active, probability, ratchets, velocity_scale

        if active:
            probability = max(probability, 0.9 if last_beat else probability)
            if final_turn and track_index in {2, 6, 7}:
                ratchets = max(ratchets, 2)
            return ghost_active, probability, ratchets, velocity_scale

        fill_chance = 0.0
        if track_index == 2 and step_in_bar in {13, 15}:
            fill_chance = 0.68
            velocity_scale = 0.52
            ratchets = 2 if step_in_bar == 15 else 1
        elif track_index == 3 and step_in_bar == 15:
            fill_chance = 0.32
            velocity_scale = 0.42
        elif track_index in {6, 7} and final_turn:
            fill_chance = 0.42
            velocity_scale = 0.5
            ratchets = 2 if step_in_bar == 15 else 1
        elif track_index == 5 and step_in_bar == 15:
            fill_chance = 0.28
            velocity_scale = 0.48

        if fill_chance <= 0.0:
            return active, probability, ratchets, velocity_scale
        return True, fill_chance, ratchets, velocity_scale

    def _trigger_step_locked(self):
        step = self.current_step
        step_samples = self._step_samples(step)
        solo_active = any(track.solo for track in self.tracks)
        pattern_has_activity = self._patterns_have_activity(self.tracks)

        for track_index, track in enumerate(self.tracks):
            local_step = step % max(1, track.track_steps)
            audible = track.solo if solo_active else not track.muted
            probability = float(np.clip(track.probabilities[local_step], 0.0, 1.0))
            probability *= float(np.clip(self.global_density, 0.0, 1.5))
            ratchets = int(np.clip(track.ratchets[local_step], 1, 4))
            active = track.pattern[local_step]
            fill_velocity_scale = 1.0
            if pattern_has_activity:
                active, probability, ratchets, fill_velocity_scale = self._fill_step_settings(
                    track_index,
                    local_step,
                    active,
                    probability,
                    ratchets,
                )
            should_play = (
                audible
                and active
                and track.volume > 0.0
                and np.random.random() <= probability
            )
            if not should_play:
                continue

            spacing = max(1, step_samples // ratchets)
            for repeat in range(ratchets):
                event_sample_position = self.transport_sample_position + repeat * spacing
                lfo_phase = self._lfo_phases_for_track(track, event_sample_position)
                velocity = track.velocities[local_step] * fill_velocity_scale
                if repeat > 0:
                    velocity *= 0.72**repeat
                if self.global_humanize > 0.0:
                    spread = float(np.clip(self.global_humanize, 0.0, 1.0)) * 0.35
                    velocity *= float(np.random.uniform(1.0 - spread, 1.0 + spread))
                voice = SampleVoice(
                    self._render_step_hit_locked(track_index, local_step, lfo_phase),
                    velocity,
                )
                offset = repeat * spacing
                if offset == 0:
                    self.voices.append(voice)
                else:
                    self.scheduled.append((offset, voice))

        self.samples_until_step = step_samples
        self.current_step = (self.current_step + 1) % STEPS
        if self.current_step == 0:
            self._advance_song_bar_locked()

    def _render_scheduled_locked(self, dry, start: int, count: int):
        pending = []
        active = []
        for offset, voice in self.scheduled:
            if offset < count:
                if voice.render_into(dry, start + offset, count - offset):
                    active.append(voice)
            else:
                pending.append((offset - count, voice))
        self.scheduled = pending
        return active

    def _audio_callback(self, outdata, frames, time, status):
        del time, status
        dry = np.zeros((frames, CHANNELS), dtype=np.float32)
        write_pos = 0

        with self.lock:
            while write_pos < frames:
                if self.playing and self.samples_until_step <= 0:
                    self._trigger_step_locked()

                chunk = frames - write_pos
                if self.playing:
                    chunk = min(chunk, self.samples_until_step)

                active = []
                for voice in self.voices:
                    if voice.render_into(dry, write_pos, chunk):
                        active.append(voice)
                active.extend(self._render_scheduled_locked(dry, write_pos, chunk))
                self.voices = active

                write_pos += chunk
                if self.playing:
                    self.samples_until_step -= chunk
                    self.transport_sample_position += chunk
                else:
                    break

            master = self.master_volume
            compressor_amount = self.compressor_amount
            filter_cutoff = self.global_filter_cutoff
            filter_resonance = self.global_filter_resonance
            drive = self.global_drive

        mix = dry * master
        mix = apply_global_filter(mix, filter_cutoff, filter_resonance)
        if drive > 0.0:
            mix = np.tanh(mix * (1.0 + float(np.clip(drive, 0.0, 1.0)) * 5.0))
        processed = self.compressor.process(mix, compressor_amount)
        with self.lock:
            self._append_spectrum_samples_locked(processed)
        outdata[:] = processed

    def _append_spectrum_samples_locked(self, block: np.ndarray):
        samples = block.astype(np.float32, copy=False)
        if samples.ndim == 1:
            samples = np.column_stack((samples, samples))
        count = len(samples)
        if count >= len(self.spectrum_buffer):
            self.spectrum_buffer[:] = samples[-len(self.spectrum_buffer) :, :CHANNELS]
            self.spectrum_write_index = 0
            return
        end = self.spectrum_write_index + count
        if end <= len(self.spectrum_buffer):
            self.spectrum_buffer[self.spectrum_write_index:end] = samples[:, :CHANNELS]
        else:
            first = len(self.spectrum_buffer) - self.spectrum_write_index
            self.spectrum_buffer[self.spectrum_write_index:] = samples[:first, :CHANNELS]
            self.spectrum_buffer[: end % len(self.spectrum_buffer)] = samples[first:, :CHANNELS]
        self.spectrum_write_index = end % len(self.spectrum_buffer)


def output_spectrum_bands(block: np.ndarray, band_count: int = 24) -> np.ndarray:
    if len(block) < 8:
        return np.zeros(band_count, dtype=np.float32)
    mono = block.mean(axis=1).astype(np.float32, copy=False) if block.ndim == 2 else block.astype(np.float32, copy=False)
    window = np.hanning(len(mono)).astype(np.float32)
    magnitudes = np.abs(np.fft.rfft(mono * window))
    frequencies = np.fft.rfftfreq(len(mono), 1.0 / SAMPLE_RATE)
    edges = np.geomspace(35.0, SAMPLE_RATE * 0.45, band_count + 1)
    bands = np.zeros(band_count, dtype=np.float32)
    for index, (low, high) in enumerate(zip(edges[:-1], edges[1:])):
        mask = (frequencies >= low) & (frequencies < high)
        if np.any(mask):
            bands[index] = float(np.sqrt(np.mean(magnitudes[mask] * magnitudes[mask])))
    if np.max(bands) > 0.0:
        bands = np.log1p(bands)
        bands /= max(float(np.max(bands)), 0.001)
    return bands.astype(np.float32)


def output_spectrum_db_bands(block: np.ndarray, band_count: int = 56) -> np.ndarray:
    if len(block) < 8:
        return np.full(band_count, -72.0, dtype=np.float32)
    mono = block.mean(axis=1).astype(np.float32, copy=False) if block.ndim == 2 else block.astype(np.float32, copy=False)
    window = np.hanning(len(mono)).astype(np.float32)
    magnitudes = np.abs(np.fft.rfft(mono * window))
    magnitudes /= max(float(np.sum(window) * 0.5), 1.0)
    frequencies = np.fft.rfftfreq(len(mono), 1.0 / SAMPLE_RATE)
    edges = np.geomspace(25.0, SAMPLE_RATE * 0.45, band_count + 1)
    bands = np.full(band_count, -72.0, dtype=np.float32)
    for index, (low, high) in enumerate(zip(edges[:-1], edges[1:])):
        mask = (frequencies >= low) & (frequencies < high)
        if np.any(mask):
            level = float(np.sqrt(np.mean(magnitudes[mask] * magnitudes[mask])))
        else:
            center = float(np.sqrt(low * high))
            nearest = int(np.argmin(np.abs(frequencies - center)))
            level = float(magnitudes[nearest])
            if nearest > 0:
                level = max(level, float(magnitudes[nearest - 1]) * 0.45)
            if nearest + 1 < len(magnitudes):
                level = max(level, float(magnitudes[nearest + 1]) * 0.45)
        if level > 0.0:
            bands[index] = 20.0 * np.log10(max(level, 1.0e-6))
    return np.clip(bands, -72.0, 6.0).astype(np.float32)


def apply_global_filter(block: np.ndarray, cutoff: float, resonance: float) -> np.ndarray:
    cutoff = float(np.clip(cutoff, 40.0, SAMPLE_RATE * 0.45))
    if cutoff >= SAMPLE_RATE * 0.42 and resonance <= 0.01:
        return block
    left = one_pole_lowpass(block[:, 0], cutoff, resonance)
    right = one_pole_lowpass(block[:, 1], cutoff, resonance)
    return np.column_stack((left, right)).astype(np.float32)
