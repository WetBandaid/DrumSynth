import copy
import json
import sys
import wave
from pathlib import Path

import numpy as np

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QDial,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QTabWidget,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from .config import (
    FILTER_TYPES,
    GENERATOR_STYLES,
    DRUM_PRESET_NAMES,
    INSTRUMENTS,
    LFO_DESTINATIONS,
    LFO_SHAPES,
    PATTERN_SCENES,
    SATURATION_MODES,
    SAMPLE_RATE,
    STEPS,
)
from .engine import DrumEngine
from .kit_packs import KIT_PACK_NAMES
from .pattern_packs import PATTERN_PACK_NAMES
from .synth import make_hit
from .tooltips import CONTROL_TOOLTIPS, PARAM_TOOLTIPS
from .widgets import (
    ChannelSpectrumDisplay,
    EnvelopeEditor,
    LevelProfilePreview,
    OutputSpectrumMeter,
    SpectrumPreview,
    StepButton,
    WaveformPreview,
)

STEP_NOTE_MIN = 24
STEP_NOTE_MAX = 60
NOTE_NAMES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.engine = DrumEngine()
        self.step_buttons: list[list[StepButton]] = []
        self.track_labels: list[QLabel] = []
        self.track_mute_buttons: list[QToolButton] = []
        self.track_widgets = []
        self.global_widgets = {"dials": {}, "buttons": {}}
        self.scene_buttons: list[QToolButton] = []
        self.pattern_track_combo: QComboBox | None = None
        self.pattern_pack_combo: QComboBox | None = None
        self.song_rows_layout: QGridLayout | None = None
        self.song_play_button: QToolButton | None = None
        self.song_loop_button: QToolButton | None = None
        self.song_position_label: QLabel | None = None
        self.song_rows_scroll: QScrollArea | None = None
        self.song_row_labels: list[QLabel] = []
        self.selected_pattern_track = 0
        self.selected_step = 0
        self.inspector_widgets: dict[str, QWidget] = {}
        self.scene_name_edit: QLineEdit | None = None
        self.morph_scene_combos: list[QComboBox] = []
        self.morph_summary_label: QLabel | None = None
        self.channel_analyzers: list[ChannelSpectrumDisplay] = []
        self.analyzer_average_spin: QSpinBox | None = None
        self.preview_update_timers: dict[int, QTimer] = {}
        self.render_cache_debounce_timer = QTimer(self)
        self.render_cache_debounce_timer.setSingleShot(True)
        self.render_cache_debounce_timer.timeout.connect(self._flush_render_cache_refresh)
        self.last_highlighted_step = -1
        self.last_synced_scene = -1
        self.last_song_position = -1

        self.setWindowTitle("NumPy Drum Machine")
        self.resize(1180, 780)

        root = QWidget()
        self.setCentralWidget(root)
        layout = QVBoxLayout(root)
        layout.setSpacing(10)
        layout.addLayout(self._build_transport())

        self.main_tabs = QTabWidget()
        self.main_tabs.setToolTip("Switch between pattern sequencing, song arrangement, and track sound design.")
        self.main_tabs.addTab(self._build_sequencer_page(), "Sequencer")
        self.main_tabs.setTabToolTip(0, "Edit the current pattern scene and global performance controls.")
        self.main_tabs.addTab(self._build_song_page(), "Song Mode")
        self.main_tabs.setTabToolTip(1, "Chain pattern scenes into a longer arrangement.")
        self.main_tabs.addTab(self._build_analyzer_page(), "Analyzer")
        self.main_tabs.setTabToolTip(2, "View the final output spectrum for left and right channels.")
        self.main_tabs.addTab(self._build_patch_page(), "Track Sound Design")
        self.main_tabs.setTabToolTip(3, "Edit the drum sound, effects, modulation, and step expression for each track.")
        layout.addWidget(self.main_tabs, 1)

        self._apply_style()
        self._sync_from_engine()

        self.ui_timer = QTimer(self)
        self.ui_timer.timeout.connect(self._refresh_playhead)
        self.ui_timer.start(30)

        try:
            self.engine.start_stream()
        except Exception as exc:
            QMessageBox.critical(
                self,
                "Audio device error",
                f"Could not start sounddevice audio output:\n\n{exc}",
            )

    def closeEvent(self, event):
        self.engine.close()
        super().closeEvent(event)

    def _set_tip(self, widget: QWidget, key: str):
        widget.setToolTip(CONTROL_TOOLTIPS.get(key, PARAM_TOOLTIPS.get(key, "")))

    def _param_tip(self, widget: QWidget, param: str):
        widget.setToolTip(PARAM_TOOLTIPS.get(param, ""))

    def _build_transport(self) -> QVBoxLayout:
        layout = QVBoxLayout()
        layout.setSpacing(6)
        transport_row = QHBoxLayout()
        transport_row.setSpacing(8)

        self.play_button = QPushButton("Play")
        self.play_button.setCheckable(True)
        self._set_tip(self.play_button, "play")
        self.play_button.clicked.connect(self._toggle_playback)
        transport_row.addWidget(self.play_button)

        stop_button = QPushButton("Stop")
        self._set_tip(stop_button, "stop")
        stop_button.clicked.connect(self._stop)
        transport_row.addWidget(stop_button)

        transport_row.addWidget(QLabel("BPM"))
        self.bpm_spin = QDoubleSpinBox()
        self.bpm_spin.setRange(40.0, 240.0)
        self.bpm_spin.setDecimals(1)
        self.bpm_spin.setSingleStep(1.0)
        self._set_tip(self.bpm_spin, "bpm")
        self.bpm_spin.valueChanged.connect(self.engine.set_bpm)
        transport_row.addWidget(self.bpm_spin)

        transport_row.addWidget(QLabel("Swing"))
        self.swing_slider = QSlider(Qt.Horizontal)
        self.swing_slider.setRange(0, 100)
        self.swing_slider.setFixedWidth(140)
        self._set_tip(self.swing_slider, "swing")
        self.swing_slider.valueChanged.connect(lambda value: self.engine.set_swing(value / 100.0))
        transport_row.addWidget(self.swing_slider)

        transport_row.addWidget(QLabel("Master"))
        self.master_slider = QSlider(Qt.Horizontal)
        self.master_slider.setRange(0, 100)
        self.master_slider.setFixedWidth(140)
        self._set_tip(self.master_slider, "master_volume")
        self.master_slider.valueChanged.connect(
            lambda value: self.engine.set_master_volume(value / 100.0)
        )
        transport_row.addWidget(self.master_slider)

        transport_row.addStretch(1)
        self.output_meter = OutputSpectrumMeter()
        transport_row.addWidget(self.output_meter)
        layout.addLayout(transport_row)

        action_row = QHBoxLayout()
        action_row.setSpacing(8)
        for text, slot, tip_key in (
            ("Default", self._load_default, "default_pattern"),
            ("Clear", self._clear, "clear_pattern"),
            ("Random", self._randomize, "random_pattern"),
            ("Save", self._save_patch, "save_patch"),
            ("Load", self._load_patch, "load_patch"),
            ("Export Pattern", self._export_current_pattern, "export_pattern"),
            ("Export Song", self._export_song, "export_song"),
        ):
            button = QPushButton(text)
            self._set_tip(button, tip_key)
            button.clicked.connect(slot)
            action_row.addWidget(button)

        action_row.addStretch(1)
        layout.addLayout(action_row)
        return layout

    def _build_sequencer_page(self) -> QWidget:
        page = QScrollArea()
        page.setWidgetResizable(True)
        page.setMinimumSize(0, 0)
        page.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        content = QWidget()
        page.setWidget(content)
        layout = QVBoxLayout(content)
        layout.setContentsMargins(18, 18, 18, 18)
        top_row = QHBoxLayout()
        top_row.setSpacing(12)
        top_row.addWidget(self._build_grid(), 1)
        top_row.addWidget(self._build_step_inspector())
        layout.addLayout(top_row)
        layout.addWidget(self._build_pattern_tools())
        layout.addWidget(self._build_pattern_morph())
        layout.addWidget(self._build_global_controls())
        layout.addStretch(1)
        return page

    def _build_song_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.addWidget(self._build_song_tools())
        layout.addStretch(1)
        return page

    def _build_analyzer_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(12)

        controls = QHBoxLayout()
        controls.setSpacing(8)
        controls.addWidget(QLabel("Average Samples"))
        self.analyzer_average_spin = QSpinBox()
        self.analyzer_average_spin.setRange(1, 128)
        self.analyzer_average_spin.setValue(12)
        self.analyzer_average_spin.setKeyboardTracking(False)
        self.analyzer_average_spin.setFixedWidth(86)
        self.analyzer_average_spin.setToolTip(
            "Set how many analyzer frames the magenta trace averages."
        )
        self.analyzer_average_spin.valueChanged.connect(self._set_analyzer_average_window)
        controls.addWidget(self.analyzer_average_spin)
        controls.addStretch(1)
        layout.addLayout(controls)

        self.channel_analyzers = [
            ChannelSpectrumDisplay("Left"),
            ChannelSpectrumDisplay("Right"),
        ]
        self._set_analyzer_average_window(self.analyzer_average_spin.value())
        for analyzer in self.channel_analyzers:
            layout.addWidget(analyzer, 1)
        return page

    def _set_analyzer_average_window(self, value: int):
        for analyzer in self.channel_analyzers:
            analyzer.set_average_window(value)

    def _track_family(self, track_index: int) -> str:
        if track_index in {0, 5}:
            return "low"
        if track_index in {1, 4}:
            return "snap"
        if track_index in {2, 3}:
            return "metal"
        return "accent"

    def _build_grid(self) -> QGroupBox:
        group = QGroupBox("Sequencer")
        group.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        grid = QGridLayout(group)
        grid.setContentsMargins(12, 18, 12, 16)
        grid.setHorizontalSpacing(5)
        grid.setVerticalSpacing(9)
        grid.setColumnMinimumWidth(0, 88)
        grid.setColumnMinimumWidth(1, 34)
        grid.setColumnStretch(0, 0)
        grid.setColumnStretch(1, 0)

        track_header = QLabel("Track")
        track_header.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        grid.addWidget(track_header, 0, 0)

        mute_header = QLabel("M")
        mute_header.setAlignment(Qt.AlignCenter)
        mute_header.setToolTip("Mute track")
        grid.addWidget(mute_header, 0, 1)

        for step in range(STEPS):
            label = QLabel(str(step + 1))
            label.setAlignment(Qt.AlignCenter)
            label.setMinimumWidth(28)
            if step % 4 == 0:
                label.setProperty("barStart", True)
            grid.addWidget(label, 0, step + 2)
            grid.setColumnStretch(step + 2, 1)

        for track_index, track_name in enumerate(INSTRUMENTS):
            name = QLabel(track_name)
            name.setMinimumWidth(84)
            name.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
            name.setProperty("trackFamily", self._track_family(track_index))
            grid.addWidget(name, track_index + 1, 0)
            self.track_labels.append(name)

            mute = QToolButton()
            mute.setText("M")
            mute.setCheckable(True)
            mute.setFixedSize(30, 28)
            mute.setProperty("sequencerMute", True)
            self._set_tip(mute, "sequencer_mute")
            mute.toggled.connect(
                lambda checked, tr=track_index: self._set_track_mute(tr, checked)
            )
            grid.addWidget(mute, track_index + 1, 1)
            self.track_mute_buttons.append(mute)

            row = []
            for step in range(STEPS):
                button = StepButton(track_index, step)
                button.setProperty("barStart", step % 4 == 0)
                button.setProperty("trackFamily", self._track_family(track_index))
                button.toggled.connect(
                    lambda checked, tr=track_index, st=step: self.engine.set_step(
                        tr, st, checked
                    )
                )
                button.clicked.connect(
                    lambda checked=False, tr=track_index, st=step: self._select_step(tr, st)
                )
                button.rightClicked.connect(self._select_step)
                grid.addWidget(button, track_index + 1, step + 2)
                row.append(button)
            self.step_buttons.append(row)

        return group

    def _build_step_inspector(self) -> QGroupBox:
        group = QGroupBox("Step Inspector")
        group.setMinimumWidth(230)
        group.setMaximumWidth(280)
        layout = QVBoxLayout(group)
        layout.setContentsMargins(14, 18, 14, 16)
        layout.setSpacing(10)

        title = QLabel("Kick / Step 1")
        title.setProperty("inspectorTitle", True)
        self.inspector_widgets["title"] = title
        layout.addWidget(title)

        badge = QLabel("")
        badge.setProperty("trackBadge", True)
        badge.setWordWrap(True)
        self.inspector_widgets["badge"] = badge
        layout.addWidget(badge)

        active = QToolButton()
        active.setText("Step On")
        active.setCheckable(True)
        self._set_tip(active, "inspector_active")
        active.toggled.connect(self._set_inspector_step_active)
        self.inspector_widgets["active"] = active
        layout.addWidget(active)

        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignLeft)
        layout.addLayout(form)

        velocity = self._step_spin(0, 150, "%")
        self._set_tip(velocity, "step_velocity")
        velocity.valueChanged.connect(
            lambda value: self._set_inspector_step_param("velocities", value / 100.0)
        )
        self.inspector_widgets["velocity"] = velocity
        form.addRow("Velocity", velocity)

        probability = self._step_spin(0, 100, "%")
        self._set_tip(probability, "step_probability")
        probability.valueChanged.connect(
            lambda value: self._set_inspector_step_param("probabilities", value / 100.0)
        )
        self.inspector_widgets["probability"] = probability
        form.addRow("Probability", probability)

        ratchet = QSpinBox()
        ratchet.setRange(1, 4)
        ratchet.setKeyboardTracking(False)
        self._set_tip(ratchet, "step_ratchet")
        ratchet.valueChanged.connect(lambda value: self._set_inspector_step_param("ratchets", value))
        self.inspector_widgets["ratchet"] = ratchet
        form.addRow("Ratchet", ratchet)

        bass_note = QComboBox()
        for note in range(STEP_NOTE_MIN, STEP_NOTE_MAX + 1):
            bass_note.addItem(self._note_name(note), note)
        self._set_tip(bass_note, "step_bass_note")
        bass_note.currentIndexChanged.connect(
            lambda index, combo=bass_note: self._set_inspector_step_param(
                "bass_notes",
                combo.itemData(index),
            )
        )
        self.inspector_widgets["bass_note"] = bass_note
        form.addRow("Step Note", bass_note)

        button_row = QHBoxLayout()
        mute = QToolButton()
        mute.setText("Mute")
        mute.setCheckable(True)
        self._set_tip(mute, "mute")
        mute.toggled.connect(lambda checked: self._set_inspector_track_param("muted", checked))
        self.inspector_widgets["mute"] = mute
        button_row.addWidget(mute)

        solo = QToolButton()
        solo.setText("Solo")
        solo.setCheckable(True)
        self._set_tip(solo, "solo")
        solo.toggled.connect(lambda checked: self._set_inspector_track_param("solo", checked))
        self.inspector_widgets["solo"] = solo
        button_row.addWidget(solo)
        layout.addLayout(button_row)

        action_row = QHBoxLayout()
        audition = QPushButton("Audition")
        self._set_tip(audition, "inspector_audition")
        audition.clicked.connect(
            lambda checked=False: self.engine.audition_step(
                self.selected_pattern_track,
                self.selected_step,
            )
        )
        action_row.addWidget(audition)

        edit_sound = QPushButton("Edit Sound")
        self._set_tip(edit_sound, "inspector_edit_sound")
        edit_sound.clicked.connect(self._jump_to_selected_sound)
        action_row.addWidget(edit_sound)
        layout.addLayout(action_row)

        layout.addStretch(1)
        return group

    def _build_pattern_tools(self) -> QGroupBox:
        group = QGroupBox("Pattern Scenes")
        layout = QVBoxLayout(group)
        layout.setContentsMargins(14, 18, 14, 16)
        layout.setSpacing(8)

        scene_row = QHBoxLayout()
        scene_row.setSpacing(8)
        scene_row.addWidget(QLabel("Scene"))
        for index in range(PATTERN_SCENES):
            button = QToolButton()
            button.setText(str(index + 1))
            button.setCheckable(True)
            button.setMinimumSize(34, 32)
            button.setToolTip(f"{CONTROL_TOOLTIPS['scene_button']} This is scene {index + 1}.")
            button.clicked.connect(
                lambda checked=False, scene_index=index: self._select_scene(scene_index)
            )
            scene_row.addWidget(button)
            self.scene_buttons.append(button)

        scene_row.addSpacing(10)
        scene_row.addWidget(QLabel("Name"))
        self.scene_name_edit = QLineEdit()
        self.scene_name_edit.setMaximumWidth(180)
        self._set_tip(self.scene_name_edit, "scene_name")
        self.scene_name_edit.editingFinished.connect(self._rename_current_scene)
        scene_row.addWidget(self.scene_name_edit)
        scene_row.addStretch(1)
        layout.addLayout(scene_row)

        scene_action_row = QHBoxLayout()
        scene_action_row.setSpacing(8)
        scene_action_row.addWidget(QLabel("Scene Tools"))
        for text, slot, tip_key in (
            ("Store", self._store_scene, "store_scene"),
            ("Copy Scene", self._copy_scene, "copy_scene"),
            ("Paste Scene", self._paste_scene, "paste_scene"),
            ("Clear Scene", self._clear_scene, "clear_scene"),
            ("Clear All", self._clear_all_patterns, "clear_all_patterns"),
        ):
            button = QPushButton(text)
            self._set_tip(button, tip_key)
            button.clicked.connect(slot)
            scene_action_row.addWidget(button)

        scene_action_row.addStretch(1)
        layout.addLayout(scene_action_row)

        groove_row = QHBoxLayout()
        groove_row.setSpacing(8)
        groove_row.addWidget(QLabel("Groove"))
        self.pattern_pack_combo = QComboBox()
        self.pattern_pack_combo.addItems(PATTERN_PACK_NAMES)
        self._set_tip(self.pattern_pack_combo, "pattern_pack")
        groove_row.addWidget(self.pattern_pack_combo)

        for text, apply_kit, tip_key in (
            ("Load Groove", False, "apply_pattern_pack"),
            ("Groove + Kit", True, "apply_pattern_pack_with_kit"),
        ):
            button = QPushButton(text)
            self._set_tip(button, tip_key)
            button.clicked.connect(lambda checked=False, kit=apply_kit: self._apply_pattern_pack(kit))
            groove_row.addWidget(button)

        groove_row.addStretch(1)
        layout.addLayout(groove_row)

        track_row = QHBoxLayout()
        track_row.setSpacing(8)
        track_row.addWidget(QLabel("Track"))
        self.pattern_track_combo = QComboBox()
        self.pattern_track_combo.addItems(INSTRUMENTS)
        self._set_tip(self.pattern_track_combo, "pattern_track")
        self.pattern_track_combo.currentIndexChanged.connect(self._set_pattern_tool_track)
        track_row.addWidget(self.pattern_track_combo)

        for text, slot, tip_key in (
            ("Copy Track", self._copy_track_pattern, "copy_track"),
            ("Paste Track", self._paste_track_pattern, "paste_track"),
            ("Clear Track", self._clear_track_pattern, "clear_track"),
            ("Rotate Left", lambda: self._rotate_track_pattern(-1), "rotate_left"),
            ("Rotate Right", lambda: self._rotate_track_pattern(1), "rotate_right"),
        ):
            button = QPushButton(text)
            self._set_tip(button, tip_key)
            button.clicked.connect(slot)
            track_row.addWidget(button)

        track_row.addStretch(1)
        layout.addLayout(track_row)
        return group

    def _build_pattern_morph(self) -> QGroupBox:
        group = QGroupBox("Pattern Morph")
        layout = QGridLayout(group)
        layout.setContentsMargins(14, 18, 14, 16)
        layout.setHorizontalSpacing(12)
        layout.setVerticalSpacing(8)

        scene_names = [f"Scene {index + 1}" for index in range(PATTERN_SCENES)]

        self.morph_source_combo = QComboBox()
        self.morph_source_combo.addItems(scene_names)
        self._set_tip(self.morph_source_combo, "morph_source")
        self.morph_source_combo.currentIndexChanged.connect(self._sync_morph_summary)
        self.morph_scene_combos.append(self.morph_source_combo)
        layout.addWidget(QLabel("From"), 0, 0)
        layout.addWidget(self.morph_source_combo, 0, 1)

        self.morph_target_combo = QComboBox()
        self.morph_target_combo.addItems(scene_names)
        self.morph_target_combo.setCurrentIndex(1 if PATTERN_SCENES > 1 else 0)
        self._set_tip(self.morph_target_combo, "morph_target")
        self.morph_target_combo.currentIndexChanged.connect(self._sync_morph_summary)
        self.morph_scene_combos.append(self.morph_target_combo)
        layout.addWidget(QLabel("To"), 0, 2)
        layout.addWidget(self.morph_target_combo, 0, 3)

        self.morph_destination_combo = QComboBox()
        self.morph_destination_combo.addItems(scene_names)
        self._set_tip(self.morph_destination_combo, "morph_destination")
        self.morph_destination_combo.currentIndexChanged.connect(self._sync_morph_summary)
        self.morph_scene_combos.append(self.morph_destination_combo)
        layout.addWidget(QLabel("Write"), 0, 4)
        layout.addWidget(self.morph_destination_combo, 0, 5)

        self.morph_amount_slider = QSlider(Qt.Horizontal)
        self.morph_amount_slider.setRange(0, 100)
        self.morph_amount_slider.setValue(50)
        self.morph_amount_slider.setMinimumWidth(180)
        self._set_tip(self.morph_amount_slider, "morph_amount")
        self.morph_amount_label = QLabel("50%")
        self.morph_amount_label.setAlignment(Qt.AlignCenter)
        self.morph_amount_label.setMinimumWidth(42)
        self._set_tip(self.morph_amount_label, "morph_amount")
        self.morph_amount_slider.valueChanged.connect(
            lambda value: self._set_morph_amount_label(value)
        )
        layout.addWidget(QLabel("Morph"), 1, 0)
        layout.addWidget(self.morph_amount_slider, 1, 1, 1, 4)
        layout.addWidget(self.morph_amount_label, 1, 5)

        apply_morph = QPushButton("Apply Morph")
        self._set_tip(apply_morph, "apply_morph")
        apply_morph.clicked.connect(self._apply_pattern_morph)
        layout.addWidget(apply_morph, 0, 6, 2, 1)

        self.morph_summary_label = QLabel("")
        self.morph_summary_label.setProperty("morphSummary", True)
        self._set_tip(self.morph_summary_label, "morph_summary")
        layout.addWidget(self.morph_summary_label, 2, 0, 1, 7)
        self._sync_morph_summary()

        layout.setColumnStretch(1, 1)
        layout.setColumnStretch(3, 1)
        layout.setColumnStretch(5, 1)
        return group

    def _build_song_tools(self) -> QGroupBox:
        group = QGroupBox("Song Mode")
        group.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        layout = QVBoxLayout(group)
        layout.setContentsMargins(14, 18, 14, 16)
        layout.setSpacing(10)

        top_row = QHBoxLayout()
        self.song_play_button = QToolButton()
        self.song_play_button.setText("Play Song")
        self.song_play_button.setCheckable(True)
        self._set_tip(self.song_play_button, "song_play")
        self.song_play_button.toggled.connect(self._toggle_song_playback)
        top_row.addWidget(self.song_play_button)

        self.song_loop_button = QToolButton()
        self.song_loop_button.setText("Loop")
        self.song_loop_button.setCheckable(True)
        self._set_tip(self.song_loop_button, "song_loop")
        self.song_loop_button.toggled.connect(self.engine.set_song_loop)
        top_row.addWidget(self.song_loop_button)

        add_slot = QPushButton("Add Slot")
        self._set_tip(add_slot, "song_add")
        add_slot.clicked.connect(self._add_song_slot)
        top_row.addWidget(add_slot)

        self.song_position_label = QLabel("Slot 1")
        self.song_position_label.setMinimumWidth(120)
        self._set_tip(self.song_position_label, "song_position")
        top_row.addWidget(self.song_position_label)
        top_row.addStretch(1)
        layout.addLayout(top_row)

        generator = QGroupBox("Session Generator")
        generator_grid = QGridLayout(generator)
        generator_grid.setContentsMargins(12, 16, 12, 12)
        generator_grid.setHorizontalSpacing(12)
        generator_grid.setVerticalSpacing(8)

        self.generator_patterns = QSpinBox()
        self.generator_patterns.setRange(1, PATTERN_SCENES)
        self.generator_patterns.setValue(min(4, PATTERN_SCENES))
        self.generator_patterns.setKeyboardTracking(False)
        self._set_tip(self.generator_patterns, "generate_patterns")
        startup = self.engine.startup_generation
        generator_grid.addWidget(QLabel("Patterns"), 0, 0)
        generator_grid.addWidget(self.generator_patterns, 0, 1)

        self.generator_bars = QSpinBox()
        self.generator_bars.setRange(1, 16)
        self.generator_bars.setValue(4)
        self.generator_bars.setKeyboardTracking(False)
        self._set_tip(self.generator_bars, "generate_bars")
        if startup:
            self.generator_bars.setValue(int(startup.get("bars_per_pattern", self.generator_bars.value())))
        generator_grid.addWidget(QLabel("Bars"), 0, 2)
        generator_grid.addWidget(self.generator_bars, 0, 3)

        self.generator_style = QComboBox()
        self.generator_style.addItems(GENERATOR_STYLES)
        if startup.get("style") in GENERATOR_STYLES:
            self.generator_style.setCurrentText(startup["style"])
        self._set_tip(self.generator_style, "generate_style")
        generator_grid.addWidget(QLabel("Style"), 0, 4)
        generator_grid.addWidget(self.generator_style, 0, 5)

        self.generator_complexity, complexity_value = self._generator_slider(55, "generate_complexity")
        self.generator_fills, fills_value = self._generator_slider(45, "generate_fills")
        self.generator_variation, variation_value = self._generator_slider(35, "generate_variation")
        if startup:
            for slider, label, key in (
                (self.generator_complexity, complexity_value, "complexity"),
                (self.generator_fills, fills_value, "fills"),
                (self.generator_variation, variation_value, "variation"),
            ):
                value = round(float(startup.get(key, slider.value() / 100.0)) * 100)
                slider.setValue(value)
                label.setText(f"{value}%")
        for column, (label, slider, value_label) in enumerate(
            (
                ("Complexity", self.generator_complexity, complexity_value),
                ("Fills", self.generator_fills, fills_value),
                ("Variation", self.generator_variation, variation_value),
            )
        ):
            generator_grid.addWidget(QLabel(label), 1, column * 2)
            generator_grid.addWidget(slider, 1, column * 2 + 1)
            generator_grid.addWidget(value_label, 2, column * 2 + 1)

        generate = QPushButton("Generate")
        self._set_tip(generate, "generate_session")
        generate.clicked.connect(self._generate_session)
        generator_grid.addWidget(generate, 0, 6, 3, 1)
        generator_grid.setColumnStretch(1, 1)
        generator_grid.setColumnStretch(3, 1)
        generator_grid.setColumnStretch(5, 1)
        layout.addWidget(generator)

        rows_content = QWidget()
        self.song_rows_layout = QGridLayout(rows_content)
        self.song_rows_layout.setContentsMargins(8, 8, 8, 8)
        self.song_rows_layout.setHorizontalSpacing(8)
        self.song_rows_layout.setVerticalSpacing(6)
        self.song_rows_layout.setAlignment(Qt.AlignTop)

        self.song_rows_scroll = QScrollArea()
        self.song_rows_scroll.setWidgetResizable(True)
        self.song_rows_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.song_rows_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.song_rows_scroll.setMinimumHeight(260)
        self.song_rows_scroll.setMaximumHeight(480)
        self.song_rows_scroll.setToolTip("Song arrangement slots. Each row chooses a scene and how many bars it plays.")
        self.song_rows_scroll.setWidget(rows_content)
        layout.addWidget(self.song_rows_scroll)
        return group

    def _generator_slider(self, value: int, tip_key: str):
        slider = QSlider(Qt.Horizontal)
        slider.setRange(0, 100)
        slider.setValue(value)
        slider.setMinimumWidth(120)
        self._set_tip(slider, tip_key)
        value_label = QLabel(f"{value}%")
        value_label.setAlignment(Qt.AlignCenter)
        value_label.setMinimumWidth(42)
        self._set_tip(value_label, tip_key)
        slider.valueChanged.connect(lambda new_value, label=value_label: label.setText(f"{new_value}%"))
        return slider, value_label

    def _build_global_controls(self) -> QGroupBox:
        group = QGroupBox("Global Performance")
        group.setProperty("compactGlobal", True)
        group.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        layout = QGridLayout(group)
        layout.setContentsMargins(8, 10, 8, 8)
        layout.setHorizontalSpacing(10)
        layout.setVerticalSpacing(4)

        controls = [
            ("global_filter_cutoff", "Cutoff", 200, 18000, 1),
            ("global_filter_resonance", "Res", 0, 95, 100),
            ("global_drive", "Drive", 0, 100, 100),
            ("compressor_amount", "Comp", 0, 100, 100),
            ("global_fx_amount", "FX Amt", 0, 150, 100),
            ("global_density", "Density", 0, 150, 100),
            ("global_humanize", "Humanize", 0, 100, 100),
        ]
        for index, spec in enumerate(controls):
            self._add_global_dial(layout, *spec, 0, index)

        fill = QToolButton()
        fill.setText("Fill")
        fill.setCheckable(True)
        fill.setFixedSize(54, 32)
        self._set_tip(fill, "fill_enabled")
        fill.toggled.connect(
            lambda checked: self.engine.set_global_param("fill_enabled", checked)
        )
        layout.addWidget(fill, 0, len(controls), alignment=Qt.AlignCenter)
        self.global_widgets["buttons"]["fill_enabled"] = fill
        layout.setColumnStretch(len(controls) + 1, 1)
        return group

    def _build_patch_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(14, 14, 14, 14)

        self.track_tabs = QTabWidget()
        self.track_tabs.setToolTip("Choose which drum track to edit.")
        layout.addWidget(self.track_tabs, 1)

        for track_index, track in enumerate(self.engine.tracks):
            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            scroll.setToolTip(f"Scrollable sound-design editor for {track.name}.")
            content = QWidget()
            scroll.setWidget(content)
            self.track_tabs.addTab(scroll, track.name)
            self.track_tabs.setTabToolTip(track_index, f"Edit sound design controls for {track.name}.")
            self._build_track_editor(content, track_index)

        return page

    def _build_track_editor(self, page: QWidget, track_index: int):
        layout = QVBoxLayout(page)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        controls = {
            "instrument": None,
            "mute": None,
            "solo": None,
            "dials": {},
            "spins": {},
            "combos": {},
            "checks": {},
            "envelopes": {},
            "step": {},
            "header_title": None,
            "header_badges": None,
            "waveform": None,
            "level_profile": None,
            "spectrum": None,
        }

        header = QGroupBox()
        header.setProperty("soundHeader", True)
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(10, 7, 10, 7)
        header_layout.setSpacing(7)

        title_column = QVBoxLayout()
        title_column.setSpacing(2)
        title = QLabel(self.engine.tracks[track_index].instrument)
        title.setProperty("soundTitle", True)
        badge = QLabel("")
        badge.setProperty("trackBadge", True)
        controls["header_title"] = title
        controls["header_badges"] = badge
        title_column.addWidget(title)
        title_column.addWidget(badge)
        header_layout.addLayout(title_column)

        kit_label = QLabel("Kit")
        kit_label.setProperty("compactLabel", True)
        header_layout.addWidget(kit_label)

        kit_pack = QComboBox()
        kit_pack.addItems(KIT_PACK_NAMES)
        kit_pack.setMinimumWidth(128)
        self._set_tip(kit_pack, "kit_pack")
        header_layout.addWidget(kit_pack)

        apply_kit = QPushButton("Apply Kit")
        self._set_tip(apply_kit, "apply_kit_pack")
        apply_kit.clicked.connect(
            lambda checked=False, tr=track_index, combo=kit_pack: self._apply_kit_pack(
                tr, combo.currentText()
            )
        )
        header_layout.addWidget(apply_kit)

        header_layout.addSpacing(10)

        instrument = QComboBox()
        instrument.addItems(DRUM_PRESET_NAMES)
        self._set_tip(instrument, "track_preset")
        controls["instrument"] = instrument
        preset_label = QLabel("Preset")
        preset_label.setProperty("compactLabel", True)
        header_layout.addWidget(preset_label)
        header_layout.addWidget(instrument)

        apply_preset = QPushButton("Apply")
        self._set_tip(apply_preset, "apply_preset")
        apply_preset.clicked.connect(
            lambda checked=False, tr=track_index, combo=instrument: self._apply_track_preset(
                tr, combo.currentText()
            )
        )
        header_layout.addWidget(apply_preset)

        save_track_preset = QPushButton("Save")
        self._set_tip(save_track_preset, "save_track_preset")
        save_track_preset.clicked.connect(
            lambda checked=False, tr=track_index: self._save_track_preset(tr)
        )
        header_layout.addWidget(save_track_preset)

        load_track_preset = QPushButton("Load")
        self._set_tip(load_track_preset, "load_track_preset")
        load_track_preset.clicked.connect(
            lambda checked=False, tr=track_index: self._load_track_preset(tr)
        )
        header_layout.addWidget(load_track_preset)

        audition = QPushButton("Audition")
        self._set_tip(audition, "audition")
        audition.clicked.connect(lambda checked=False, tr=track_index: self.engine.audition_track(tr))
        header_layout.addWidget(audition)

        mute = QToolButton()
        mute.setText("M")
        mute.setCheckable(True)
        self._set_tip(mute, "mute")
        mute.toggled.connect(
            lambda checked, tr=track_index: self._set_track_mute(tr, checked)
        )
        controls["mute"] = mute
        header_layout.addWidget(mute)

        solo = QToolButton()
        solo.setText("S")
        solo.setCheckable(True)
        self._set_tip(solo, "solo")
        solo.toggled.connect(
            lambda checked, tr=track_index: self.engine.set_track_param(tr, "solo", checked)
        )
        controls["solo"] = solo
        header_layout.addWidget(solo)
        header_layout.addStretch(1)
        layout.addWidget(header)

        dials_group = QGroupBox("Performance")
        dials_group.setProperty("soundPanel", True)
        dials = QGridLayout(dials_group)
        dials.setHorizontalSpacing(8)
        dials.setVerticalSpacing(6)

        dial_specs = [
            ("volume", "Volume", 0, 120, 100),
            ("decay", "Length", 5, 100, 100),
            ("pitch", "Pitch", 35, 250, 100),
            ("drive", "Drive", 0, 100, 100),
            ("pan", "Pan", -100, 100, 100),
            ("transient_attack", "Transient", -100, 100, 100),
            ("transient_body", "Body", 0, 200, 100),
            ("track_steps", "Steps", 1, 16, 1),
            ("tone_level", "Tone", 0, 150, 100),
            ("noise_level", "Noise", 0, 150, 100),
            ("click_level", "Click", 0, 150, 100),
            ("kick_mute", "Dampen", 0, 100, 100),
            ("filter_resonance", "Res", 0, 95, 100),
        ]
        for index, spec in enumerate(dial_specs):
            self._add_track_dial(dials, controls, track_index, *spec, 0, index, size=44)
        layout.addWidget(dials_group)

        editor_columns = QHBoxLayout()
        editor_columns.setSpacing(8)
        layout.addLayout(editor_columns, 1)

        left_column = QVBoxLayout()
        left_column.setContentsMargins(0, 0, 0, 0)
        left_column.setSpacing(8)
        left_host = QWidget()
        left_host.setMaximumWidth(430)
        left_host.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        left_host.setLayout(left_column)
        editor_columns.addWidget(left_host)

        right_host = QWidget()
        right_host.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        right_layout = QHBoxLayout(right_host)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(8)
        editor_columns.addWidget(right_host, 1)

        editor_tabs = QTabWidget()
        editor_tabs.setDocumentMode(True)
        editor_tabs.setMinimumHeight(190)
        editor_tabs.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        editor_tabs.setToolTip("Core synthesis and tone controls for this track.")
        left_column.addWidget(editor_tabs, 1)

        synthesis = QWidget()
        synthesis_form = self._form_layout(synthesis)
        editor_tabs.addTab(synthesis, "Synthesis")
        editor_tabs.setTabToolTip(0, "Core voice controls such as saturation, attack, and digital reduction.")

        tone = QWidget()
        tone_form = self._form_layout(tone)
        editor_tabs.addTab(tone, "Tone / Filter")
        editor_tabs.setTabToolTip(1, "Pitched tone, noise decay, pitch envelope, and filter controls.")

        detail_tabs = QTabWidget()
        detail_tabs.setDocumentMode(True)
        detail_tabs.setMinimumHeight(190)
        detail_tabs.setMinimumWidth(360)
        detail_tabs.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        detail_tabs.setToolTip("Effects, modulation, and per-step expression controls for this track.")
        right_layout.addWidget(detail_tabs, 1)

        signal_group = QGroupBox("Signal")
        signal_group.setProperty("soundPanel", True)
        signal_group.setMinimumWidth(360)
        signal_group.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        signal_layout = QVBoxLayout(signal_group)
        signal_layout.setContentsMargins(8, 10, 8, 8)
        signal_layout.setSpacing(6)

        waveform = WaveformPreview()
        level_profile = LevelProfilePreview()
        spectrum = SpectrumPreview()
        controls["waveform"] = waveform
        controls["level_profile"] = level_profile
        controls["spectrum"] = spectrum
        signal_layout.addWidget(waveform, 1)
        signal_layout.addWidget(level_profile, 1)
        signal_layout.addWidget(spectrum, 1)
        right_layout.addWidget(signal_group, 2)

        effects = QWidget()
        effects_form = self._form_layout(effects)
        detail_tabs.addTab(effects, "Effects")
        detail_tabs.setTabToolTip(0, "Per-patch delay and reverb send controls.")

        modulation = QWidget()
        modulation_layout = QVBoxLayout(modulation)
        modulation_layout.setContentsMargins(4, 4, 4, 4)
        modulation_layout.setSpacing(6)
        modulation_tabs = QTabWidget()
        modulation_tabs.setDocumentMode(True)
        modulation_tabs.setMinimumHeight(190)
        modulation_tabs.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        modulation_tabs.setToolTip("Switch between this patch's modulation sources.")
        modulation_layout.addWidget(modulation_tabs, 1)
        detail_tabs.addTab(modulation, "Modulation")
        detail_tabs.setTabToolTip(1, "LFO and envelope modulation controls for this patch.")

        lfo1_page, lfo1_form, lfo_grid = self._build_lfo_modulation_page()
        modulation_tabs.addTab(lfo1_page, "LFO 1")
        modulation_tabs.setTabToolTip(0, "First per-patch low-frequency oscillator.")

        lfo2_page, lfo2_form, lfo2_grid = self._build_lfo_modulation_page()
        modulation_tabs.addTab(lfo2_page, "LFO 2")
        modulation_tabs.setTabToolTip(1, "Second per-patch low-frequency oscillator.")

        envelope_page, envelope_form, env_grid, envelope_graph_layout = self._build_envelope_modulation_page()
        modulation_tabs.addTab(envelope_page, "Envelope")
        modulation_tabs.setTabToolTip(2, "One-shot modulation envelope for per-hit movement.")

        step_group = QWidget()
        step_form = self._form_layout(step_group)
        detail_tabs.addTab(step_group, "Step")
        detail_tabs.setTabToolTip(2, "Per-step velocity, probability, and ratchet settings.")

        self._add_combo(
            synthesis, synthesis_form, controls, track_index, "saturation_mode", "Saturation", SATURATION_MODES
        )
        self._add_check(synthesis_form, controls, track_index, "bass_enabled", "Note Mode")
        self._add_spin(synthesis_form, controls, track_index, "attack_ms", "Attack", 0, 30, 0.1, 1, " ms")
        self._add_spin(synthesis_form, controls, track_index, "bit_depth", "Bit Depth", 4, 16, 1, 1, " bit")
        self._add_spin(
            synthesis_form,
            controls,
            track_index,
            "sample_rate_reduction",
            "Rate Crush",
            1,
            32,
            1,
            1,
            "x",
        )

        self._add_combo(tone, tone_form, controls, track_index, "filter_type", "Filter", FILTER_TYPES)
        self._add_spin(tone_form, controls, track_index, "filter_cutoff", "Cutoff", 20, 18000, 10, 1, " Hz")
        self._add_spin(tone_form, controls, track_index, "tone_start", "Start Freq", 20, 12000, 1, 1, " Hz")
        self._add_spin(tone_form, controls, track_index, "tone_end", "End Freq", 20, 12000, 1, 1, " Hz")
        self._add_spin(tone_form, controls, track_index, "tone_decay", "Tone Decay", 0.005, 1.5, 0.005, 1, " s")
        self._add_spin(tone_form, controls, track_index, "noise_decay", "Noise Decay", 0.005, 1.5, 0.005, 1, " s")
        self._add_spin(tone_form, controls, track_index, "pitch_env_amount", "Pitch Env", -48, 48, 1, 1, " st")
        self._add_spin(tone_form, controls, track_index, "pitch_env_decay", "Env Decay", 0.005, 1.0, 0.005, 1, " s")

        fx_grid = QGridLayout()
        fx_grid.setHorizontalSpacing(8)
        fx_grid.setVerticalSpacing(6)
        effects_form.addRow(fx_grid)
        fx_specs = [
            ("delay_send", "Delay", 0, 100, 100),
            ("delay_feedback", "Feedback", 0, 88, 100),
            ("delay_tone", "Delay Tone", 0, 100, 100),
            ("delay_width", "Width", 0, 100, 100),
            ("reverb_send", "Reverb", 0, 100, 100),
            ("reverb_size", "Size", 10, 100, 100),
            ("reverb_decay", "Decay", 0, 92, 100),
            ("reverb_tone", "Rev Tone", 0, 100, 100),
        ]
        for index, spec in enumerate(fx_specs):
            self._add_track_dial(fx_grid, controls, track_index, *spec, index // 4, index % 4, size=44)
        self._add_spin(effects_form, controls, track_index, "delay_time", "Delay Time", 0.03, 1.5, 0.01, 1, " s")

        self._add_check(lfo1_form, controls, track_index, "lfo_enabled", "Enabled")
        self._add_combo(
            lfo1_page,
            lfo1_form,
            controls,
            track_index,
            "lfo_shape",
            "Shape",
            LFO_SHAPES,
        )
        self._add_combo(
            lfo1_page,
            lfo1_form,
            controls,
            track_index,
            "lfo_destination",
            "Destination",
            LFO_DESTINATIONS,
        )
        self._add_spin(lfo1_form, controls, track_index, "lfo_rate", "Rate", 0.05, 80, 0.05, 1, " Hz")
        self._add_spin(lfo1_form, controls, track_index, "lfo_phase", "Phase", 0, 100, 1, 100, "%")
        self._add_track_dial(lfo_grid, controls, track_index, "lfo_amount", "Amount", 0, 100, 100, 0, 0, size=44)

        self._add_check(lfo2_form, controls, track_index, "lfo2_enabled", "Enabled")
        self._add_combo(
            lfo2_page,
            lfo2_form,
            controls,
            track_index,
            "lfo2_shape",
            "Shape",
            LFO_SHAPES,
        )
        self._add_combo(
            lfo2_page,
            lfo2_form,
            controls,
            track_index,
            "lfo2_destination",
            "Destination",
            LFO_DESTINATIONS,
        )
        self._add_spin(lfo2_form, controls, track_index, "lfo2_rate", "Rate", 0.05, 80, 0.05, 1, " Hz")
        self._add_spin(lfo2_form, controls, track_index, "lfo2_phase", "Phase", 0, 100, 1, 100, "%")
        self._add_track_dial(lfo2_grid, controls, track_index, "lfo2_amount", "Amount", 0, 100, 100, 0, 0, size=44)

        self._add_check(envelope_form, controls, track_index, "env_mod_enabled", "Enabled")
        self._add_combo(
            envelope_page,
            envelope_form,
            controls,
            track_index,
            "env_mod_destination",
            "Destination",
            LFO_DESTINATIONS,
        )
        self._add_track_dial(env_grid, controls, track_index, "env_mod_amount", "Amount", -100, 100, 100, 0, 0, size=38)
        envelope = EnvelopeEditor(
            self.engine.tracks[track_index].env_mod_points,
            lambda points, tr=track_index: self._set_track_param_and_preview(tr, "env_mod_points", points),
        )
        self._set_tip(envelope, "envelope_editor")
        controls["envelopes"]["env_mod_points"] = envelope
        envelope_graph_layout.addWidget(envelope, 1)

        self._add_step_editor(step_form, controls, track_index)

        self.track_widgets.append(controls)

    def _form_layout(self, group: QGroupBox) -> QFormLayout:
        form = QFormLayout(group)
        form.setLabelAlignment(Qt.AlignRight)
        form.setFormAlignment(Qt.AlignTop)
        form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        form.setContentsMargins(8, 8, 8, 8)
        form.setHorizontalSpacing(8)
        form.setVerticalSpacing(7)
        return form

    def _scrollable_tab_page(self, content: QWidget) -> QScrollArea:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll.setWidget(content)
        return scroll

    def _build_lfo_modulation_page(self):
        content = QWidget()
        content.setMinimumHeight(180)
        layout = QHBoxLayout(content)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        source_group = QGroupBox("Source")
        source_group.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        source_form = self._form_layout(source_group)
        source_form.setHorizontalSpacing(12)
        source_form.setVerticalSpacing(10)

        depth_group = QGroupBox("Depth")
        depth_group.setMinimumWidth(112)
        depth_group.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        depth_layout = QVBoxLayout(depth_group)
        depth_layout.setContentsMargins(8, 10, 8, 8)
        depth_layout.setSpacing(4)
        depth_grid = QGridLayout()
        depth_grid.setHorizontalSpacing(6)
        depth_grid.setVerticalSpacing(4)
        depth_layout.addLayout(depth_grid)
        depth_layout.addStretch(1)

        layout.addWidget(source_group, 3, alignment=Qt.AlignTop)
        layout.addWidget(depth_group, alignment=Qt.AlignTop)
        return self._scrollable_tab_page(content), source_form, depth_grid

    def _build_envelope_modulation_page(self):
        content = QWidget()
        content.setMinimumHeight(220)
        content.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout = QVBoxLayout(content)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)
        layout.setAlignment(Qt.AlignTop)

        control_row = QHBoxLayout()
        control_row.setContentsMargins(0, 0, 0, 0)
        control_row.setSpacing(6)

        source_group = QGroupBox("Source")
        source_group.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        source_form = self._form_layout(source_group)
        source_form.setContentsMargins(6, 8, 6, 6)
        source_form.setHorizontalSpacing(8)
        source_form.setVerticalSpacing(4)
        control_row.addWidget(source_group, 2)

        depth_group = QGroupBox("Depth")
        depth_group.setMinimumWidth(112)
        depth_group.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        depth_layout = QVBoxLayout(depth_group)
        depth_layout.setContentsMargins(6, 8, 6, 6)
        depth_layout.setSpacing(2)
        depth_grid = QGridLayout()
        depth_grid.setHorizontalSpacing(4)
        depth_grid.setVerticalSpacing(2)
        depth_layout.addLayout(depth_grid)
        control_row.addWidget(depth_group)

        graph_group = QGroupBox("Envelope Shape")
        graph_group.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        graph_layout = QVBoxLayout(graph_group)
        graph_layout.setContentsMargins(6, 8, 6, 6)
        graph_layout.setSpacing(3)

        layout.addLayout(control_row)
        layout.addWidget(graph_group)
        return content, source_form, depth_grid, graph_layout

    def _add_global_dial(
        self,
        grid: QGridLayout,
        param: str,
        label: str,
        minimum: int,
        maximum: int,
        scale: int,
        row: int,
        column: int,
    ):
        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignCenter)
        layout.setSpacing(2)
        name = QLabel(label)
        name.setAlignment(Qt.AlignCenter)
        self._param_tip(name, param)
        dial = QDial()
        dial.setRange(minimum, maximum)
        dial.setNotchesVisible(True)
        dial.setFixedSize(46, 46)
        self._param_tip(dial, param)
        value_label = QLabel()
        value_label.setAlignment(Qt.AlignCenter)
        value_label.setFixedWidth(58)
        self._param_tip(value_label, param)
        layout.addWidget(name)
        layout.addWidget(dial)
        layout.addWidget(value_label)
        grid.addLayout(layout, row, column)

        def update(value):
            display = value / scale
            value_label.setText(self._format_global_value(param, display, value))
            self.engine.set_global_param(param, display)

        dial.valueChanged.connect(update)
        self.global_widgets["dials"][param] = (dial, value_label, scale)

    def _add_track_dial(
        self,
        grid: QGridLayout,
        controls: dict,
        track_index: int,
        param: str,
        label: str,
        minimum: int,
        maximum: int,
        scale: int,
        row: int,
        column: int,
        size: int = 74,
    ):
        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignCenter)
        layout.setSpacing(5)
        name = QLabel(label)
        name.setAlignment(Qt.AlignCenter)
        self._param_tip(name, param)
        dial = QDial()
        dial.setRange(minimum, maximum)
        dial.setNotchesVisible(True)
        dial.setFixedSize(size, size)
        self._param_tip(dial, param)
        value_label = QLabel()
        value_label.setAlignment(Qt.AlignCenter)
        value_label.setFixedWidth(max(58, size + 8))
        self._param_tip(value_label, param)
        layout.addWidget(name)
        layout.addWidget(dial)
        layout.addWidget(value_label)
        grid.addLayout(layout, row, column)

        def update(value):
            display = value / scale
            value_label.setText(self._format_value(param, display, value))
            self._set_track_param_and_preview(
                track_index, param, int(display) if param == "track_steps" else display
            )

        dial.valueChanged.connect(update)
        controls["dials"][param] = (dial, value_label, scale)

    def _add_combo(
        self,
        group: QGroupBox,
        form: QFormLayout,
        controls: dict,
        track_index: int,
        param: str,
        label: str,
        values: list[str],
    ):
        del group
        combo = QComboBox()
        combo.addItems(values)
        self._param_tip(combo, param)
        combo.currentTextChanged.connect(
            lambda text, tr=track_index, key=param: self._set_track_param_and_preview(tr, key, text)
        )
        controls["combos"][param] = combo
        form.addRow(label, combo)

    def _add_check(
        self,
        form: QFormLayout,
        controls: dict,
        track_index: int,
        param: str,
        label: str,
    ):
        button = QToolButton()
        button.setText("On" if label == "Enabled" else label)
        button.setCheckable(True)
        if label == "Enabled":
            button.setMinimumSize(58, 24)
            button.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self._param_tip(button, param)

        def update(checked):
            self.engine.set_track_param(track_index, param, checked, render_async=False)
            if param in {"lfo_enabled", "lfo2_enabled", "env_mod_enabled"} and checked:
                amount_param = {
                    "lfo_enabled": "lfo_amount",
                    "lfo2_enabled": "lfo2_amount",
                    "env_mod_enabled": "env_mod_amount",
                }[param]
                phase_param = {
                    "lfo_enabled": "lfo_phase",
                    "lfo2_enabled": "lfo2_phase",
                    "env_mod_enabled": None,
                }[param]
                with self.engine.lock:
                    amount = getattr(self.engine.tracks[track_index], amount_param)
                    phase = getattr(self.engine.tracks[track_index], phase_param) if phase_param else 0.25
                if abs(amount) < 0.001:
                    self.engine.set_track_param(track_index, amount_param, 0.5, render_async=False)
                    dial, value_label, scale = controls["dials"][amount_param]
                    dial.blockSignals(True)
                    dial.setValue(round(0.5 * scale))
                    value_label.setText(self._format_value(amount_param, 0.5, 50))
                    dial.blockSignals(False)
                if phase_param and phase == 0.0:
                    self.engine.set_track_param(track_index, phase_param, 0.25, render_async=False)
                    spin, scale = controls["spins"][phase_param]
                    spin.blockSignals(True)
                    spin.setValue(0.25 * scale)
                    spin.blockSignals(False)
            self._schedule_waveform_preview(track_index)
            self._schedule_render_cache_refresh()

        button.toggled.connect(update)
        controls["checks"][param] = button
        form.addRow(label, button)

    def _add_spin(
        self,
        form: QFormLayout,
        controls: dict,
        track_index: int,
        param: str,
        label: str,
        minimum: float,
        maximum: float,
        step: float,
        scale: int,
        suffix: str,
    ):
        spin = QDoubleSpinBox()
        spin.setRange(minimum, maximum)
        spin.setSingleStep(step)
        spin.setSuffix(suffix)
        spin.setDecimals(3 if step < 0.01 else 2 if step < 1 else 0)
        spin.setKeyboardTracking(False)
        self._param_tip(spin, param)
        spin.valueChanged.connect(
            lambda value, tr=track_index, key=param, factor=scale: self._set_track_param_and_preview(
                tr,
                key,
                int(value / factor)
                if key in {"bit_depth", "sample_rate_reduction"}
                else value / factor,
            )
        )
        controls["spins"][param] = (spin, scale)
        form.addRow(label, spin)

    def _add_step_editor(self, form: QFormLayout, controls: dict, track_index: int):
        step_spin = QSpinBox()
        step_spin.setRange(1, STEPS)
        step_spin.setValue(1)
        step_spin.setKeyboardTracking(False)
        self._set_tip(step_spin, "edit_step")
        step_spin.valueChanged.connect(
            lambda value, tr=track_index: self._sync_step_editor(tr, value - 1)
        )
        controls["step"]["selected"] = step_spin
        form.addRow("Edit Step", step_spin)

        velocity = self._step_spin(0, 150, "%")
        self._set_tip(velocity, "step_velocity")
        velocity.valueChanged.connect(
            lambda value, tr=track_index: self._set_step_param(tr, "velocities", value / 100.0)
        )
        controls["step"]["velocity"] = velocity
        form.addRow("Velocity", velocity)

        probability = self._step_spin(0, 100, "%")
        self._set_tip(probability, "step_probability")
        probability.valueChanged.connect(
            lambda value, tr=track_index: self._set_step_param(
                tr, "probabilities", value / 100.0
            )
        )
        controls["step"]["probability"] = probability
        form.addRow("Probability", probability)

        ratchet = QSpinBox()
        ratchet.setRange(1, 4)
        ratchet.setKeyboardTracking(False)
        self._set_tip(ratchet, "step_ratchet")
        ratchet.valueChanged.connect(
            lambda value, tr=track_index: self._set_step_param(tr, "ratchets", value)
        )
        controls["step"]["ratchet"] = ratchet
        form.addRow("Ratchet", ratchet)

        bass_note = QComboBox()
        for note in range(STEP_NOTE_MIN, STEP_NOTE_MAX + 1):
            bass_note.addItem(self._note_name(note), note)
        self._set_tip(bass_note, "step_bass_note")
        bass_note.currentIndexChanged.connect(
            lambda index, tr=track_index, combo=bass_note: self._set_step_param(
                tr,
                "bass_notes",
                combo.itemData(index),
            )
        )
        controls["step"]["bass_note"] = bass_note
        form.addRow("Step Note", bass_note)

    def _step_spin(self, minimum: int, maximum: int, suffix: str) -> QSpinBox:
        spin = QSpinBox()
        spin.setRange(minimum, maximum)
        spin.setSuffix(suffix)
        spin.setKeyboardTracking(False)
        return spin

    def _note_name(self, note: int) -> str:
        octave = note // 12 - 1
        return f"{NOTE_NAMES[note % 12]}{octave}"

    def _format_value(self, param: str, display: float, raw_value: int) -> str:
        if param == "pan":
            if abs(display) < 0.01:
                return "C"
            return f"{'R' if display > 0 else 'L'} {abs(display):.2f}"
        if param == "track_steps":
            return str(raw_value)
        if param in {"pitch", "kick_mute"}:
            return f"{raw_value}%"
        return f"{display:.2f}"

    def _format_global_value(self, param: str, display: float, raw_value: int) -> str:
        if param == "global_filter_cutoff":
            if raw_value >= 1000:
                return f"{raw_value / 1000:.1f}k"
            return f"{raw_value} Hz"
        if param in {"global_fx_amount", "global_density"}:
            return f"{raw_value}%"
        return f"{display:.2f}"

    def _apply_style(self):
        QApplication.instance().setStyle("Fusion")
        palette = QPalette()
        palette.setColor(QPalette.Window, QColor("#20242a"))
        palette.setColor(QPalette.WindowText, QColor("#f1f4f8"))
        palette.setColor(QPalette.Base, QColor("#15191f"))
        palette.setColor(QPalette.AlternateBase, QColor("#252b33"))
        palette.setColor(QPalette.Text, QColor("#f1f4f8"))
        palette.setColor(QPalette.Button, QColor("#2c333d"))
        palette.setColor(QPalette.ButtonText, QColor("#f1f4f8"))
        palette.setColor(QPalette.Highlight, QColor("#58c4dd"))
        QApplication.instance().setPalette(palette)

        self.setStyleSheet(
            """
            QGroupBox {
                border: 1px solid #3b4552;
                border-radius: 5px;
                margin-top: 7px;
                padding: 6px;
                font-weight: 600;
            }
            QGroupBox[soundHeader="true"] {
                background: #252c36;
                border: 1px solid #465363;
                border-radius: 6px;
                margin-top: 0;
                padding: 0;
            }
            QGroupBox[soundPanel="true"] {
                background: #20262e;
                border: 1px solid #343f4c;
                border-radius: 5px;
                margin-top: 6px;
                padding: 5px;
            }
            QGroupBox[compactGlobal="true"] {
                margin-top: 6px;
                padding: 5px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 8px;
                padding: 0 3px;
            }
            QLabel[soundTitle="true"] {
                color: #f2d16b;
                font-size: 15px;
                font-weight: 700;
            }
            QLabel[compactLabel="true"] {
                color: #b8c4d2;
                font-weight: 700;
            }
            QPushButton, QToolButton {
                border: 1px solid #4a5565;
                border-radius: 5px;
                padding: 4px 8px;
                background: #2d3540;
            }
            QPushButton:hover, QToolButton:hover {
                background: #37414f;
            }
            QPushButton:checked, QToolButton:checked {
                background: #58c4dd;
                border-color: #91deef;
                color: #071217;
                font-weight: 700;
            }
            QToolButton[sequencerMute="true"] {
                padding: 2px 0;
                font-weight: 700;
            }
            QToolButton[sequencerMute="true"]:checked {
                background: #d86a6a;
                border-color: #f2a0a0;
                color: #1b0b0b;
            }
            QTabWidget::pane {
                border: 1px solid #3b4552;
                border-radius: 5px;
                padding: 4px;
                top: -1px;
            }
            QTabBar::tab {
                background: #252c36;
                border: 1px solid #3b4552;
                border-bottom: none;
                border-top-left-radius: 5px;
                border-top-right-radius: 5px;
                padding: 5px 9px;
                margin-right: 2px;
            }
            QTabBar::tab:selected {
                background: #364250;
                color: #f2d16b;
            }
            QDoubleSpinBox, QSpinBox, QComboBox {
                min-height: 22px;
                padding: 2px 6px;
            }
            StepButton {
                padding: 0;
            }
            StepButton[playhead="true"] {
                border: 2px solid #f2d16b;
            }
            StepButton[accent="true"] {
                border-color: #6f7c8e;
            }
            StepButton[barStart="true"] {
                margin-left: 4px;
            }
            StepButton[selectedStep="true"] {
                border: 2px solid #f2d16b;
            }
            StepButton[trackFamily="low"]:checked {
                background: #5fbf91;
                color: #071217;
            }
            StepButton[trackFamily="snap"]:checked {
                background: #f2d16b;
                color: #16120a;
            }
            StepButton[trackFamily="metal"]:checked {
                background: #58c4dd;
                color: #071217;
            }
            StepButton[trackFamily="accent"]:checked {
                background: #d883ff;
                color: #120719;
            }
            QLabel[trackFamily="low"] {
                color: #9fe3bf;
            }
            QLabel[trackFamily="snap"] {
                color: #f2d16b;
            }
            QLabel[trackFamily="metal"] {
                color: #8fdbec;
            }
            QLabel[trackFamily="accent"] {
                color: #dfa7ff;
            }
            QLabel[barStart="true"] {
                color: #f2d16b;
                font-weight: 700;
            }
            QLabel[inspectorTitle="true"] {
                font-size: 15px;
                font-weight: 700;
                color: #f2d16b;
            }
            QLabel[trackBadge="true"], QLabel[morphSummary="true"] {
                color: #b8c4d2;
                padding: 3px 0;
            }
            QLabel[songHeader="true"] {
                color: #b8c4d2;
                font-weight: 700;
            }
            QLabel[songActive="true"] {
                background: #58c4dd;
                border-radius: 4px;
                color: #071217;
                font-weight: 700;
                padding: 4px;
            }
            QSlider::groove:horizontal {
                height: 6px;
                background: #15191f;
                border-radius: 3px;
            }
            QSlider::handle:horizontal {
                width: 16px;
                margin: -5px 0;
                border-radius: 8px;
                background: #f2d16b;
            }
            """
        )

    def _sync_from_engine(self):
        with self.engine.lock:
            self.bpm_spin.setValue(self.engine.bpm)
            self.swing_slider.setValue(round(self.engine.swing * 100))
            self.master_slider.setValue(round(self.engine.master_volume * 100))
            tracks = list(self.engine.tracks)
            global_state = self.engine.global_state()
            current_scene = self.engine.current_scene
            playing = self.engine.playing

        for param, (dial, value_label, scale) in self.global_widgets["dials"].items():
            value = global_state[param]
            raw = round(value * scale)
            dial.blockSignals(True)
            dial.setValue(raw)
            value_label.setText(self._format_global_value(param, value, raw))
            dial.blockSignals(False)

        for param, button in self.global_widgets["buttons"].items():
            button.blockSignals(True)
            button.setChecked(bool(global_state[param]))
            button.blockSignals(False)

        for index, button in enumerate(self.scene_buttons):
            button.blockSignals(True)
            button.setChecked(index == current_scene)
            button.blockSignals(False)
        self._sync_scene_name_controls()

        self.play_button.blockSignals(True)
        self.play_button.setChecked(playing)
        self.play_button.setText("Pause" if playing else "Play")
        self.play_button.blockSignals(False)
        self.last_synced_scene = current_scene
        self._sync_song_controls()
        with self.engine.lock:
            self.last_song_position = self.engine.song_position

        if self.pattern_track_combo is not None:
            self.pattern_track_combo.blockSignals(True)
            self.pattern_track_combo.setCurrentIndex(self.selected_pattern_track)
            self.pattern_track_combo.blockSignals(False)

        self._sync_sequencer_grid(tracks)

        for track_index, track in enumerate(tracks):
            widgets = self.track_widgets[track_index]
            widgets["instrument"].blockSignals(True)
            widgets["instrument"].setCurrentText(track.instrument)
            widgets["instrument"].blockSignals(False)

            for param, (dial, value_label, scale) in widgets["dials"].items():
                value = getattr(track, param)
                raw = round(value * scale)
                dial.blockSignals(True)
                dial.setValue(raw)
                value_label.setText(self._format_value(param, value, raw))
                dial.blockSignals(False)

            for param, (spin, scale) in widgets["spins"].items():
                spin.blockSignals(True)
                spin.setValue(getattr(track, param) * scale)
                spin.blockSignals(False)

            for param, combo in widgets["combos"].items():
                combo.blockSignals(True)
                combo.setCurrentText(getattr(track, param))
                combo.blockSignals(False)

            for param, check in widgets["checks"].items():
                check.blockSignals(True)
                check.setChecked(bool(getattr(track, param)))
                check.blockSignals(False)

            for param, envelope in widgets["envelopes"].items():
                envelope.set_points(getattr(track, param))

            widgets["mute"].blockSignals(True)
            widgets["mute"].setChecked(track.muted)
            widgets["mute"].blockSignals(False)

            widgets["solo"].blockSignals(True)
            widgets["solo"].setChecked(track.solo)
            widgets["solo"].blockSignals(False)

            badges = self._track_badges(track)
            self.track_labels[track_index].setText(
                f"{track.instrument} {' '.join(badges)}" if badges else track.instrument
            )
            widgets["header_title"].setText(track.instrument)
            widgets["header_badges"].setText(" ".join(badges) if badges else "Clean")
            self.track_tabs.setTabText(track_index, f"{track_index + 1}: {track.instrument}")
            self._sync_step_editor(track_index, widgets["step"]["selected"].value() - 1)
            self._update_waveform_preview(track_index)
        self._sync_step_inspector()

    def _sync_sequencer_grid(self, tracks):
        for track_index, track in enumerate(tracks):
            for step, enabled in enumerate(track.pattern):
                button = self.step_buttons[track_index][step]
                button.blockSignals(True)
                button.setChecked(enabled)
                button.setProperty("accent", step % 4 == 0)
                button.setProperty(
                    "selectedStep",
                    track_index == self.selected_pattern_track and step == self.selected_step,
                )
                button.setText(
                    self._note_name(track.bass_notes[step])
                    if enabled
                    else str(step + 1)
                )
                button.style().unpolish(button)
                button.style().polish(button)
                button.blockSignals(False)

            self._sync_track_mute_state(track_index, track)

    def _sync_live_scene_from_engine(self):
        with self.engine.lock:
            current_scene = self.engine.current_scene
            playing = self.engine.playing
            tracks = copy.deepcopy(self.engine.tracks)

        for index, button in enumerate(self.scene_buttons):
            button.blockSignals(True)
            button.setChecked(index == current_scene)
            button.blockSignals(False)
        self._sync_scene_name_controls()
        self._sync_sequencer_grid(tracks)
        self._sync_step_inspector()

        self.play_button.blockSignals(True)
        self.play_button.setChecked(playing)
        self.play_button.setText("Pause" if playing else "Play")
        self.play_button.blockSignals(False)
        self.last_synced_scene = current_scene

        if self.pattern_track_combo is not None:
            self.pattern_track_combo.blockSignals(True)
            self.pattern_track_combo.setCurrentIndex(self.selected_pattern_track)
            self.pattern_track_combo.blockSignals(False)

    def _sync_track_sound_controls(self, track_indices):
        with self.engine.lock:
            tracks = {
                track_index: copy.deepcopy(self.engine.tracks[track_index])
                for track_index in track_indices
                if 0 <= track_index < len(self.engine.tracks)
            }

        for track_index, track in tracks.items():
            widgets = self.track_widgets[track_index]
            widgets["instrument"].blockSignals(True)
            widgets["instrument"].setCurrentText(track.instrument)
            widgets["instrument"].blockSignals(False)

            for param, (dial, value_label, scale) in widgets["dials"].items():
                value = getattr(track, param)
                raw = round(value * scale)
                dial.blockSignals(True)
                dial.setValue(raw)
                value_label.setText(self._format_value(param, value, raw))
                dial.blockSignals(False)

            for param, (spin, scale) in widgets["spins"].items():
                spin.blockSignals(True)
                spin.setValue(getattr(track, param) * scale)
                spin.blockSignals(False)

            for param, combo in widgets["combos"].items():
                combo.blockSignals(True)
                combo.setCurrentText(getattr(track, param))
                combo.blockSignals(False)

            for param, check in widgets["checks"].items():
                check.blockSignals(True)
                check.setChecked(bool(getattr(track, param)))
                check.blockSignals(False)

            for param, envelope in widgets["envelopes"].items():
                envelope.set_points(getattr(track, param))

            widgets["mute"].blockSignals(True)
            widgets["mute"].setChecked(track.muted)
            widgets["mute"].blockSignals(False)

            widgets["solo"].blockSignals(True)
            widgets["solo"].setChecked(track.solo)
            widgets["solo"].blockSignals(False)

            badges = self._track_badges(track)
            self.track_labels[track_index].setText(
                f"{track.instrument} {' '.join(badges)}" if badges else track.instrument
            )
            widgets["header_title"].setText(track.instrument)
            widgets["header_badges"].setText(" ".join(badges) if badges else "Clean")
            self.track_tabs.setTabText(track_index, f"{track_index + 1}: {track.instrument}")
            self._sync_step_editor(track_index, widgets["step"]["selected"].value() - 1)
            self._schedule_waveform_preview(track_index)

        self._sync_step_inspector()

    def _update_waveform_preview(self, track_index: int):
        if track_index >= len(self.track_widgets):
            return
        preview = self.track_widgets[track_index].get("waveform")
        level_profile = self.track_widgets[track_index].get("level_profile")
        spectrum = self.track_widgets[track_index].get("spectrum")
        if preview is None and level_profile is None and spectrum is None:
            return
        with self.engine.lock:
            track = copy.deepcopy(self.engine.tracks[track_index])
        with self.engine.render_random_lock:
            random_state = np.random.get_state()
            np.random.seed(10_000 + track_index)
            try:
                audio = make_hit(track)
            finally:
                np.random.set_state(random_state)
        if preview is not None:
            preview.set_audio(audio)
        if level_profile is not None:
            level_profile.set_audio(audio)
        if spectrum is not None:
            spectrum.set_audio(audio)

    def _schedule_waveform_preview(self, track_index: int):
        if track_index >= len(self.track_widgets):
            return
        timer = self.preview_update_timers.get(track_index)
        if timer is None:
            timer = QTimer(self)
            timer.setSingleShot(True)
            timer.timeout.connect(lambda tr=track_index: self._update_waveform_preview(tr))
            self.preview_update_timers[track_index] = timer
        timer.start(180)

    def _schedule_render_cache_refresh(self):
        with self.engine.lock:
            playing = self.engine.playing
        if playing:
            self.render_cache_debounce_timer.start(180)

    def _flush_render_cache_refresh(self):
        with self.engine.lock:
            playing = self.engine.playing
        if playing:
            self.engine.prepare_render_cache_async()

    def _set_track_param_and_preview(self, track_index: int, param: str, value):
        self.engine.set_track_param(track_index, param, value, render_async=False)
        self._schedule_waveform_preview(track_index)
        self._schedule_render_cache_refresh()

    def _set_track_mute(self, track_index: int, muted: bool):
        self.engine.set_track_param(track_index, "muted", muted)
        self._sync_track_mute_state(track_index)
        self._sync_step_inspector()

    def _sync_track_mute_state(self, track_index: int, track=None):
        if track is None:
            with self.engine.lock:
                track = copy.copy(self.engine.tracks[track_index])

        if track_index < len(self.track_mute_buttons):
            button = self.track_mute_buttons[track_index]
            button.blockSignals(True)
            button.setChecked(bool(track.muted))
            button.blockSignals(False)

        if track_index < len(self.track_widgets):
            mute = self.track_widgets[track_index].get("mute")
            if mute is not None:
                mute.blockSignals(True)
                mute.setChecked(bool(track.muted))
                mute.blockSignals(False)

        if track_index < len(self.track_labels):
            badges = self._track_badges(track)
            self.track_labels[track_index].setText(
                f"{track.instrument} {' '.join(badges)}" if badges else track.instrument
            )

    def _sync_step_editor(self, track_index: int, step: int):
        widgets = self.track_widgets[track_index]["step"]
        with self.engine.lock:
            track = self.engine.tracks[track_index]
            step = max(0, min(STEPS - 1, step))
            velocity = round(track.velocities[step] * 100)
            probability = round(track.probabilities[step] * 100)
            ratchet = track.ratchets[step]
            bass_note = int(track.bass_notes[step])

        for key, value in (
            ("velocity", velocity),
            ("probability", probability),
            ("ratchet", ratchet),
        ):
            widget = widgets[key]
            widget.blockSignals(True)
            widget.setValue(value)
            widget.blockSignals(False)

        note_combo = widgets["bass_note"]
        note_combo.blockSignals(True)
        note_combo.setCurrentIndex(max(0, min(STEP_NOTE_MAX - STEP_NOTE_MIN, bass_note - STEP_NOTE_MIN)))
        note_combo.blockSignals(False)

    def _set_step_param(self, track_index: int, param: str, value):
        step = self.track_widgets[track_index]["step"]["selected"].value() - 1
        self.engine.set_step_param(track_index, step, param, value)
        if track_index == self.selected_pattern_track and step == self.selected_step:
            self._sync_step_inspector()
        if param == "bass_notes":
            self._sync_from_engine()

    def _select_step(self, track_index: int, step: int):
        self.selected_pattern_track = track_index
        self.selected_step = step
        if self.pattern_track_combo is not None:
            self.pattern_track_combo.blockSignals(True)
            self.pattern_track_combo.setCurrentIndex(track_index)
            self.pattern_track_combo.blockSignals(False)
        selector = self.track_widgets[track_index]["step"]["selected"]
        selector.setValue(step + 1)
        self._sync_step_editor(track_index, step)
        self._sync_step_inspector()
        self._update_step_selection_styles()

    def _set_inspector_step_active(self, checked: bool):
        self.engine.set_step(self.selected_pattern_track, self.selected_step, checked)
        button = self.step_buttons[self.selected_pattern_track][self.selected_step]
        button.blockSignals(True)
        button.setChecked(checked)
        button.blockSignals(False)
        self._sync_step_inspector()
        self._sync_step_button_text(self.selected_pattern_track, self.selected_step)

    def _set_inspector_step_param(self, param: str, value):
        self.engine.set_step_param(self.selected_pattern_track, self.selected_step, param, value)
        self._sync_step_editor(self.selected_pattern_track, self.selected_step)
        if param == "bass_notes":
            self._sync_step_button_text(self.selected_pattern_track, self.selected_step)
            self._sync_from_engine()
        else:
            self._sync_step_inspector()

    def _set_inspector_track_param(self, param: str, value):
        if param == "muted":
            self._set_track_mute(self.selected_pattern_track, value)
            return
        self.engine.set_track_param(self.selected_pattern_track, param, value)
        self._sync_from_engine()

    def _jump_to_selected_sound(self):
        self.track_tabs.setCurrentIndex(self.selected_pattern_track)
        self.main_tabs.setCurrentIndex(3)

    def _sync_step_inspector(self):
        if not self.inspector_widgets:
            return
        with self.engine.lock:
            track = self.engine.tracks[self.selected_pattern_track]
            step = max(0, min(STEPS - 1, self.selected_step))
            active = bool(track.pattern[step])
            velocity = round(track.velocities[step] * 100)
            probability = round(track.probabilities[step] * 100)
            ratchet = int(track.ratchets[step])
            bass_note = int(track.bass_notes[step])
            muted = bool(track.muted)
            solo = bool(track.solo)
            title = f"{track.instrument} / Step {step + 1}"
            badge = self._track_badges(track)

        self.inspector_widgets["title"].setText(title)
        self.inspector_widgets["badge"].setText(" ".join(badge) if badge else "No active modifiers")
        for key, value in (
            ("active", active),
            ("mute", muted),
            ("solo", solo),
        ):
            widget = self.inspector_widgets[key]
            widget.blockSignals(True)
            widget.setChecked(value)
            widget.blockSignals(False)
        for key, value in (
            ("velocity", velocity),
            ("probability", probability),
            ("ratchet", ratchet),
        ):
            widget = self.inspector_widgets[key]
            widget.blockSignals(True)
            widget.setValue(value)
            widget.blockSignals(False)
        note_combo = self.inspector_widgets["bass_note"]
        note_combo.blockSignals(True)
        note_combo.setCurrentIndex(max(0, min(STEP_NOTE_MAX - STEP_NOTE_MIN, bass_note - STEP_NOTE_MIN)))
        note_combo.setEnabled(True)
        note_combo.blockSignals(False)

    def _track_badges(self, track) -> list[str]:
        badges = []
        if track.bass_enabled:
            badges.append("NOTE")
        if track.delay_send > 0.01 or track.reverb_send > 0.01:
            badges.append("FX")
        if track.lfo_enabled and abs(track.lfo_amount) > 0.001:
            badges.append("LFO1")
        if track.lfo2_enabled and abs(track.lfo2_amount) > 0.001:
            badges.append("LFO2")
        if track.env_mod_enabled and abs(track.env_mod_amount) > 0.001:
            badges.append("ENV")
        if track.muted:
            badges.append("MUTE")
        if track.solo:
            badges.append("SOLO")
        return badges

    def _sync_step_button_text(self, track_index: int, step: int):
        with self.engine.lock:
            track = self.engine.tracks[track_index]
            enabled = bool(track.pattern[step])
            show_note = enabled
            text = self._note_name(track.bass_notes[step]) if show_note else str(step + 1)
        self.step_buttons[track_index][step].setText(text)

    def _update_step_selection_styles(self):
        for track_index, row in enumerate(self.step_buttons):
            for button in row:
                selected = track_index == self.selected_pattern_track and button.step == self.selected_step
                button.setProperty("selectedStep", selected)
                button.style().unpolish(button)
                button.style().polish(button)

    def _clear_layout(self, layout):
        while layout.count():
            item = layout.takeAt(0)
            child_layout = item.layout()
            widget = item.widget()
            if child_layout is not None:
                self._clear_layout(child_layout)
            if widget is not None:
                widget.deleteLater()

    def _sync_song_controls(self):
        if self.song_rows_layout is None:
            return

        with self.engine.lock:
            song = self.engine.song_state()
            playing = self.engine.playing
            scene_names = self.engine.scene_labels()

        self._clear_layout(self.song_rows_layout)
        self.song_row_labels = []
        for row in range(70):
            self.song_rows_layout.setRowStretch(row, 0)

        headers = ("Slot", "Scene", "Bars", "")
        for column, text in enumerate(headers):
            label = QLabel(text)
            label.setProperty("songHeader", True)
            label.setToolTip(
                {
                    "Slot": "Song slot number. The highlighted slot is currently playing.",
                    "Scene": "Pattern scene played by this song slot.",
                    "Bars": "Number of bars this song slot plays before advancing.",
                    "": "Move or remove song slots.",
                }[text]
            )
            self.song_rows_layout.addWidget(label, 0, column)

        for row, slot in enumerate(song["chain"], start=1):
            slot_index = row - 1
            slot_label = QLabel(str(row))
            slot_label.setAlignment(Qt.AlignCenter)
            slot_label.setMinimumWidth(34)
            is_active = song["playing"] and slot_index == song["position"]
            slot_label.setProperty("songActive", is_active)
            slot_label.setToolTip(f"Song slot {row}. Highlighted while this slot is playing.")
            self.song_rows_layout.addWidget(slot_label, row, 0)
            self.song_row_labels.append(slot_label)

            scene_combo = QComboBox()
            scene_combo.addItems([f"{index + 1}: {scene_names[index]}" for index in range(PATTERN_SCENES)])
            scene_combo.setCurrentIndex(slot["scene"])
            self._set_tip(scene_combo, "song_scene")
            scene_combo.currentIndexChanged.connect(
                lambda scene, index=slot_index: self._set_song_slot_scene(index, scene)
            )
            self.song_rows_layout.addWidget(scene_combo, row, 1)

            bars = QSpinBox()
            bars.setRange(1, 16)
            bars.setKeyboardTracking(False)
            bars.setValue(slot["bars"])
            self._set_tip(bars, "song_bars")
            bars.valueChanged.connect(
                lambda value, index=slot_index: self._set_song_slot_bars(index, value)
            )
            self.song_rows_layout.addWidget(bars, row, 2)

            buttons = QHBoxLayout()
            up = QToolButton()
            up.setText("Up")
            self._set_tip(up, "song_up")
            up.clicked.connect(lambda checked=False, index=slot_index: self._move_song_slot(index, -1))
            buttons.addWidget(up)

            down = QToolButton()
            down.setText("Down")
            self._set_tip(down, "song_down")
            down.clicked.connect(lambda checked=False, index=slot_index: self._move_song_slot(index, 1))
            buttons.addWidget(down)

            remove = QToolButton()
            remove.setText("Remove")
            self._set_tip(remove, "song_remove")
            remove.clicked.connect(lambda checked=False, index=slot_index: self._remove_song_slot(index))
            buttons.addWidget(remove)
            buttons.addStretch(1)
            self.song_rows_layout.addLayout(buttons, row, 3)

        if self.song_loop_button is not None:
            self.song_loop_button.blockSignals(True)
            self.song_loop_button.setChecked(song["loop"])
            self.song_loop_button.blockSignals(False)

        if self.song_play_button is not None:
            self.song_play_button.blockSignals(True)
            self.song_play_button.setChecked(song["playing"] and playing)
            self.song_play_button.setText("Pause Song" if song["playing"] and playing else "Play Song")
            self.song_play_button.blockSignals(False)

        if self.song_position_label is not None:
            position = song["position"] + 1
            bars = max(1, song["chain"][song["position"]]["bars"])
            progress = min(song["bar_progress"] + 1, bars)
            self.song_position_label.setText(f"Slot {position}, bar {progress}/{bars}")

        for column in range(4):
            self.song_rows_layout.setColumnStretch(column, 1 if column in {1, 3} else 0)
        self.song_rows_layout.setRowStretch(len(song["chain"]) + 1, 1)

    def _sync_song_playback_status(self):
        with self.engine.lock:
            song = self.engine.song_state()
            playing = self.engine.playing

        for index, label in enumerate(self.song_row_labels):
            is_active = song["playing"] and index == song["position"]
            if label.property("songActive") != is_active:
                label.setProperty("songActive", is_active)
                label.style().unpolish(label)
                label.style().polish(label)

        if self.song_loop_button is not None:
            self.song_loop_button.blockSignals(True)
            self.song_loop_button.setChecked(song["loop"])
            self.song_loop_button.blockSignals(False)

        if self.song_play_button is not None:
            self.song_play_button.blockSignals(True)
            self.song_play_button.setChecked(song["playing"] and playing)
            self.song_play_button.setText("Pause Song" if song["playing"] and playing else "Play Song")
            self.song_play_button.blockSignals(False)

        if self.song_position_label is not None and song["chain"]:
            position = song["position"] + 1
            bars = max(1, song["chain"][song["position"]]["bars"])
            progress = min(song["bar_progress"] + 1, bars)
            self.song_position_label.setText(f"Slot {position}, bar {progress}/{bars}")

    def _add_song_slot(self):
        self.engine.add_song_slot()
        self._sync_song_controls()

    def _remove_song_slot(self, index: int):
        self.engine.remove_song_slot(index)
        self._sync_song_controls()

    def _move_song_slot(self, index: int, amount: int):
        self.engine.move_song_slot(index, amount)
        self._sync_song_controls()

    def _set_song_slot_scene(self, index: int, scene: int):
        self.engine.set_song_slot(index, scene=scene)

    def _set_song_slot_bars(self, index: int, bars: int):
        self.engine.set_song_slot(index, bars=bars)

    def _generate_session(self):
        self.engine.generate_session(
            self.generator_patterns.value(),
            self.generator_bars.value(),
            self.generator_style.currentText(),
            self.generator_complexity.value() / 100.0,
            self.generator_fills.value() / 100.0,
            self.generator_variation.value() / 100.0,
        )
        self._sync_from_engine()
        self.main_tabs.setCurrentIndex(1)

    def _apply_pattern_pack(self, apply_kit: bool = False):
        if self.pattern_pack_combo is None:
            return
        if self.engine.apply_pattern_pack(self.pattern_pack_combo.currentText(), apply_kit=apply_kit):
            self._sync_from_engine()

    def _toggle_song_playback(self, checked: bool):
        if checked:
            self.engine.start_song()
            self.engine.prepare_render_cache_async()
        else:
            self.engine.stop_song()
            self.engine.set_playing(False)
        self._sync_from_engine()

    def _set_pattern_tool_track(self, track_index: int):
        self.selected_pattern_track = max(0, min(len(INSTRUMENTS) - 1, track_index))
        self._sync_step_inspector()
        self._update_step_selection_styles()

    def _select_scene(self, scene_index: int):
        self.engine.select_scene(scene_index)
        with self.engine.lock:
            playing = self.engine.playing
        if playing:
            self.engine.prepare_render_cache_async()
        self._sync_from_engine()

    def _store_scene(self):
        self.engine.store_scene()
        self._sync_scene_buttons()

    def _copy_scene(self):
        self.engine.copy_scene()

    def _paste_scene(self):
        if self.engine.paste_scene():
            self._sync_from_engine()

    def _clear_scene(self):
        self.engine.clear_scene()
        self._sync_from_engine()

    def _clear_all_patterns(self):
        reply = QMessageBox.question(
            self,
            "Clear all patterns?",
            "Clear every pattern scene? Track sounds and scene names will be kept.",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return
        self.engine.clear_all_patterns()
        self._sync_from_engine()

    def _copy_track_pattern(self):
        self.engine.copy_track_pattern(self.selected_pattern_track)

    def _paste_track_pattern(self):
        if self.engine.paste_track_pattern(self.selected_pattern_track):
            self._sync_from_engine()

    def _clear_track_pattern(self):
        self.engine.clear_track_pattern(self.selected_pattern_track)
        self._sync_from_engine()

    def _rotate_track_pattern(self, amount: int):
        self.engine.rotate_track_pattern(self.selected_pattern_track, amount)
        self._sync_from_engine()

    def _apply_pattern_morph(self):
        self.engine.morph_scenes(
            self.morph_source_combo.currentIndex(),
            self.morph_target_combo.currentIndex(),
            self.morph_amount_slider.value() / 100.0,
            self.morph_destination_combo.currentIndex(),
        )
        with self.engine.lock:
            playing = self.engine.playing
        if playing:
            self.engine.prepare_render_cache_async()
        self._sync_from_engine()

    def _set_morph_amount_label(self, value: int):
        self.morph_amount_label.setText(f"{value}%")
        self._sync_morph_summary()

    def _sync_morph_summary(self, *args):
        del args
        if self.morph_summary_label is None:
            return
        source = self.morph_source_combo.currentText()
        target = self.morph_target_combo.currentText()
        destination = self.morph_destination_combo.currentText()
        amount = self.morph_amount_slider.value()
        self.morph_summary_label.setText(f"{source} -> {target} at {amount}% writes to {destination}")

    def _rename_current_scene(self):
        if self.scene_name_edit is None:
            return
        with self.engine.lock:
            current_scene = self.engine.current_scene
        self.engine.set_scene_name(current_scene, self.scene_name_edit.text())
        self._sync_scene_name_controls()

    def _sync_scene_name_controls(self):
        with self.engine.lock:
            names = self.engine.scene_labels()
            current_scene = self.engine.current_scene

        if self.scene_name_edit is not None:
            self.scene_name_edit.blockSignals(True)
            self.scene_name_edit.setText(names[current_scene])
            self.scene_name_edit.blockSignals(False)

        labels = [f"{index + 1}: {name}" for index, name in enumerate(names)]
        for combo in self.morph_scene_combos:
            index = combo.currentIndex()
            combo.blockSignals(True)
            combo.clear()
            combo.addItems(labels)
            combo.setCurrentIndex(max(0, min(PATTERN_SCENES - 1, index)))
            combo.blockSignals(False)

        for index, button in enumerate(self.scene_buttons):
            button.setText(str(index + 1))
            button.setToolTip(
                f"{CONTROL_TOOLTIPS['scene_button']} This is scene {index + 1}: {names[index]}."
            )
        self._sync_morph_summary()

    def _sync_scene_buttons(self):
        with self.engine.lock:
            current_scene = self.engine.current_scene
        for index, button in enumerate(self.scene_buttons):
            button.blockSignals(True)
            button.setChecked(index == current_scene)
            button.blockSignals(False)

    def _apply_track_preset(self, track_index: int, instrument: str):
        self.engine.apply_track_preset(track_index, instrument)
        self._sync_track_sound_controls([track_index])
        with self.engine.lock:
            playing = self.engine.playing
        if not playing:
            self.engine.audition_track(track_index)

    def _apply_kit_pack(self, track_index: int, pack_name: str):
        if self.engine.apply_kit_pack_to_track(track_index, pack_name):
            self._sync_track_sound_controls([track_index])
            if track_index == self.selected_pattern_track:
                self._sync_step_inspector()
            with self.engine.lock:
                playing = self.engine.playing
            if not playing:
                self.engine.audition_track(track_index)

    def _preset_directory(self) -> Path:
        directory = Path(__file__).resolve().parent.parent / "presets"
        directory.mkdir(exist_ok=True)
        return directory

    def _preset_filename(self, track_index: int) -> str:
        with self.engine.lock:
            track = self.engine.tracks[track_index]
            name = f"{track_index + 1}_{track.instrument}_sound"
        return "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in name) + ".json"

    def _save_track_preset(self, track_index: int):
        default_path = self._preset_directory() / self._preset_filename(track_index)
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save track sound preset",
            str(default_path),
            "Track sound presets (*.json);;All files (*.*)",
        )
        if not path:
            return

        data = {
            "version": 1,
            "type": "DSynth Track Sound Preset",
            "track_preset": self.engine.track_sound_preset_state(track_index),
        }
        try:
            with open(path, "w", encoding="utf-8") as handle:
                json.dump(data, handle, indent=2)
        except OSError as exc:
            QMessageBox.warning(self, "Save failed", str(exc))

    def _load_track_preset(self, track_index: int):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Load track sound preset",
            str(self._preset_directory()),
            "Track sound presets (*.json);;All files (*.*)",
        )
        if not path:
            return
        try:
            with open(path, "r", encoding="utf-8") as handle:
                data = json.load(handle)
        except (OSError, json.JSONDecodeError) as exc:
            QMessageBox.warning(self, "Load failed", str(exc))
            return

        preset = data.get("track_preset", data) if isinstance(data, dict) else None
        if not isinstance(preset, dict):
            QMessageBox.warning(self, "Load failed", "This file is not a DSynth track sound preset.")
            return

        self.engine.load_track_sound_preset(track_index, preset)
        self._sync_from_engine()
        self.engine.audition_track(track_index)

    def _toggle_playback(self, checked: bool):
        self.play_button.setText("Pause" if checked else "Play")
        if checked:
            self.engine.prepare_render_cache()
        self.engine.set_playing(checked)
        self._sync_song_controls()

    def _stop(self):
        self.play_button.setChecked(False)
        self.play_button.setText("Play")
        with self.engine.lock:
            self.engine.current_step = 0
            self.engine.samples_until_step = 0
        self.engine.set_playing(False)
        self._refresh_playhead(force=True)

    def _clear(self):
        self.engine.clear_pattern()
        self._sync_from_engine()

    def _load_default(self):
        self.engine.load_default_pattern()
        self._sync_from_engine()

    def _randomize(self):
        self.engine.randomize_pattern()
        with self.engine.lock:
            playing = self.engine.playing
        if playing:
            self.engine.prepare_render_cache_async()
        self._sync_from_engine()

    def _save_patch(self):
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save pattern and sounds",
            "drum_machine_patch.json",
            "Patch files (*.json);;All files (*.*)",
        )
        if not path:
            return
        with self.engine.lock:
            data = {
                "version": 6,
                "bpm": self.engine.bpm,
                "swing": self.engine.swing,
                "master_volume": self.engine.master_volume,
                "global": self.engine.global_state(),
                "pattern_bank": self.engine.pattern_bank_state(),
                "song": self.engine.song_state(),
                "tracks": [track.to_dict() for track in self.engine.tracks],
            }
        try:
            with open(path, "w", encoding="utf-8") as handle:
                json.dump(data, handle, indent=2)
        except OSError as exc:
            QMessageBox.warning(self, "Save failed", str(exc))

    def _export_current_pattern(self):
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Export current pattern",
            "dsynth_pattern.wav",
            "WAV files (*.wav);;All files (*.*)",
        )
        if not path:
            return
        self._render_and_write_wav(path, lambda: self.engine.render_current_pattern(bars=4), "Pattern export")

    def _export_song(self):
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Song Mode arrangement",
            "dsynth_song.wav",
            "WAV files (*.wav);;All files (*.*)",
        )
        if not path:
            return
        self._render_and_write_wav(path, self.engine.render_song_arrangement, "Song export")

    def _render_and_write_wav(self, path: str, render_func, title: str):
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            audio = render_func()
            self._write_wav(path, audio)
        except Exception as exc:
            QMessageBox.warning(self, f"{title} failed", str(exc))
            return
        finally:
            QApplication.restoreOverrideCursor()

        duration = len(audio) / SAMPLE_RATE
        QMessageBox.information(
            self,
            f"{title} complete",
            f"Saved {Path(path).name}\n\nLength: {duration:.1f} seconds",
        )

    def _write_wav(self, path: str, audio: np.ndarray):
        audio = np.asarray(audio, dtype=np.float32)
        if audio.ndim == 1:
            audio = np.column_stack((audio, audio))
        if audio.shape[1] != 2:
            raise ValueError("Export audio must be stereo.")
        peak = float(np.max(np.abs(audio))) if len(audio) else 0.0
        if peak > 1.0:
            audio = audio / peak * 0.98
        pcm = np.clip(audio, -1.0, 1.0)
        pcm = (pcm * 32767.0).astype("<i2", copy=False)
        with wave.open(path, "wb") as handle:
            handle.setnchannels(2)
            handle.setsampwidth(2)
            handle.setframerate(SAMPLE_RATE)
            handle.writeframes(pcm.tobytes())

    def _load_patch(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Load pattern and sounds",
            "",
            "Patch files (*.json);;Pattern files (*.txt);;All files (*.*)",
        )
        if not path:
            return
        try:
            with open(path, "r", encoding="utf-8") as handle:
                text = handle.read()
        except OSError as exc:
            QMessageBox.warning(self, "Load failed", str(exc))
            return

        with self.engine.lock:
            try:
                data = json.loads(text)
            except json.JSONDecodeError:
                rows = [line.strip().split(":", 1)[-1] for line in text.splitlines() if line.strip()]
                for track, row in zip(self.engine.tracks, rows):
                    values = [char == "1" for char in row[:STEPS]]
                    if len(values) == STEPS:
                        track.pattern = values
            else:
                self.engine.bpm = float(data.get("bpm", self.engine.bpm))
                self.engine.swing = float(data.get("swing", self.engine.swing))
                self.engine.master_volume = float(
                    data.get("master_volume", self.engine.master_volume)
                )
                self.engine.update_global_state(data.get("global", {}))
                for track, saved in zip(self.engine.tracks, data.get("tracks", [])):
                    track.update_from_dict(saved)
                if "pattern_bank" in data:
                    self.engine.update_pattern_bank_state(data["pattern_bank"])
                else:
                    self.engine.reset_pattern_bank_from_current()
                self.engine.update_song_state(data.get("song", {}))
                self.engine.render_cache.clear()
        self._sync_from_engine()

    def _refresh_playhead(self, force: bool = False):
        if hasattr(self, "output_meter"):
            self.output_meter.set_bands(self.engine.spectrum_levels())
        if (
            self.channel_analyzers
            and hasattr(self, "main_tabs")
            and self.main_tabs.currentIndex() == 2
        ):
            channel_bands, total_levels = self.engine.spectrum_channel_db_levels()
            for index, analyzer in enumerate(self.channel_analyzers):
                analyzer.set_spectrum(channel_bands[index], float(total_levels[index]))

        with self.engine.lock:
            visible_step = (self.engine.current_step - 1) % STEPS if self.engine.playing else 0
            current_scene = self.engine.current_scene
            song_position = self.engine.song_position
            song_playing = self.engine.song_playing
            playing = self.engine.playing

        if current_scene != self.last_synced_scene:
            if playing and song_playing:
                self._sync_live_scene_from_engine()
                self._sync_song_playback_status()
            else:
                self._sync_from_engine()
        elif song_position != self.last_song_position or force:
            if playing and song_playing:
                self._sync_song_playback_status()
            else:
                self._sync_song_controls()
        self.last_song_position = song_position

        if self.play_button.isChecked() != playing:
            self.play_button.blockSignals(True)
            self.play_button.setChecked(playing)
            self.play_button.setText("Pause" if playing else "Play")
            self.play_button.blockSignals(False)
        if self.song_play_button is not None and self.song_play_button.isChecked() != (song_playing and playing):
            self.song_play_button.blockSignals(True)
            self.song_play_button.setChecked(song_playing and playing)
            self.song_play_button.setText("Pause Song" if song_playing and playing else "Play Song")
            self.song_play_button.blockSignals(False)

        if not force and visible_step == self.last_highlighted_step:
            return

        for row in self.step_buttons:
            for button in row:
                active = button.step == visible_step
                button.setProperty("playhead", active)
                button.style().unpolish(button)
                button.style().polish(button)

        self.last_highlighted_step = visible_step


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    return app.exec()
