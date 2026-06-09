# DSynth

DSynth is a PySide6 desktop drum synthesizer and 8-track step sequencer. It uses NumPy for synthesis and sounddevice for realtime audio output.

## Features

- 8 drum tracks with 16-step sequencing
- Extra Crash cymbal sound preset loadable onto any track
- 8 pattern scenes per project
- Pattern Morph tool for blending one scene toward another
- Named scenes for clearer arranging, song slots, and morph routing
- Sequencer Step Inspector for editing the selected step without leaving the Sequencer tab
- Step note labels on active steps when Note Mode is enabled
- Song mode for chaining pattern scenes into longer arrangements
- Static session generator for creating multiple pattern scenes and a Song Mode chain
- Random generated startup session chosen from the generator styles
- Per-track sound design tabs with nested editor tabs
- Per-step velocity, probability, and ratchet controls
- Note Mode for tuned per-step playback on any track
- Pattern tools for copy, paste, clear, and rotate
- Global performance controls for filter, drive, compression, density, humanize, FX amount, and fill
- Compact output spectrum display in the top toolbar
- Per-patch delay and reverb
- Per-track modulation with two LFOs and a draggable envelope editor
- Per-track sound preset save/load for reusing individual drum patches
- Tooltips on transport, sequencer, song, global, patch, modulation, and step-expression controls
- Patch save/load as JSON
- WAV export for the current pattern loop or full Song Mode arrangement

## Requirements

- Python 3.11 or newer
- PySide6
- NumPy
- sounddevice

The project currently includes a local virtual environment named `Template.venv`.

## Running

From the project folder:

```powershell
.\Template.venv\Scripts\python.exe DSynth.py
```

## Project Layout

```text
DSynth.py                App launcher
drum_machine/config.py   Constants, presets, menu values
drum_machine/model.py    Track patch and sequencer data model
drum_machine/synth.py    Drum voice construction and per-hit rendering
drum_machine/effects.py  Filters, saturation, bitcrush, patch FX, modulation helpers, bus compressor
drum_machine/engine.py   Audio engine, sequencer, pattern scenes
drum_machine/main_window.py PySide6 UI
presets/                Default folder for saved track sound presets
```

## Basic Use

1. Press `Play`.
2. DSynth starts with a randomly generated session. Use the Sequencer tab to edit or replace it.
3. Use the scene buttons to switch between pattern scenes.
4. Use the Song Mode tab to chain scenes into a longer arrangement.
5. Use Track Sound Design to edit each drum patch.
6. Save patches with `Save`; load them with `Load`.
7. Use `Export Pattern` or `Export Song` to render a WAV file.

The `Fill` control adds temporary end-of-bar hat and percussion variations during playback. It does not edit the saved pattern.

## Track Sound Presets

Each track editor has `Save Sound` and `Load Sound` buttons. These save or load only that track's sound design, including synthesis, tone/filter, patch effects, and modulation.

Track sound presets do not change pattern steps, step expression, mute/solo state, or song scenes. The default save/load folder is `presets/`.
The preset selector includes an extra `Crash` cymbal sound while the sequencer remains an 8-track layout.

## Pattern Scenes

The numbered scene buttons store separate sequencer patterns. Patch sounds are shared across scenes, while step data is stored per scene.

Use the `Name` field beside the scene buttons to label scenes such as `Intro`, `Main`, `Fill`, or `Drop`. These names appear in Pattern Morph and Song Mode.

The Sequencer tab includes a Step Inspector beside the grid. Click a step to edit its on/off state, velocity, probability, ratchet, step note, mute/solo state, or audition the track without jumping into Track Sound Design.

When Note Mode is enabled on a track, active steps show note names directly in the sequencer grid.

Scene tools:

- `Store`: saves the current pattern into the selected scene
- `Copy Scene` / `Paste Scene`: duplicates full scene step data
- `Clear Scene`: clears all tracks in the current scene
- `Pattern Morph`: chooses a source scene, target scene, write scene, and blend amount, then writes a morphed pattern that blends step activity, velocity, probability, ratchets, and track length

Track tools:

- `Copy Track` / `Paste Track`: duplicates one track pattern
- `Clear Track`: clears one track pattern
- `Rotate Left` / `Rotate Right`: shifts one track pattern by one step

## Note Mode

Enable `Note Mode` in a track's Synthesis tab. In Step Expression or the Sequencer Step Inspector, use `Step Note` to choose the note for the selected step. During playback, hits are rendered at those step notes, so any track can carry tuned movement.

Step notes are stored per pattern scene, rotate with the track pattern, and are saved in project JSON files.

## Song Mode

Song Mode chains pattern scenes together so DSynth can play a longer arrangement.

- `Play Song`: starts playback from the selected song slot
- `Loop`: repeats the song chain when the final slot finishes
- `Add Slot`: adds the current scene to the song chain
- `Scene`: chooses which pattern scene a slot plays
- `Bars`: chooses how many bars that slot lasts
- `Up` / `Down` / `Remove`: edits the slot order
- `Generate`: writes a static multi-scene drum session using the selected pattern count, bars, style, complexity, fills, and variation. Styles include House, Techno, Rock, Hip Hop, Boom Bap, Breakbeat, UK Garage, Jungle, Drum & Bass, Funk, Trap, Reggaeton, Latin, Disco, Electro, Synthwave, Dark Synth, Industrial, IDM, Minimal, Dub, and Half-time.

On startup, DSynth randomly chooses one of these generator styles and writes a fresh multi-scene session before playback begins. The generator controls show the startup values so the session can be regenerated or adjusted from there.

Scene changes happen at bar boundaries. Song chains are saved and loaded with the JSON patch file.
Song Mode has its own main tab, and the top-aligned slot list scrolls when the arrangement grows.

## WAV Export

Use `Export Pattern` to render the current scene as a four-bar WAV loop. Use `Export Song` to render the Song Mode chain once from start to finish.

Exports include per-track patch effects, note-mode steps, ratchets, fill mode, global filter, drive, compression, and master volume.

## LFO Modulation

Each track editor is split into tabs for Synthesis, Tone / Filter, Patch Effects, Modulation, and Step Expression. The Modulation tab contains scrollable sub-tabs for LFO 1, LFO 2, and one one-shot envelope, with source controls separated from modulation depth and the envelope graph.

Controls:

- `LFO 1` / `LFO 2`: turns each LFO source on
- `Shape`: Sine, Triangle, Square, or Random
- `Destination`: Pitch, Filter Cutoff, Pan, Volume, Drive, Tone Level, Noise Level, Delay Send, or Reverb Send
- `Rate`: LFO speed in Hz
- `Phase`: where the LFO starts inside the drum hit
- `Amount`: modulation depth
- `Envelope`: enables the one-shot envelope source
- `Envelope Shape`: drag the four points in the graph to shape modulation over the hit

If the LFO is enabled while Amount is zero, DSynth starts it at 50% so you can hear the change immediately.
For short sounds like kicks, higher Amount, a faster Rate, or a Phase around 25% usually makes pitch and filter modulation much easier to hear.
During playback, LFO phase advances with the sequencer transport, so repeated hits can land at different modulation positions instead of restarting the same pitch/filter movement on every beat.

## Notes

- Patch effects are rendered into each drum hit and cached for stable playback.
- Changing sound-design controls clears the affected track cache.
- Some LFO destinations are more obvious with longer decay sounds, higher LFO amount, or a lower filter cutoff.
