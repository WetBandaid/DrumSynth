r"""
DSynth - PySide6 / NumPy drum synthesizer and step sequencer.

Project:
    Desktop drum machine with 8-track sequencing, song mode, patch editing,
    modulation, effects, spectrum analysis, and WAV export.

Run:
    .\Template.venv\Scripts\python.exe DSynth.py

Dependencies:
    Python 3.11+
    PySide6
    NumPy
    sounddevice

Entry point:
    Imports and launches drum_machine.main_window.main().
"""

from drum_machine.main_window import main


if __name__ == "__main__":
    raise SystemExit(main())
