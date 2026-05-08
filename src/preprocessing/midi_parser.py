from pathlib import Path
from typing import List

import pretty_midi

from src.config import GENRE_TO_ID, RAW_MIDI_DIR
from src.preprocessing.tokenizer import (
    duration_steps_to_token,
    note_off_token,
    note_on_token,
    velocity_to_token,
)


def _midi_paths_under(root: Path) -> List[Path]:
    return sorted(p for pat in ("*.mid", "*.midi") for p in root.rglob(pat))


def _step_duration_seconds(midi: pretty_midi.PrettyMIDI) -> float:
    default_tempo = 120.0
    est = float(midi.estimate_tempo())
    tempo = est if est > 0 else default_tempo
    # 16 steps per 4/4 bar => 4 steps per beat.
    return (60.0 / tempo) / 4.0


def midi_to_event_sequence(midi_path: Path) -> List[int]:
    midi = pretty_midi.PrettyMIDI(str(midi_path))

    step_sec = _step_duration_seconds(midi)
    notes = []
    for instrument in midi.instruments:
        if instrument.is_drum:
            continue
        for note in instrument.notes:
            duration = max(1e-4, note.end - note.start)
            duration_steps = max(1, int(round(duration / step_sec)))
            notes.append((note.start, note.pitch, note.velocity, duration_steps))
    notes.sort(key=lambda x: x[0])

    events: List[int] = []
    for _, pitch, velocity, duration_steps in notes:
        events.extend(
            [
                velocity_to_token(int(velocity)),
                note_on_token(int(pitch)),
                duration_steps_to_token(int(duration_steps)),
                note_off_token(int(pitch)),
            ]
        )
    return events


def collect_midi_files(genre: str | None = None) -> List[Path]:
    root = RAW_MIDI_DIR / genre if genre else RAW_MIDI_DIR
    if genre and not root.exists():
        return []
    return _midi_paths_under(root)


def infer_genre_id(midi_path: Path) -> int:
    rel = midi_path.relative_to(RAW_MIDI_DIR)
    top = rel.parts[0].lower() if rel.parts else "unknown"
    return GENRE_TO_ID.get(top, GENRE_TO_ID["unknown"])