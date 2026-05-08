from pathlib import Path

import pretty_midi

from src.preprocessing.tokenizer import (
    DURATION_START,
    NOTE_OFF_START,
    NOTE_ON_START,
    SOS_TOKEN,
    VELOCITY_START,
    token_to_duration_steps,
    token_to_velocity,
)


def event_sequence_to_midi(sequence: list[int], output_path: Path, tempo: float = 120.0) -> None:
    pm = pretty_midi.PrettyMIDI(initial_tempo=tempo)
    instrument = pretty_midi.Instrument(program=0)
    step_duration = (60.0 / tempo) / 4.0
    current_time = 0.0
    current_velocity = 100
    pending_pitch: int | None = None
    pending_duration_steps = 4

    for token in sequence:
        tok = int(token)
        if tok == SOS_TOKEN:
            continue
        if VELOCITY_START <= tok < VELOCITY_START + 32:
            current_velocity = token_to_velocity(tok)
        elif NOTE_ON_START <= tok < NOTE_ON_START + 128:
            pending_pitch = tok - NOTE_ON_START
        elif DURATION_START <= tok < DURATION_START + 32:
            pending_duration_steps = token_to_duration_steps(tok)
        elif NOTE_OFF_START <= tok < NOTE_OFF_START + 128:
            pitch = tok - NOTE_OFF_START
            if pending_pitch is not None:
                pitch = pending_pitch
            duration = pending_duration_steps * step_duration
            note = pretty_midi.Note(
                velocity=int(current_velocity),
                pitch=int(pitch),
                start=current_time,
                end=current_time + duration,
            )
            instrument.notes.append(note)
            current_time += duration
            pending_pitch = None

    pm.instruments.append(instrument)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pm.write(str(output_path))
