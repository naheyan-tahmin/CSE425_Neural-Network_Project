from typing import Iterable, List

import numpy as np


NOTE_ON_START = 0
NOTE_OFF_START = 128
VELOCITY_START = 256
DURATION_START = 288
VELOCITY_BINS = 32
DURATION_BINS = 32
PAD_TOKEN = 320
SOS_TOKEN = 321
VOCAB_SIZE = 322


def pad_or_truncate(sequence: Iterable[int], seq_len: int) -> np.ndarray:
    seq = list(sequence)[:seq_len]
    if len(seq) < seq_len:
        seq += [PAD_TOKEN] * (seq_len - len(seq))
    return np.array(seq, dtype=np.int64)


def add_sos(sequence: np.ndarray) -> np.ndarray:
    return np.concatenate([[SOS_TOKEN], sequence[:-1]])


def batch_pad(sequences: List[Iterable[int]], seq_len: int) -> np.ndarray:
    return np.stack([pad_or_truncate(s, seq_len) for s in sequences], axis=0)


def velocity_to_token(velocity: int) -> int:
    v = int(np.clip(velocity, 0, 127))
    bin_idx = min(VELOCITY_BINS - 1, (v * VELOCITY_BINS) // 128)
    return VELOCITY_START + bin_idx


def token_to_velocity(token: int) -> int:
    bin_idx = int(np.clip(token - VELOCITY_START, 0, VELOCITY_BINS - 1))
    return int((bin_idx + 0.5) * (128.0 / VELOCITY_BINS))


def duration_steps_to_token(steps: int) -> int:
    s = int(np.clip(steps, 1, DURATION_BINS))
    return DURATION_START + (s - 1)


def token_to_duration_steps(token: int) -> int:
    return int(np.clip(token - DURATION_START + 1, 1, DURATION_BINS))


def note_on_token(pitch: int) -> int:
    return NOTE_ON_START + int(np.clip(pitch, 0, 127))


def note_off_token(pitch: int) -> int:
    return NOTE_OFF_START + int(np.clip(pitch, 0, 127))


def pitch_sequence_to_event_tokens(pitches: Iterable[int], duration_steps: int = 4, velocity: int = 100) -> np.ndarray:
    events: List[int] = []
    vel_tok = velocity_to_token(velocity)
    dur_tok = duration_steps_to_token(duration_steps)
    for p in pitches:
        pitch = int(np.clip(p, 0, 127))
        events.extend([vel_tok, note_on_token(pitch), dur_tok, note_off_token(pitch)])
    return np.array(events, dtype=np.int64)
