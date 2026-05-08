import numpy as np
import pretty_midi

from src.evaluation.pitch_histogram import histogram_distance, pitch_histogram
from src.evaluation.rhythm_score import repetition_ratio, rhythm_diversity


def extract_note_durations(midi_path: str) -> np.ndarray:
    midi = pretty_midi.PrettyMIDI(midi_path)
    durations = []
    for inst in midi.instruments:
        if inst.is_drum:
            continue
        for note in inst.notes:
            durations.append(max(1e-4, note.end - note.start))
    if not durations:
        return np.array([], dtype=np.float32)
    return np.array(durations, dtype=np.float32)


def evaluate_generated_vs_reference(generated: np.ndarray, reference: np.ndarray, durations: np.ndarray | None = None) -> dict:
    gen_hist = pitch_histogram(generated)
    ref_hist = pitch_histogram(reference)
    pitch_dist = histogram_distance(gen_hist, ref_hist)

    if durations is None or durations.size == 0:
        # Fallback when MIDI note durations are unavailable.
        local_durations = []
        run = 1
        for i in range(1, generated.size):
            if generated[i] == generated[i - 1]:
                run += 1
            else:
                local_durations.append(run)
                run = 1
        local_durations.append(run)
        durations = np.array(local_durations, dtype=np.int64)

    return {
        "pitch_histogram_distance": float(pitch_dist),
        "rhythm_diversity": float(rhythm_diversity(durations)),
        "repetition_ratio": float(repetition_ratio(generated)),
    }
