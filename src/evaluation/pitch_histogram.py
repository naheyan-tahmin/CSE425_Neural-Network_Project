import numpy as np


def pitch_histogram(sequence: np.ndarray) -> np.ndarray:
    valid = sequence[(sequence >= 0) & (sequence <= 127)]
    if valid.size == 0:
        return np.zeros(12, dtype=np.float32)
    bins = np.zeros(12, dtype=np.float32)
    for p in valid:
        bins[int(p) % 12] += 1
    bins /= bins.sum() + 1e-8
    return bins


def histogram_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.abs(a - b).sum())
