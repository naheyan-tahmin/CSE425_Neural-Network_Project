import numpy as np


def rhythm_diversity(durations: np.ndarray) -> float:
    if durations.size == 0:
        return 0.0
    quantized = np.round(durations.astype(np.float32), 3)
    unique = np.unique(quantized).size
    return float(unique / max(1, durations.size))


def repetition_ratio(sequence: np.ndarray, ngram: int = 4) -> float:
    if sequence.size < ngram:
        return 0.0
    patterns = [tuple(sequence[i : i + ngram].tolist()) for i in range(sequence.size - ngram + 1)]
    total = len(patterns)
    repeated = total - len(set(patterns))
    return float(repeated / max(1, total))
