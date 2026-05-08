import json
from pathlib import Path

import pandas as pd

from src.config import PLOT_DIR


def _load(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    before = _load(PLOT_DIR / "transformer_generation_metrics.json")
    after = _load(PLOT_DIR / "rlhf_generation_metrics.json")
    rows = []
    for key in ["pitch_histogram_distance", "rhythm_diversity", "repetition_ratio"]:
        b = before.get(key)
        a = after.get(key)
        rows.append(
            {
                "metric": key,
                "before_transformer": b,
                "after_rlhf": a,
                "delta_after_minus_before": (a - b) if (a is not None and b is not None) else None,
            }
        )
    out = pd.DataFrame(rows)
    out_path = PLOT_DIR / "task4_before_after.csv"
    out.to_csv(out_path, index=False)


if __name__ == "__main__":
    main()
