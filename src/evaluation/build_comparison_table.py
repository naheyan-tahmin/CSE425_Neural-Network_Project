import json
from pathlib import Path

import pandas as pd

from src.config import OUTPUT_DIR, PLOT_DIR, SURVEY_DIR, ensure_dirs


def load_json(path: Path) -> dict:
    # Return empty dict if metric file is missing to keep table builder robust.
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def mean_human_score_by_model() -> dict[str, float]:
    # Aggregate optional human survey ratings into a model->mean score mapping.
    survey_file = SURVEY_DIR / "human_survey_results.csv"
    if not survey_file.exists():
        return {}
    df = pd.read_csv(survey_file)
    if "score_1_to_5" not in df.columns or "model" not in df.columns:
        return {}
    df["score_1_to_5"] = pd.to_numeric(df["score_1_to_5"], errors="coerce")
    df = df.dropna(subset=["score_1_to_5"])
    if df.empty:
        return {}
    grouped = df.groupby("model")["score_1_to_5"].mean().to_dict()
    return {str(k): float(v) for k, v in grouped.items()}


def build_table() -> None:
    # Assemble one final CSV by merging automatic metrics, task metrics, and human ratings.
    ensure_dirs()
    rows = [
        {"model": "Random Generator"},
        {"model": "Markov Chain"},
        {"model": "Task 1: Autoencoder"},
        {"model": "Task 2: VAE Multi-Genre"},
        {"model": "Task 3: Transformer"},
        {"model": "Task 4: RLHF-Tuned"},
    ]
    df = pd.DataFrame(rows)

    metric_map = {
        "Random Generator": "random_generation_metrics.json",
        "Markov Chain": "markov_generation_metrics.json",
        "Task 1: Autoencoder": "ae_generation_metrics.json",
        "Task 2: VAE Multi-Genre": "vae_generation_metrics.json",
        "Task 3: Transformer": "transformer_generation_metrics.json",
        "Task 4: RLHF-Tuned": "rlhf_generation_metrics.json",
    }
    for idx, row in df.iterrows():
        metrics = load_json(PLOT_DIR / metric_map.get(row["model"], ""))
        df.loc[idx, "pitch_histogram_distance"] = metrics.get("pitch_histogram_distance")
        df.loc[idx, "rhythm_diversity"] = metrics.get("rhythm_diversity")
        df.loc[idx, "repetition_ratio"] = metrics.get("repetition_ratio")

    task1 = load_json(PLOT_DIR / "task1_metrics.json")
    task2 = load_json(PLOT_DIR / "task2_metrics.json")
    task3 = load_json(PLOT_DIR / "task3_metrics.json")
    task4 = load_json(PLOT_DIR / "task4_metrics.json")

    df.loc[df["model"] == "Task 1: Autoencoder", "loss"] = task1.get("final_train_loss")
    df.loc[df["model"] == "Task 2: VAE Multi-Genre", "loss"] = task2.get(
        "final_total_val_loss",
        task2.get("final_total_loss"),
    )
    df.loc[df["model"] == "Task 3: Transformer", "perplexity"] = task3.get("perplexity")
    df.loc[df["model"] == "Task 4: RLHF-Tuned", "reward_mean"] = task4.get("mean_reward")

    human_by_model = mean_human_score_by_model()
    model_key = {
        "Random Generator": "random",
        "Markov Chain": "markov",
        "Task 1: Autoencoder": "ae",
        "Task 2: VAE Multi-Genre": "vae",
        "Task 3: Transformer": "transformer",
        "Task 4: RLHF-Tuned": "rlhf",
    }
    for row_name, key in model_key.items():
        if key in human_by_model:
            df.loc[df["model"] == row_name, "human_score"] = human_by_model[key]

    out_csv = OUTPUT_DIR / "plots" / "comparison_table.csv"
    df.to_csv(out_csv, index=False)


if __name__ == "__main__":
    build_table()
