import json
import random
import warnings
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

from src.config import PLOT_DIR, PROCESSED_DIR, TrainConfig, ensure_dirs
from src.preprocessing.midi_parser import collect_midi_files, infer_genre_id, midi_to_event_sequence
from src.preprocessing.tokenizer import batch_pad


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def segment_sequence(sequence: List[int], seq_len: int) -> List[List[int]]:
    if len(sequence) < seq_len:
        return [sequence]
    windows: List[List[int]] = []
    for start in range(0, len(sequence) - seq_len + 1, seq_len):
        windows.append(sequence[start : start + seq_len])
    return windows


def load_sequences(cfg: TrainConfig, genre: str | None = None) -> np.ndarray:
    midi_files = collect_midi_files(genre)
    sequences: List[List[int]] = []
    skipped = 0
    for midi_file in midi_files:
        try:
            seq = midi_to_event_sequence(midi_file)
        except Exception as exc:
            skipped += 1
            warnings.warn(f"Skipping unreadable MIDI file '{midi_file}': {exc}", RuntimeWarning)
            continue
        if len(seq) >= 8:
            sequences.extend(segment_sequence(seq, cfg.seq_len))

    if not sequences:
        raise ValueError(
            "No MIDI sequences found: add .mid/.midi files under the raw data directory "
            "or check that parsing produces sequences of at least 8 tokens."
        )

    if skipped:
        warnings.warn(f"Skipped {skipped} unreadable MIDI file(s).", RuntimeWarning)

    return batch_pad(sequences, cfg.seq_len)


def load_sequences_with_genre(cfg: TrainConfig) -> Tuple[np.ndarray, np.ndarray]:
    midi_files = collect_midi_files(genre=None)
    sequences: List[List[int]] = []
    genre_ids: List[int] = []
    skipped = 0
    for midi_file in midi_files:
        try:
            seq = midi_to_event_sequence(midi_file)
        except Exception as exc:
            skipped += 1
            warnings.warn(f"Skipping unreadable MIDI file '{midi_file}': {exc}", RuntimeWarning)
            continue
        if len(seq) >= 8:
            windows = segment_sequence(seq, cfg.seq_len)
            sequences.extend(windows)
            genre_ids.extend([infer_genre_id(midi_file)] * len(windows))

    if not sequences:
        raise ValueError(
            "No MIDI sequences found: add .mid/.midi files under the raw data directory "
            "or check that parsing produces sequences of at least 8 tokens."
        )

    if skipped:
        warnings.warn(f"Skipped {skipped} unreadable MIDI file(s).", RuntimeWarning)

    return batch_pad(sequences, cfg.seq_len), np.array(genre_ids, dtype=np.int64)


def save_processed(data: np.ndarray, name: str) -> Path:
    ensure_dirs()
    out_path = PROCESSED_DIR / f"{name}.npy"
    np.save(out_path, data)
    return out_path


def build_dataloaders(data: np.ndarray, cfg: TrainConfig) -> Tuple[DataLoader, DataLoader]:
    train_x, test_x = train_test_split(data, train_size=cfg.train_split, random_state=cfg.seed, shuffle=True)
    train_t = torch.tensor(train_x, dtype=torch.long)
    test_t = torch.tensor(test_x, dtype=torch.long)
    train_loader = DataLoader(TensorDataset(train_t), batch_size=cfg.batch_size, shuffle=True)
    test_loader = DataLoader(TensorDataset(test_t), batch_size=cfg.batch_size, shuffle=False)
    return train_loader, test_loader


def build_dataloaders_with_genre(data: np.ndarray, genres: np.ndarray, cfg: TrainConfig) -> Tuple[DataLoader, DataLoader]:
    indices = np.arange(len(data))
    train_idx, test_idx = train_test_split(indices, train_size=cfg.train_split, random_state=cfg.seed, shuffle=True)
    train_x = torch.tensor(data[train_idx], dtype=torch.long)
    test_x = torch.tensor(data[test_idx], dtype=torch.long)
    train_g = torch.tensor(genres[train_idx], dtype=torch.long)
    test_g = torch.tensor(genres[test_idx], dtype=torch.long)
    train_loader = DataLoader(TensorDataset(train_x, train_g), batch_size=cfg.batch_size, shuffle=True)
    test_loader = DataLoader(TensorDataset(test_x, test_g), batch_size=cfg.batch_size, shuffle=False)
    return train_loader, test_loader


def save_loss_plot(losses: List[float], out_name: str) -> None:
    ensure_dirs()
    plt.figure(figsize=(7, 4))
    plt.plot(losses, label="Train Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(out_name.replace("_", " ").title())
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOT_DIR / f"{out_name}.png", dpi=150)
    plt.close()


def save_train_val_loss_plot(train_losses: List[float], val_losses: List[float], out_name: str) -> None:
    ensure_dirs()
    plt.figure(figsize=(7, 4))
    plt.plot(train_losses, label="Train Loss", linestyle="-")
    plt.plot(val_losses, label="Validation Loss", linestyle="--")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(out_name.replace("_", " ").title())
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOT_DIR / f"{out_name}.png", dpi=150)
    plt.close()


def save_metrics_json(metrics: dict, out_name: str) -> None:
    ensure_dirs()
    with open(PLOT_DIR / f"{out_name}.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
