import argparse
import json
from pathlib import Path

import numpy as np
import torch

from src.config import GENRE_TO_ID, MIDI_OUTPUT_DIR, OUTPUT_DIR, TrainConfig, ensure_dirs
from src.evaluation.metrics import evaluate_generated_vs_reference, extract_note_durations
from src.generation.midi_export import event_sequence_to_midi
from src.models.autoencoder import LSTMAutoencoder
from src.models.transformer import MusicTransformer
from src.models.vae import LSTMVAE
from src.preprocessing.midi_parser import collect_midi_files, midi_to_event_sequence
from src.preprocessing.tokenizer import NOTE_ON_START, PAD_TOKEN, SOS_TOKEN, pitch_sequence_to_event_tokens


def _load_latent_decoder(
    model: LSTMAutoencoder | LSTMVAE,
    weights: Path,
    device: torch.device,
) -> None:
    if weights.exists():
        model.load_state_dict(torch.load(weights, map_location=device), strict=False)
    model.eval()


def _decode_from_random_latent(
    model: LSTMAutoencoder | LSTMVAE,
    cfg: TrainConfig,
    num_samples: int,
    device: torch.device,
    autoregressive: bool = False,
    temperature: float = 1.0,
    top_k: int = 32,
) -> np.ndarray:
    # Sample random latent vectors and decode them into token sequences.
    z = torch.randn(num_samples, cfg.latent_dim, device=device)
    if not autoregressive:
        dec_in = torch.full((num_samples, cfg.seq_len), SOS_TOKEN, dtype=torch.long, device=device)
        with torch.no_grad():
            logits = model.decode(z, dec_in)
        return torch.argmax(logits, dim=-1).cpu().numpy()

    # Optional autoregressive latent decoding with top-k sampling.
    cur = torch.full((num_samples, 1), SOS_TOKEN, dtype=torch.long, device=device)
    banned_tokens = torch.tensor([PAD_TOKEN, SOS_TOKEN], dtype=torch.long, device=device)
    out_tokens: list[torch.Tensor] = []
    with torch.no_grad():
        for _ in range(cfg.seq_len):
            logits = model.decode(z, cur)
            step_logits = logits[:, -1, :] / max(1e-6, temperature)
            step_logits[:, banned_tokens] = -1e9
            if top_k > 0:
                topk_vals, topk_idx = torch.topk(step_logits, k=min(top_k, step_logits.size(-1)), dim=-1)
                masked = torch.full_like(step_logits, -1e9)
                masked.scatter_(1, topk_idx, topk_vals)
                step_logits = masked
            probs = torch.softmax(step_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            out_tokens.append(next_token.squeeze(1))
            cur = torch.cat([cur, next_token], dim=1)
    return torch.stack(out_tokens, dim=1).cpu().numpy()


def _write_samples(
    tokens: np.ndarray,
    out_prefix: str,
) -> tuple[list[np.ndarray], list[str]]:
    # Export generated token sequences into MIDI files and return both arrays and paths.
    generated, paths = [], []
    for i in range(tokens.shape[0]):
        seq = tokens[i].astype(np.int64)
        generated.append(seq)
        out_path = MIDI_OUTPUT_DIR / f"{out_prefix}_{i + 1}.mid"
        event_sequence_to_midi(seq.tolist(), out_path)
        paths.append(str(out_path))
    return generated, paths


def generate_from_ae(cfg: TrainConfig, num_samples: int, out_prefix: str) -> tuple[list[np.ndarray], list[str]]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTMAutoencoder(cfg.vocab_size, cfg.embed_dim, cfg.hidden_dim, cfg.latent_dim, cfg.num_layers).to(device)
    _load_latent_decoder(model, OUTPUT_DIR / "ae_model.pt", device)
    tokens = _decode_from_random_latent(model, cfg, num_samples, device)
    return _write_samples(tokens, out_prefix)


def generate_from_vae(cfg: TrainConfig, num_samples: int, out_prefix: str) -> tuple[list[np.ndarray], list[str]]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTMVAE(cfg.vocab_size, cfg.embed_dim, cfg.hidden_dim, cfg.latent_dim, cfg.num_layers).to(device)
    _load_latent_decoder(model, OUTPUT_DIR / "vae_model.pt", device)
    tokens = _decode_from_random_latent(model, cfg, num_samples, device, autoregressive=True)
    return _write_samples(tokens, out_prefix)


def _autoregressive_transformer_samples(
    model: MusicTransformer,
    cfg: TrainConfig,
    num_samples: int,
    genre_id: int,
    device: torch.device,
) -> list[np.ndarray]:
    genre_tensor = torch.tensor([genre_id], dtype=torch.long, device=device)
    banned_tokens = torch.tensor([PAD_TOKEN, SOS_TOKEN], dtype=torch.long, device=device)
    out: list[np.ndarray] = []
    with torch.no_grad():
        for _ in range(num_samples):
            cur = torch.full((1, 1), SOS_TOKEN, dtype=torch.long, device=device)
            step_tokens: list[int] = []
            for _ in range(cfg.seq_len):
                logits = model(cur, genre_ids=genre_tensor)
                step_logits = logits[:, -1, :]
                step_logits[:, banned_tokens] = -1e9
                probs = torch.softmax(step_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                step_tokens.append(int(next_token.item()))
                cur = torch.cat([cur, next_token], dim=1)
            out.append(np.array(step_tokens, dtype=np.int64))
    return out


def generate_from_transformer(
    cfg: TrainConfig,
    num_samples: int,
    out_prefix: str,
    genre_name: str = "unknown",
) -> tuple[list[np.ndarray], list[str]]:
    return _generate_transformer_family(
        cfg, num_samples, out_prefix, genre_name, OUTPUT_DIR / "transformer_model.pt"
    )


def generate_from_rlhf_transformer(
    cfg: TrainConfig,
    num_samples: int,
    out_prefix: str,
    genre_name: str = "unknown",
) -> tuple[list[np.ndarray], list[str]]:
    return _generate_transformer_family(
        cfg, num_samples, out_prefix, genre_name, OUTPUT_DIR / "transformer_rlhf_model.pt"
    )


def _generate_transformer_family(
    cfg: TrainConfig,
    num_samples: int,
    out_prefix: str,
    genre_name: str,
    weights: Path,
) -> tuple[list[np.ndarray], list[str]]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MusicTransformer(cfg.vocab_size, num_genres=cfg.num_genres).to(device)
    if weights.exists():
        model.load_state_dict(torch.load(weights, map_location=device), strict=False)
    model.eval()
    genre_id = GENRE_TO_ID.get(genre_name.lower(), GENRE_TO_ID["unknown"])
    seqs = _autoregressive_transformer_samples(model, cfg, num_samples, genre_id, device)
    return _write_samples(np.stack(seqs), out_prefix)


def _export_baseline_midis(sequences: list[np.ndarray], stem: str) -> list[str]:
    paths: list[str] = []
    for i, seq in enumerate(sequences):
        out_path = MIDI_OUTPUT_DIR / f"{stem}_{i + 1}.mid"
        event_sequence_to_midi(seq.tolist(), out_path)
        paths.append(str(out_path))
    return paths


def random_baseline(seq_len: int, num_samples: int) -> list[np.ndarray]:
    rng = np.random.default_rng(42)
    seqs = []
    for _ in range(num_samples):
        pitch_count = max(1, seq_len // 4)
        pitches = rng.integers(48, 84, size=pitch_count, dtype=np.int64)
        seqs.append(pitch_sequence_to_event_tokens(pitches)[:seq_len])
    return seqs


def markov_baseline(seq_len: int, num_samples: int) -> list[np.ndarray]:
    transitions = {
        60: [62, 64, 67],
        62: [64, 65, 69],
        64: [65, 67, 71],
        65: [67, 69, 72],
        67: [69, 71, 74],
        69: [71, 72, 76],
        71: [72, 74, 79],
        72: [60, 64, 67],
    }
    rng = np.random.default_rng(7)
    samples = []
    for _ in range(num_samples):
        cur = 60
        seq = [cur]
        for _ in range(seq_len - 1):
            nxt = rng.choice(transitions.get(cur, [60, 62, 64]))
            seq.append(int(nxt))
            cur = int(nxt)
        events = pitch_sequence_to_event_tokens(seq)
        samples.append(events[:seq_len])
    return samples


def _reference_event_sequence(seq_len: int) -> np.ndarray:
    # Use the first available dataset sample as a lightweight reference for automatic metrics.
    files = collect_midi_files(None)
    if not files:
        raise ValueError(
            "No MIDI files in the raw data directory; cannot build an evaluation reference sequence."
        )
    events = midi_to_event_sequence(files[0])
    arr = np.array(events[:seq_len], dtype=np.int64)
    if arr.size < seq_len:
        arr = np.pad(arr, (0, seq_len - arr.size), constant_values=PAD_TOKEN)
    return arr


def evaluate_and_save(name: str, generated: list[np.ndarray], midi_paths: list[str]) -> None:
    # Compute per-sample metrics and store model-level average JSON.
    reference = _reference_event_sequence(len(generated[0]))
    metrics = []
    for seq, midi_path in zip(generated, midi_paths):
        pitches = seq[(seq >= NOTE_ON_START) & (seq < NOTE_ON_START + 128)] - NOTE_ON_START
        ref_pitches = reference[(reference >= NOTE_ON_START) & (reference < NOTE_ON_START + 128)] - NOTE_ON_START
        durations = extract_note_durations(midi_path)
        metrics.append(evaluate_generated_vs_reference(pitches, ref_pitches, durations=durations))
    avg = {
        key: float(np.mean([m[key] for m in metrics]))
        for key in ["pitch_histogram_distance", "rhythm_diversity", "repetition_ratio"]
    }
    out_path = OUTPUT_DIR / "plots" / f"{name}_generation_metrics.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(avg, f, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["ae", "vae", "transformer", "rlhf", "random", "markov"], required=True)
    parser.add_argument("--num-samples", type=int, default=5)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--genre", type=str, default="unknown")
    args = parser.parse_args()

    ensure_dirs()
    cfg = TrainConfig(seq_len=args.seq_len)

    genre_tag = args.genre.lower()
    if args.model == "ae":
        generated, midi_paths = generate_from_ae(cfg, args.num_samples, f"task1_ae_genre-{genre_tag}")
    elif args.model == "vae":
        generated, midi_paths = generate_from_vae(cfg, args.num_samples, f"task2_vae_genre-{genre_tag}")
    elif args.model == "transformer":
        generated, midi_paths = generate_from_transformer(
            cfg, args.num_samples, f"task3_transformer_genre-{genre_tag}", args.genre
        )
    elif args.model == "rlhf":
        generated, midi_paths = generate_from_rlhf_transformer(
            cfg, args.num_samples, f"task4_rlhf_genre-{genre_tag}", args.genre
        )
    elif args.model == "random":
        generated = random_baseline(cfg.seq_len, args.num_samples)
        midi_paths = _export_baseline_midis(generated, "baseline_random")
    else:
        generated = markov_baseline(cfg.seq_len, args.num_samples)
        midi_paths = _export_baseline_midis(generated, "baseline_markov")

    evaluate_and_save(args.model, generated, midi_paths)


if __name__ == "__main__":
    main()
