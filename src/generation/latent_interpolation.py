import argparse

import numpy as np
import torch

from src.config import MIDI_OUTPUT_DIR, OUTPUT_DIR, TrainConfig, ensure_dirs
from src.generation.midi_export import event_sequence_to_midi
from src.models.vae import LSTMVAE
from src.preprocessing.tokenizer import SOS_TOKEN


def run_interpolation(steps: int = 8, seq_len: int = 128) -> None:
    ensure_dirs()
    cfg = TrainConfig(seq_len=seq_len)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTMVAE(cfg.vocab_size, cfg.embed_dim, cfg.hidden_dim, cfg.latent_dim, cfg.num_layers).to(device)
    model_path = OUTPUT_DIR / "vae_model.pt"
    if model_path.exists():
        model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    with torch.no_grad():
        z_a = torch.randn(1, cfg.latent_dim, device=device)
        z_b = torch.randn(1, cfg.latent_dim, device=device)
        dec_in = torch.full((1, cfg.seq_len), SOS_TOKEN, dtype=torch.long, device=device)
        for i, alpha in enumerate(np.linspace(0.0, 1.0, steps), start=1):
            z = (1.0 - float(alpha)) * z_a + float(alpha) * z_b
            logits = model.decode(z, dec_in)
            seq = torch.argmax(logits, dim=-1).squeeze(0).cpu().numpy()
            seq = seq.astype(np.int64)
            out_path = MIDI_OUTPUT_DIR / f"task2_vae_interp_{i}.mid"
            event_sequence_to_midi(seq.tolist(), out_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=128)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_interpolation(steps=args.steps, seq_len=args.seq_len)
