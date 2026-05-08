import argparse

import numpy as np
import torch
from tqdm import tqdm

from src.config import OUTPUT_DIR, TrainConfig, ensure_dirs
from src.models.autoencoder import LSTMAutoencoder
from src.preprocessing.tokenizer import PAD_TOKEN, add_sos
from src.training.common import (
    build_dataloaders,
    load_sequences,
    save_metrics_json,
    save_processed,
    save_train_val_loss_plot,
    set_seed,
)


def train(args: argparse.Namespace) -> None:
    # Build runtime config from CLI arguments and initialize deterministic behavior.
    cfg = TrainConfig(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)
    set_seed(cfg.seed)
    ensure_dirs()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = load_sequences(cfg, genre=args.genre)
    save_processed(data, f"ae_{args.genre or 'all'}")
    # Split token windows into train/validation loaders.
    train_loader, val_loader = build_dataloaders(data, cfg)

    model = LSTMAutoencoder(
        vocab_size=cfg.vocab_size,
        embed_dim=cfg.embed_dim,
        hidden_dim=cfg.hidden_dim,
        latent_dim=cfg.latent_dim,
        num_layers=cfg.num_layers,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    train_losses = []
    val_losses = []
    token_values = torch.arange(cfg.vocab_size, dtype=torch.float32, device=device).view(1, 1, -1)
    for _ in range(cfg.epochs):
        model.train()
        total_train = 0.0
        for (x,) in tqdm(train_loader, desc="AE training", leave=False):
            x = x.to(device)
            # Teacher-forcing input is the same sequence shifted with SOS token.
            dec_in_np = np.stack([add_sos(sample.cpu().numpy()) for sample in x], axis=0)
            dec_in = torch.from_numpy(dec_in_np).to(device=device, dtype=torch.long)
            logits = model(x, dec_in)
            probs = torch.softmax(logits, dim=-1)
            # Convert token distribution to expected token value and optimize masked MSE.
            x_hat = (probs * token_values).sum(dim=-1)
            mask = (x != PAD_TOKEN).float()
            mse = (x_hat - x.float()) ** 2
            loss = (mse * mask).sum() / mask.sum().clamp_min(1.0)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_train += loss.item()
        train_losses.append(total_train / max(1, len(train_loader)))

        model.eval()
        total_val = 0.0
        with torch.no_grad():
            for (x,) in tqdm(val_loader, desc="AE validation", leave=False):
                x = x.to(device)
                dec_in_np = np.stack([add_sos(sample.cpu().numpy()) for sample in x], axis=0)
                dec_in = torch.from_numpy(dec_in_np).to(device=device, dtype=torch.long)
                logits = model(x, dec_in)
                probs = torch.softmax(logits, dim=-1)
                x_hat = (probs * token_values).sum(dim=-1)
                mask = (x != PAD_TOKEN).float()
                mse = (x_hat - x.float()) ** 2
                loss = (mse * mask).sum() / mask.sum().clamp_min(1.0)
                total_val += loss.item()
        val_losses.append(total_val / max(1, len(val_loader)))

    save_train_val_loss_plot(train_losses, val_losses, "task1_ae_reconstruction_loss")
    torch.save(model.state_dict(), OUTPUT_DIR / "ae_model.pt")
    save_metrics_json(
        {
            "task": "task1_ae",
            "final_train_loss": train_losses[-1],
            "final_val_loss": val_losses[-1],
            "genre": args.genre or "all",
        },
        "task1_metrics",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--genre", type=str, default="unknown")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
