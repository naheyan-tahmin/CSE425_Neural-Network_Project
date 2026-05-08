import argparse

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from src.config import OUTPUT_DIR, TrainConfig, ensure_dirs
from src.models.vae import LSTMVAE, kl_divergence
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
    # VAE uses beta to balance reconstruction fidelity and latent regularization.
    cfg = TrainConfig(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, beta=args.beta)
    set_seed(cfg.seed)
    ensure_dirs()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = load_sequences(cfg, genre=None)
    save_processed(data, "vae_all_genres")
    train_loader, val_loader = build_dataloaders(data, cfg)

    model = LSTMVAE(
        vocab_size=cfg.vocab_size,
        embed_dim=cfg.embed_dim,
        hidden_dim=cfg.hidden_dim,
        latent_dim=cfg.latent_dim,
        num_layers=cfg.num_layers,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    train_losses = []
    val_losses = []
    train_recon_losses = []
    train_kl_losses = []
    val_recon_losses = []
    val_kl_losses = []
    for _ in range(cfg.epochs):
        model.train()
        total_train = 0.0
        total_train_recon = 0.0
        total_train_kl = 0.0
        for (x,) in tqdm(train_loader, desc="VAE training", leave=False):
            x = x.to(device)
            # Decoder receives SOS-shifted tokens for teacher forcing.
            dec_in_np = np.stack([add_sos(sample.cpu().numpy()) for sample in x], axis=0)
            dec_in = torch.from_numpy(dec_in_np).to(device=device, dtype=torch.long)
            logits, mu, logvar = model(x, dec_in)
            recon = F.cross_entropy(
                logits.reshape(-1, cfg.vocab_size),
                x.reshape(-1),
                ignore_index=PAD_TOKEN,
            )
            kl = kl_divergence(mu, logvar)
            # Standard beta-VAE objective.
            loss = recon + cfg.beta * kl
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_train += loss.item()
            total_train_recon += recon.item()
            total_train_kl += kl.item()
        train_losses.append(total_train / max(1, len(train_loader)))
        train_recon_losses.append(total_train_recon / max(1, len(train_loader)))
        train_kl_losses.append(total_train_kl / max(1, len(train_loader)))

        model.eval()
        total_val = 0.0
        total_val_recon = 0.0
        total_val_kl = 0.0
        with torch.no_grad():
            for (x,) in tqdm(val_loader, desc="VAE validation", leave=False):
                x = x.to(device)
                dec_in_np = np.stack([add_sos(sample.cpu().numpy()) for sample in x], axis=0)
                dec_in = torch.from_numpy(dec_in_np).to(device=device, dtype=torch.long)
                logits, mu, logvar = model(x, dec_in)
                recon = F.cross_entropy(
                    logits.reshape(-1, cfg.vocab_size),
                    x.reshape(-1),
                    ignore_index=PAD_TOKEN,
                )
                kl = kl_divergence(mu, logvar)
                loss = recon + cfg.beta * kl
                total_val += loss.item()
                total_val_recon += recon.item()
                total_val_kl += kl.item()
        val_losses.append(total_val / max(1, len(val_loader)))
        val_recon_losses.append(total_val_recon / max(1, len(val_loader)))
        val_kl_losses.append(total_val_kl / max(1, len(val_loader)))

    save_train_val_loss_plot(train_losses, val_losses, "task2_vae_total_loss")
    save_train_val_loss_plot(train_recon_losses, val_recon_losses, "task2_vae_reconstruction_loss")
    save_train_val_loss_plot(train_kl_losses, val_kl_losses, "task2_vae_kl_loss")
    torch.save(model.state_dict(), OUTPUT_DIR / "vae_model.pt")
    save_metrics_json(
        {
            "task": "task2_vae",
            "final_total_train_loss": train_losses[-1],
            "final_total_val_loss": val_losses[-1],
            "final_reconstruction_train_loss": train_recon_losses[-1],
            "final_reconstruction_val_loss": val_recon_losses[-1],
            "final_kl_train_loss": train_kl_losses[-1],
            "final_kl_val_loss": val_kl_losses[-1],
            "beta": cfg.beta,
        },
        "task2_metrics",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--beta", type=float, default=0.01)
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
