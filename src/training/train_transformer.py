import argparse
import math

import torch
import torch.nn.functional as F
from tqdm import tqdm

from src.config import OUTPUT_DIR, TrainConfig, ensure_dirs
from src.models.transformer import MusicTransformer
from src.preprocessing.tokenizer import PAD_TOKEN
from src.training.common import (
    build_dataloaders_with_genre,
    load_sequences_with_genre,
    save_metrics_json,
    save_processed,
    save_train_val_loss_plot,
    set_seed,
)


def train(args: argparse.Namespace) -> None:
    # Train an autoregressive next-token model with genre-conditioning.
    cfg = TrainConfig(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)
    set_seed(cfg.seed)
    ensure_dirs()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data, genre_ids = load_sequences_with_genre(cfg)
    save_processed(data, "transformer_all_genres")
    train_loader, val_loader = build_dataloaders_with_genre(data, genre_ids, cfg)

    model = MusicTransformer(vocab_size=cfg.vocab_size, num_genres=cfg.num_genres).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    train_losses = []
    val_losses = []
    for _ in range(cfg.epochs):
        model.train()
        total_train = 0.0
        for x, g in tqdm(train_loader, desc="Transformer training", leave=False):
            x = x.to(device)
            g = g.to(device)
            # Shifted inputs/targets for next-token prediction.
            inp = x[:, :-1]
            tgt = x[:, 1:]
            logits = model(inp, genre_ids=g, padding_mask=(inp == PAD_TOKEN))
            loss = F.cross_entropy(logits.reshape(-1, cfg.vocab_size), tgt.reshape(-1), ignore_index=PAD_TOKEN)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_train += loss.item()
        train_losses.append(total_train / max(1, len(train_loader)))

        model.eval()
        total_val = 0.0
        with torch.no_grad():
            for x, g in tqdm(val_loader, desc="Transformer validation", leave=False):
                x = x.to(device)
                g = g.to(device)
                inp = x[:, :-1]
                tgt = x[:, 1:]
                logits = model(inp, genre_ids=g, padding_mask=(inp == PAD_TOKEN))
                loss = F.cross_entropy(logits.reshape(-1, cfg.vocab_size), tgt.reshape(-1), ignore_index=PAD_TOKEN)
                total_val += loss.item()
        val_losses.append(total_val / max(1, len(val_loader)))

    # Convert final validation NLL to perplexity for reporting.
    perplexity = math.exp(val_losses[-1]) if val_losses[-1] < 20 else float("inf")
    save_train_val_loss_plot(train_losses, val_losses, "task3_transformer_nll")
    torch.save(model.state_dict(), OUTPUT_DIR / "transformer_model.pt")
    save_metrics_json(
        {
            "task": "task3_transformer",
            "final_train_nll_loss": train_losses[-1],
            "final_val_nll_loss": val_losses[-1],
            "perplexity": perplexity,
        },
        "task3_metrics",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
