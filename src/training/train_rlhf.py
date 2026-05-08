import argparse

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from src.config import OUTPUT_DIR, TrainConfig, ensure_dirs
from src.evaluation.rhythm_score import repetition_ratio
from src.models.transformer import MusicTransformer
from src.preprocessing.tokenizer import PAD_TOKEN, SOS_TOKEN
from src.training.common import save_metrics_json, set_seed


def heuristic_reward(sequence: np.ndarray) -> float:
    rep = repetition_ratio(sequence, ngram=4)
    unique_pitch = len(np.unique(sequence[(sequence >= 0) & (sequence <= 127)]))
    diversity = min(1.0, unique_pitch / 24.0)
    return float(0.7 * diversity + 0.3 * (1.0 - rep))


def sample_sequence(
    model: MusicTransformer,
    seq_len: int,
    device: torch.device,
    temperature: float = 1.0,
    top_k: int = 32,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    cur = torch.full((1, 1), SOS_TOKEN, dtype=torch.long, device=device)
    genre_id = torch.zeros((1,), dtype=torch.long, device=device)
    log_probs = []
    entropies = []
    tokens = []
    for _ in range(seq_len):
        logits = model(cur, genre_ids=genre_id)
        step_logits = logits[:, -1, :] / max(1e-6, temperature)
        step_logits[:, PAD_TOKEN] = -1e9
        step_logits[:, SOS_TOKEN] = -1e9
        if top_k > 0:
            topk_vals, topk_idx = torch.topk(step_logits, k=min(top_k, step_logits.size(-1)), dim=-1)
            masked = torch.full_like(step_logits, -1e9)
            masked.scatter_(1, topk_idx, topk_vals)
            step_logits = masked
        probs = torch.softmax(step_logits, dim=-1)
        dist = torch.distributions.Categorical(probs=probs)
        token = dist.sample()
        log_probs.append(dist.log_prob(token))
        entropies.append(dist.entropy())
        tokens.append(int(token.item()))
        cur = torch.cat([cur, token.unsqueeze(1)], dim=1)
    return (
        torch.stack(log_probs),
        torch.stack(entropies),
        torch.tensor(tokens, dtype=torch.long, device=device),
    )


def train(args: argparse.Namespace) -> None:
    cfg = TrainConfig(seq_len=args.seq_len, lr=args.lr)
    set_seed(cfg.seed)
    ensure_dirs()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = MusicTransformer(cfg.vocab_size, num_genres=cfg.num_genres).to(device)
    base_path = OUTPUT_DIR / "transformer_model.pt"
    if base_path.exists():
        model.load_state_dict(torch.load(base_path, map_location=device), strict=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    rewards = []
    baseline = 0.0
    baseline_momentum = 0.95
    entropy_coef = 1e-3
    model.train()
    for _ in tqdm(range(args.steps), desc="RLHF tuning"):
        log_probs, entropies, tokens = sample_sequence(model, cfg.seq_len, device)
        reward = heuristic_reward(tokens.detach().cpu().numpy())
        baseline = baseline_momentum * baseline + (1.0 - baseline_momentum) * reward
        advantage = reward - baseline
        # REINFORCE with moving baseline and entropy bonus to reduce mode collapse.
        loss = -(advantage * log_probs.sum()) - entropy_coef * entropies.sum()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        rewards.append(reward)

    torch.save(model.state_dict(), OUTPUT_DIR / "transformer_rlhf_model.pt")
    save_metrics_json(
        {
            "task": "task4_rlhf",
            "steps": args.steps,
            "mean_reward": float(np.mean(rewards)),
            "max_reward": float(np.max(rewards)),
        },
        "task4_metrics",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--seq-len", type=int, default=128)
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
