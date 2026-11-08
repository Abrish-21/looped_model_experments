from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import torch
from torch import nn


@dataclass(frozen=True)
class OptimizerSpec:
    name: str
    lr: float


class LoopedLatentReasoner(nn.Module):
    """A compact recurrent/looped transformer-style model with shared block weights."""

    def __init__(
        self,
        vocab_size: int = 2,
        d_model: int = 64,
        n_heads: int = 4,
        ff_mult: int = 2,
        num_loops: int = 4,
        seq_len: int = 24,
    ) -> None:
        super().__init__()
        self.num_loops = num_loops
        self.seq_len = seq_len
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Parameter(torch.zeros(1, seq_len, d_model))
        self.loop_block = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * ff_mult,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.token_embedding(input_ids)
        x = x + self.position_embedding[:, : x.size(1), :]
        for _ in range(self.num_loops):
            x = self.loop_block(x)
        x = self.norm(x)
        return self.head(x[:, -1, :])


def _make_batch(batch_size: int, seq_len: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    tokens = torch.randint(0, 2, (batch_size, seq_len), device=device)
    targets = tokens.sum(dim=1).remainder(2)
    return tokens, targets


def _build_optimizer(name: str, lr: float, model: nn.Module) -> torch.optim.Optimizer:
    lowered = name.lower()
    if lowered == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr)
    if lowered == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr)
    if lowered == "sgd":
        return torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    raise ValueError(f"Unsupported optimizer '{name}'. Use one of: adamw, adam, sgd")


def run_optimizer_comparison(
    specs: List[OptimizerSpec],
    *,
    seed: int = 7,
    epochs: int = 3,
    train_steps_per_epoch: int = 40,
    batch_size: int = 64,
    seq_len: int = 24,
    device: str = "cpu",
) -> Dict[str, Dict[str, float]]:
    """Train the same looped model across optimizers and report final metrics."""
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)

    device_obj = torch.device(device)
    results: Dict[str, Dict[str, float]] = {}
    loss_fn = nn.CrossEntropyLoss()

    for spec in specs:
        torch.manual_seed(seed)
        model = LoopedLatentReasoner(seq_len=seq_len).to(device_obj)
        optimizer = _build_optimizer(spec.name, spec.lr, model)

        for _ in range(epochs):
            model.train()
            for _ in range(train_steps_per_epoch):
                inputs, targets = _make_batch(batch_size, seq_len, device_obj)
                logits = model(inputs)
                loss = loss_fn(logits, targets)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

        model.eval()
        with torch.no_grad():
            eval_inputs, eval_targets = _make_batch(1024, seq_len, device_obj)
            eval_logits = model(eval_inputs)
            eval_loss = loss_fn(eval_logits, eval_targets).item()
            eval_pred = eval_logits.argmax(dim=-1)
            eval_acc = (eval_pred == eval_targets).float().mean().item()

        results[spec.name.lower()] = {
            "learning_rate": spec.lr,
            "eval_loss": round(eval_loss, 6),
            "eval_accuracy": round(eval_acc, 6),
        }

    return results
