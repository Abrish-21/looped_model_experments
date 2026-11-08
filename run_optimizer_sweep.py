from __future__ import annotations

import argparse
import json

from looped_model_experiments import OptimizerSpec, run_optimizer_comparison


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare optimizers on a compact looped latent-reasoning model."
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seq-len", type=int, default=24)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument(
        "--optimizers",
        nargs="+",
        default=["adamw", "adam", "sgd"],
        help="Space-separated optimizer names",
    )
    parser.add_argument("--lr", type=float, default=1e-3)

    args = parser.parse_args()
    specs = [OptimizerSpec(name=name, lr=args.lr) for name in args.optimizers]
    results = run_optimizer_comparison(
        specs,
        seed=args.seed,
        epochs=args.epochs,
        train_steps_per_epoch=args.steps,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        device=args.device,
    )
    print(json.dumps(results, indent=2))
