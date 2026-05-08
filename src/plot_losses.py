"""Plot mean per-epoch loss curves from a training CSV.

Reads a ``epoch,mean_loss_d,mean_loss_g`` (DCGAN) or
``epoch,mean_wasserstein,mean_loss_g,mean_gp`` (WGAN-GP) CSV and saves a PNG.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot training loss curves from a CSV.")
    p.add_argument("--csv", type=Path, required=True, help="Path to loss CSV (DCGAN or WGAN-GP).")
    p.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output image path (PNG).",
    )
    p.add_argument("--title", type=str, default="Training loss")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    with args.csv.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise SystemExit(f"No rows in {args.csv}")

    epochs = [int(r["epoch"]) for r in rows]
    fig, ax = plt.subplots(figsize=(7, 4))

    if "mean_loss_d" in rows[0]:
        ax.plot(epochs, [float(r["mean_loss_d"]) for r in rows], marker="o", label="D loss")
        ax.plot(epochs, [float(r["mean_loss_g"]) for r in rows], marker="o", label="G loss")
    elif "mean_wasserstein" in rows[0]:
        ax.plot(epochs, [float(r["mean_wasserstein"]) for r in rows], marker="o", label="Wasserstein estimate")
        ax.plot(epochs, [float(r["mean_loss_g"]) for r in rows], marker="o", label="G loss")
        ax.plot(epochs, [float(r["mean_gp"]) for r in rows], marker="o", label="Gradient penalty")
    else:
        raise SystemExit(f"Unrecognized columns in {args.csv}: {list(rows[0].keys())}")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Mean loss")
    ax.set_title(args.title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150)
    print(f"Saved loss curve to {args.out}")


if __name__ == "__main__":
    main()
