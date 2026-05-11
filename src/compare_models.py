"""Build a side-by-side DCGAN vs WGAN-GP visual comparison.

Loads both trained generators, evaluates them on the **same** noise batch, and
saves a single PNG with two stacked sample grids (DCGAN on top, WGAN-GP below)
and a captioned PNG showing the two loss curves on a shared epoch axis.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch
from torchvision.utils import save_image, make_grid
import matplotlib.pyplot as plt

from src.dcgan import Generator


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare DCGAN and WGAN-GP generators side by side.")
    p.add_argument("--dcgan-weights", type=Path, default=Path("checkpoints/generator_final.pt"))
    p.add_argument(
        "--wgan-weights",
        type=Path,
        default=Path("checkpoints/wgan_gp_generator_final.pt"),
    )
    p.add_argument("--dcgan-csv", type=Path, default=Path("outputs/logs/dcgan_losses.csv"))
    p.add_argument(
        "--wgan-csv",
        type=Path,
        default=Path("outputs/wgan_gp/logs/wgan_gp_losses.csv"),
    )
    p.add_argument("--nz", type=int, default=100)
    p.add_argument("--n-samples", type=int, default=32, help="Samples per model (will be one 8-wide row stack).")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--samples-out",
        type=Path,
        default=Path("outputs/samples/comparison_samples.png"),
    )
    p.add_argument(
        "--loss-out",
        type=Path,
        default=Path("outputs/samples/comparison_loss.png"),
    )
    return p.parse_args()


def load_generator(weights: Path, nz: int, device: torch.device) -> Generator:
    g = Generator(nz=nz).to(device)
    state = torch.load(weights, map_location=device, weights_only=True)
    g.load_state_dict(state)
    g.eval()
    return g


def render_samples(args: argparse.Namespace, device: torch.device) -> None:
    torch.manual_seed(args.seed)
    z = torch.randn(args.n_samples, args.nz, 1, 1, device=device)

    dcgan = load_generator(args.dcgan_weights, args.nz, device)
    wgan = load_generator(args.wgan_weights, args.nz, device)

    with torch.no_grad():
        dcgan_imgs = dcgan(z)
        wgan_imgs = wgan(z)

    dcgan_grid = make_grid(dcgan_imgs, nrow=8, normalize=True, value_range=(-1, 1))
    wgan_grid = make_grid(wgan_imgs, nrow=8, normalize=True, value_range=(-1, 1))
    stacked = torch.cat([dcgan_grid, wgan_grid], dim=1)
    args.samples_out.parent.mkdir(parents=True, exist_ok=True)
    save_image(stacked, args.samples_out)
    print(f"Saved sample comparison to {args.samples_out} (top: DCGAN, bottom: WGAN-GP)")


def render_loss(args: argparse.Namespace) -> None:
    def _read(path: Path, value_col: str) -> tuple[list[int], list[float]]:
        with path.open("r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        return [int(r["epoch"]) for r in rows], [float(r[value_col]) for r in rows]

    dcgan_ep, dcgan_g = _read(args.dcgan_csv, "mean_loss_g")
    wgan_ep, wgan_g = _read(args.wgan_csv, "mean_loss_g")

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(dcgan_ep, dcgan_g, marker="o", label="DCGAN G loss (BCE)")
    ax.plot(wgan_ep, wgan_g, marker="s", label="WGAN-GP G loss (-D(G(z)))")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Generator loss (different scales)")
    ax.set_title("DCGAN vs WGAN-GP — generator loss over epochs")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

    args.loss_out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.loss_out, dpi=150)
    print(f"Saved loss comparison to {args.loss_out}")


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    render_samples(args, device)
    render_loss(args)


if __name__ == "__main__":
    main()
