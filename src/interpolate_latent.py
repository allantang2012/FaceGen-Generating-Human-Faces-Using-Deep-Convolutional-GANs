from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torchvision.utils import save_image

from src.dcgan import Generator


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Interpolate between two latent vectors and save a single image row of samples."
    )
    p.add_argument(
        "--generator-weights",
        type=Path,
        required=True,
        help="State dict for the DCGAN generator (e.g. checkpoints/generator_final.pt).",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=Path("outputs/latent_interpolation.png"),
        help="Output image path (PNG).",
    )
    p.add_argument("--nz", type=int, default=100)
    p.add_argument("--ngf", type=int, default=64)
    p.add_argument("--steps", type=int, default=8, help="Number of interpolation steps (including endpoints).")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)

    net_g = Generator(nz=args.nz, ngf=args.ngf).to(device)
    state = torch.load(args.generator_weights, map_location=device)
    net_g.load_state_dict(state)
    net_g.eval()

    z0 = torch.randn(1, args.nz, 1, 1, device=device)
    z1 = torch.randn(1, args.nz, 1, 1, device=device)

    alphas = torch.linspace(0.0, 1.0, steps=args.steps, device=device)
    zs = []
    for a in alphas:
        z = (1.0 - a) * z0 + a * z1
        zs.append(z)
    z_batch = torch.cat(zs, dim=0)

    with torch.no_grad():
        imgs = net_g(z_batch)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    save_image(
        imgs,
        args.out,
        nrow=args.steps,
        normalize=True,
        value_range=(-1, 1),
    )
    print(f"Saved {args.steps}-step interpolation to {args.out}")


if __name__ == "__main__":
    main()
