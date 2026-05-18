"""Compute Fréchet Inception Distance (FID) between real images and GAN samples.

Uses ``torchmetrics.image.fid.FrechetInceptionDistance``. Real images are read
from a folder; fake images are produced by a saved generator checkpoint.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from torchmetrics.image.fid import FrechetInceptionDistance

from src.dataset import CelebAFolderDataset
from src.dcgan import Generator


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compute FID for one or two generator checkpoints.")
    p.add_argument("--real-dir", type=Path, required=True, help="Folder of real face images.")
    p.add_argument(
        "--generator-weights",
        type=Path,
        action="append",
        dest="generators",
        help="Generator checkpoint (.pt). Pass twice to compare DCGAN and WGAN-GP.",
    )
    p.add_argument(
        "--label",
        type=str,
        action="append",
        dest="labels",
        help="Label for each --generator-weights (same order).",
    )
    p.add_argument("--num-samples", type=int, default=2048, help="Images per side of FID.")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--nz", type=int, default=100)
    p.add_argument("--image-size", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--out-csv",
        type=Path,
        default=Path("outputs/logs/fid_scores.csv"),
    )
    return p.parse_args()


def to_fid_input(batch: torch.Tensor) -> torch.Tensor:
    """Map [-1, 1] training tensors to uint8 RGB for torch-fidelity / torchmetrics FID."""
    return ((batch + 1.0) * 127.5).clamp(0, 255).to(torch.uint8)


@torch.no_grad()
def update_real(fid: FrechetInceptionDistance, loader: DataLoader, device: torch.device) -> None:
    for batch in tqdm(loader, desc="real", leave=False):
        imgs = to_fid_input(batch.to(device))
        fid.update(imgs, real=True)


@torch.no_grad()
def fid_for_generator(
    fid: FrechetInceptionDistance,
    weights: Path,
    nz: int,
    n: int,
    batch_size: int,
    device: torch.device,
    seed: int,
) -> float:
    net_g = Generator(nz=nz).to(device)
    state = torch.load(weights, map_location=device, weights_only=True)
    net_g.load_state_dict(state)
    net_g.eval()

    torch.manual_seed(seed)
    remaining = n
    while remaining > 0:
        b = min(batch_size, remaining)
        z = torch.randn(b, nz, 1, 1, device=device)
        fake = to_fid_input(net_g(z))
        fid.update(fake, real=False)
        remaining -= b
    return float(fid.compute().item())


def main() -> None:
    args = parse_args()
    if not args.generators:
        raise SystemExit("Provide at least one --generator-weights path.")

    labels = args.labels or []
    if labels and len(labels) != len(args.generators):
        raise SystemExit("Number of --label values must match --generator-weights.")
    if not labels:
        labels = [p.stem for p in args.generators]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = CelebAFolderDataset(args.real_dir, image_size=args.image_size)
    n = min(args.num_samples, len(dataset))
    indices = list(range(n))
    subset = Subset(dataset, indices)
    loader = DataLoader(
        subset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, str | float | int]] = []

    for weights, label in zip(args.generators, labels):
        fid = FrechetInceptionDistance(normalize=False).to(device)
        update_real(fid, loader, device)
        score = fid_for_generator(
            fid, weights, args.nz, n, args.batch_size, device, args.seed
        )
        print(f"FID [{label}] = {score:.2f}  (n={n}, weights={weights})")
        rows.append(
            {
                "model": label,
                "fid": round(score, 4),
                "num_samples": n,
                "weights": str(weights),
                "real_dir": str(args.real_dir),
            }
        )

    write_header = not args.out_csv.exists()
    with args.out_csv.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        if write_header:
            writer.writeheader()
        writer.writerows(rows)
    print(f"Appended scores to {args.out_csv}")


if __name__ == "__main__":
    main()
