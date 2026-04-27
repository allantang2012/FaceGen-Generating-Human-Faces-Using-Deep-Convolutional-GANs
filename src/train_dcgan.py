from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision.utils import save_image

from src.dataset import CelebAFolderDataset
from src.dcgan import Discriminator, Generator, dcgan_weights_init


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train DCGAN on a folder of face images (e.g. CelebA).")
    p.add_argument(
        "--celeba-root",
        type=Path,
        required=True,
        help="Directory containing CelebA images (e.g. .../img_align_celeba).",
    )
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=0.0002)
    p.add_argument("--beta1", type=float, default=0.5)
    p.add_argument("--nz", type=int, default=100, help="Latent dimension.")
    p.add_argument("--ngf", type=int, default=64)
    p.add_argument("--ndf", type=int, default=64)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--image-size", type=int, default=64)
    p.add_argument("--out-dir", type=Path, default=Path("outputs"))
    p.add_argument("--checkpoint-every", type=int, default=0, help="Save weights every N epochs (0=off).")
    return p.parse_args()


def ensure_dirs(out: Path) -> tuple[Path, Path, Path]:
    samples = out / "samples"
    logs = out / "logs"
    ckpt = Path("checkpoints")
    samples.mkdir(parents=True, exist_ok=True)
    logs.mkdir(parents=True, exist_ok=True)
    ckpt.mkdir(parents=True, exist_ok=True)
    return samples, logs, ckpt


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    samples_dir, logs_dir, ckpt_dir = ensure_dirs(args.out_dir)
    loss_csv = logs_dir / "dcgan_losses.csv"

    dataset = CelebAFolderDataset(args.celeba_root, image_size=args.image_size)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        drop_last=True,
        pin_memory=device.type == "cuda",
    )

    net_g = Generator(nz=args.nz, ngf=args.ngf).to(device)
    net_d = Discriminator(ndf=args.ndf).to(device)
    net_g.apply(dcgan_weights_init)
    net_d.apply(dcgan_weights_init)

    criterion = nn.BCEWithLogitsLoss()
    opt_g = torch.optim.Adam(net_g.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
    opt_d = torch.optim.Adam(net_d.parameters(), lr=args.lr, betas=(args.beta1, 0.999))

    fixed_noise = torch.randn(64, args.nz, 1, 1, device=device)

    with loss_csv.open("w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(["epoch", "mean_loss_d", "mean_loss_g"])

    for epoch in range(1, args.epochs + 1):
        net_g.train()
        net_d.train()
        sum_d = 0.0
        sum_g = 0.0
        n_batches = 0

        bar = tqdm(loader, desc=f"epoch {epoch}/{args.epochs}")
        for i, real in enumerate(bar):
            real = real.to(device, non_blocking=True)
            b = real.size(0)

            real_labels = torch.full((b,), 1.0, device=device)
            fake_labels = torch.full((b,), 0.0, device=device)

            # ----- Discriminator -----
            net_d.zero_grad(set_to_none=True)
            out_real = net_d(real)
            loss_d_real = criterion(out_real, real_labels)

            noise = torch.randn(b, args.nz, 1, 1, device=device)
            fake = net_g(noise)
            out_fake = net_d(fake.detach())
            loss_d_fake = criterion(out_fake, fake_labels)

            loss_d = 0.5 * (loss_d_real + loss_d_fake)
            loss_d.backward()
            opt_d.step()

            # ----- Generator -----
            net_g.zero_grad(set_to_none=True)
            out_fake_g = net_d(fake)
            loss_g = criterion(out_fake_g, real_labels)
            loss_g.backward()
            opt_g.step()

            sum_d += float(loss_d.detach())
            sum_g += float(loss_g.detach())
            n_batches += 1
            bar.set_postfix(loss_d=float(loss_d), loss_g=float(loss_g))

        avg_d = sum_d / max(n_batches, 1)
        avg_g = sum_g / max(n_batches, 1)

        with loss_csv.open("a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([epoch, avg_d, avg_g])

        with torch.no_grad():
            net_g.eval()
            grid = net_g(fixed_noise)
            save_image(
                grid,
                samples_dir / f"epoch_{epoch:03d}.png",
                nrow=8,
                normalize=True,
                value_range=(-1, 1),
            )

        if args.checkpoint_every > 0 and epoch % args.checkpoint_every == 0:
            torch.save(net_g.state_dict(), ckpt_dir / f"generator_epoch_{epoch:03d}.pt")
            torch.save(net_d.state_dict(), ckpt_dir / f"discriminator_epoch_{epoch:03d}.pt")

        print(f"epoch {epoch} mean loss_d={avg_d:.4f} mean loss_g={avg_g:.4f} device={device}")

    torch.save(net_g.state_dict(), ckpt_dir / "generator_final.pt")
    torch.save(net_d.state_dict(), ckpt_dir / "discriminator_final.pt")
    print(f"Done. Samples in {samples_dir}, losses in {loss_csv}, weights in {ckpt_dir}")


if __name__ == "__main__":
    main()
