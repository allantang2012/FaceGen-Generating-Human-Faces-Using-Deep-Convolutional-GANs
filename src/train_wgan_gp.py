from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision.utils import save_image

from src.dataset import CelebAFolderDataset
from src.dcgan import Discriminator, Generator, dcgan_weights_init
from src.wgan_gp import gradient_penalty


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train WGAN-GP (same DCGAN backbone, critic + GP).")
    p.add_argument(
        "--celeba-root",
        type=Path,
        required=True,
        help="Directory containing CelebA images (e.g. .../img_align_celeba).",
    )
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=0.0001, help="WGAN-GP paper uses 1e-4.")
    p.add_argument("--beta1", type=float, default=0.0, help="WGAN-GP uses beta1=0 for Adam.")
    p.add_argument("--beta2", type=float, default=0.9)
    p.add_argument("--nz", type=int, default=100)
    p.add_argument("--ngf", type=int, default=64)
    p.add_argument("--ndf", type=int, default=64)
    p.add_argument("--n-critic", type=int, default=5, help="Critic updates per generator step.")
    p.add_argument("--gp-lambda", type=float, default=10.0)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--image-size", type=int, default=64)
    p.add_argument("--out-dir", type=Path, default=Path("outputs/wgan_gp"))
    p.add_argument("--checkpoint-every", type=int, default=0)
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
    loss_csv = logs_dir / "wgan_gp_losses.csv"

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

    opt_g = torch.optim.Adam(net_g.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
    opt_d = torch.optim.Adam(net_d.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))

    fixed_noise = torch.randn(64, args.nz, 1, 1, device=device)

    with loss_csv.open("w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(["epoch", "mean_wasserstein", "mean_loss_g", "mean_gp"])

    for epoch in range(1, args.epochs + 1):
        net_g.train()
        net_d.train()

        sum_w = 0.0
        sum_g = 0.0
        sum_gp = 0.0
        n_batches = 0

        bar = tqdm(loader, desc=f"epoch {epoch}/{args.epochs}")
        for real in bar:
            real = real.to(device, non_blocking=True)
            b = real.size(0)

            # ----- Critic -----
            w_mean = 0.0
            gp_mean = 0.0
            for _ in range(args.n_critic):
                noise = torch.randn(b, args.nz, 1, 1, device=device)
                fake = net_g(noise).detach()

                net_d.zero_grad(set_to_none=True)
                score_real = net_d(real)
                score_fake = net_d(fake)
                gp = gradient_penalty(net_d, real, fake)
                loss_d = score_fake.mean() - score_real.mean() + args.gp_lambda * gp
                loss_d.backward()
                opt_d.step()

                wasserstein = (score_real.mean() - score_fake.mean()).detach()
                w_mean += float(wasserstein)
                gp_mean += float(gp.detach())

            w_mean /= args.n_critic
            gp_mean /= args.n_critic
            sum_w += w_mean
            sum_gp += gp_mean

            # ----- Generator (once per batch) -----
            net_g.zero_grad(set_to_none=True)
            noise = torch.randn(b, args.nz, 1, 1, device=device)
            gen = net_g(noise)
            loss_g = -net_d(gen).mean()
            loss_g.backward()
            opt_g.step()

            sum_g += float(loss_g.detach())
            n_batches += 1
            bar.set_postfix(w=w_mean, g=float(loss_g.detach()), gp=gp_mean)

        avg_w = sum_w / max(n_batches, 1)
        avg_g = sum_g / max(n_batches, 1)
        avg_gp = sum_gp / max(n_batches, 1)

        with loss_csv.open("a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([epoch, avg_w, avg_g, avg_gp])

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
            torch.save(net_g.state_dict(), ckpt_dir / f"wgan_gp_generator_epoch_{epoch:03d}.pt")
            torch.save(net_d.state_dict(), ckpt_dir / f"wgan_gp_critic_epoch_{epoch:03d}.pt")

        print(
            f"epoch {epoch} mean_w={avg_w:.4f} mean_loss_g={avg_g:.4f} mean_gp={avg_gp:.4f} device={device}"
        )

    torch.save(net_g.state_dict(), ckpt_dir / "wgan_gp_generator_final.pt")
    torch.save(net_d.state_dict(), ckpt_dir / "wgan_gp_critic_final.pt")
    print(f"Done. Samples in {samples_dir}, losses in {loss_csv}, weights in {ckpt_dir}")


if __name__ == "__main__":
    main()
