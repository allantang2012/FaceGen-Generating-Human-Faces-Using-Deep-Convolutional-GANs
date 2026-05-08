"""Download a small subset of CelebA face images from a public Hugging Face mirror.

Saves PNG/JPG files into ``data/celeba_subset/`` so training can point at them.
Run from the repo root::

    python -m src.download_celeba_subset --num-images 10000
"""
from __future__ import annotations

import argparse
from pathlib import Path

from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Download a CelebA-faces subset for training.")
    p.add_argument("--num-images", type=int, default=10000)
    p.add_argument("--out-dir", type=Path, default=Path("data/celeba_subset"))
    p.add_argument(
        "--repo",
        type=str,
        default="nielsr/CelebA-faces",
        help="Hugging Face dataset id (default: nielsr/CelebA-faces).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    existing = sum(1 for _ in args.out_dir.iterdir())
    if existing >= args.num_images:
        print(f"{existing} images already in {args.out_dir}; skipping download.")
        return

    from datasets import load_dataset

    ds = load_dataset(args.repo, split="train", streaming=True)
    saved = existing
    bar = tqdm(total=args.num_images, initial=saved, desc="downloading")
    for example in ds:
        if saved >= args.num_images:
            break
        img = example.get("image")
        if img is None:
            continue
        path = args.out_dir / f"face_{saved:06d}.jpg"
        if not path.exists():
            img.convert("RGB").save(path, "JPEG", quality=92)
        saved += 1
        bar.update(1)
    bar.close()

    print(f"Done. {saved} images in {args.out_dir.resolve()}")


if __name__ == "__main__":
    main()
