from __future__ import annotations

from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class CelebAFolderDataset(Dataset):
    """Flat or nested folder of face images (e.g. CelebA ``img_align_celeba``)."""

    _EXT = {".jpg", ".jpeg", ".png", ".bmp"}

    def __init__(self, root: str | Path, image_size: int = 64) -> None:
        root = Path(root).expanduser().resolve()
        if not root.is_dir():
            raise FileNotFoundError(f"CelebA root is not a directory: {root}")

        paths = [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in self._EXT]
        paths.sort()
        if not paths:
            raise ValueError(
                f"No images found under {root}. Point --celeba-root at a folder of images "
                "(for example the extracted ``img_align_celeba`` directory)."
            )

        self._paths = paths
        self._transform = transforms.Compose(
            [
                transforms.Resize(image_size),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize(0.5, 0.5),
            ]
        )

    def __len__(self) -> int:
        return len(self._paths)

    def __getitem__(self, index: int):
        path = self._paths[index]
        with Image.open(path) as img:
            rgb = img.convert("RGB")
        return self._transform(rgb)
