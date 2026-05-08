# FaceGen: Generating Human Faces Using Deep Convolutional GANs

A course image-generation project: train a GAN **from scratch** on human faces and study how training and the latent space behave.

## 1. Project name and overview

**FaceGen: Generating Human Faces Using Deep Convolutional GANs.**

The model learns to generate 64×64 RGB human faces from random noise. Training is done **from scratch on CelebA** with two objectives implemented on the **same DCGAN backbone**:

- **Baseline:** **DCGAN** with binary cross-entropy adversarial loss.
- **Extension:** **WGAN-GP** (Wasserstein critic + gradient penalty) for more stable training.

The repository ships training scripts for both, a CelebA folder dataset, a latent-space interpolation utility, and per-epoch sample grids so progress is visible in the file system.

## 2. Installation and run instructions

**Requirements:** Python 3.10+ and the packages in `requirements.txt` (PyTorch, torchvision, NumPy, Pillow, tqdm, matplotlib). A CUDA GPU is strongly recommended; CPU works for smoke tests only.

```bash
python -m venv .venv
.venv\Scripts\activate         # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

**Get CelebA** (aligned & cropped) and place the image files locally, for example:

```
data/img_align_celeba/000001.jpg
data/img_align_celeba/000002.jpg
...
```

CelebA is **not** committed; the loader scans whatever directory you pass to `--celeba-root`.

**Train the DCGAN baseline:**

```bash
python -m src.train_dcgan --celeba-root data/img_align_celeba --epochs 5 --batch-size 128
```

Outputs:
- `outputs/samples/epoch_XXX.png` — 8×8 grid of fixed-noise samples per epoch.
- `outputs/logs/dcgan_losses.csv` — mean per-epoch generator and discriminator losses.
- `checkpoints/generator_final.pt`, `checkpoints/discriminator_final.pt`.

**Train the WGAN-GP extension** (separate output directory so DCGAN runs are preserved):

```bash
python -m src.train_wgan_gp --celeba-root data/img_align_celeba --epochs 5 --batch-size 64
```

Outputs land under `outputs/wgan_gp/` and `checkpoints/wgan_gp_*_final.pt`.

**Latent-space interpolation** between two random noise vectors using a saved generator:

```bash
python -m src.interpolate_latent --generator-weights checkpoints/generator_final.pt --steps 8
```

This saves a single image row showing the smooth path from one latent vector to another to `outputs/latent_interpolation.png`.

## 3. Results

A representative sample grid produced by the DCGAN generator on a fixed noise batch is shown below.

![Generated sample grid](outputs/samples/results.png)

> The image is created automatically by `src/train_dcgan.py` each epoch. To regenerate it, run the training command above and copy your favorite epoch grid to `outputs/samples/results.png`. Loss curves live in `outputs/logs/dcgan_losses.csv` (and `outputs/wgan_gp/logs/wgan_gp_losses.csv` for the extension); a quick `matplotlib` plot of those CSVs reproduces the convergence figure.

## 4. Extra criteria pursued

Concretely shipped in this repository:

| Extra criterion | What is implemented |
|---|---|
| **Hyperparameter tuning** | CLI flags for learning rate, batch size, latent dim `nz`, generator/discriminator widths (`ngf`, `ndf`), Adam betas, and (WGAN-GP only) `n_critic` and `gp_lambda`. Every run is fully described by its command line. |
| **Metrics tracking** | Per-epoch mean **generator and discriminator/critic losses** logged to CSV (`dcgan_losses.csv`, `wgan_gp_losses.csv`). For WGAN-GP the **Wasserstein estimate** and **gradient-penalty value** are also tracked. |
| **Latent space exploration** | `src/interpolate_latent.py` performs a linear walk between two random latent vectors and saves a sample row showing how features morph. |
| **Model comparison** | DCGAN vs. WGAN-GP on the **same backbone, dataset, and image size**, so loss curves and sample grids are directly comparable. |
| **Image gallery** | A sample grid is saved **every epoch** under `outputs/samples/` (and `outputs/wgan_gp/samples/`) for an at-a-glance visual log of training progress. |

## 5. Difficulties faced and how they were solved

- **GAN training instability.** Vanilla GAN training oscillates and can mode-collapse. Mitigations: DCGAN architecture conventions (strided convolutions, BatchNorm, LeakyReLU in the discriminator), DCGAN-style weight init (Normal(0, 0.02)), Adam with `beta1=0.5` for the baseline, and an **alternative WGAN-GP** trainer that uses a critic with gradient penalty for a smoother optimization landscape.
- **Comparing DCGAN and WGAN-GP fairly.** Different output folders and CSV log filenames so runs do not overwrite each other (`outputs/samples/` vs. `outputs/wgan_gp/samples/`), while sharing the same generator/discriminator code in `src/dcgan.py`.
- **Logging without bloating commits.** Per-epoch CSV summaries instead of per-batch logs (cheaper I/O, smaller logs) and `.gitignore` rules for `data/`, `checkpoints/`, and generated artifacts.
- **Cross-platform path handling.** Used `pathlib.Path` everywhere and avoided shell-specific commands so the scripts work on both Windows PowerShell and POSIX shells.
- **Process expectations (small commits).** Built incrementally: proposal/layout, then dataset and DCGAN trainer, then WGAN-GP and latent interpolation. The git history shows distinct stages instead of a single end-of-term dump.

## Repository layout

```
data/                       # CelebA (not committed; place locally)
outputs/samples/            # DCGAN sample grids per epoch
outputs/wgan_gp/samples/    # WGAN-GP sample grids per epoch
outputs/logs/               # Loss CSVs (local; gitignored)
src/
  dataset.py                # CelebAFolderDataset (resize, crop, normalize)
  dcgan.py                  # DCGAN Generator + Discriminator + weight init
  wgan_gp.py                # gradient_penalty()
  train_dcgan.py            # DCGAN training entrypoint
  train_wgan_gp.py          # WGAN-GP training entrypoint
  interpolate_latent.py     # latent-space interpolation utility
checkpoints/                # Saved weights (gitignored)
```

See [`CODE_EXPLANATION.md`](CODE_EXPLANATION.md) for a brief module-by-module walkthrough.

## Notes

- **Repo:** Private; collaborators `jdeandria` and `barrieca` are invited per course instructions.
- **Commit cadence:** Weekly small commits to reflect the development process, not a one-shot upload.
