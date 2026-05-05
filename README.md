# FaceGen: Generating Human Faces Using Deep Convolutional GANs

Course image-generation project proposal: train a generative model from scratch on human faces.

## 1. Title

**FaceGen: Generating Human Faces Using Deep Convolutional GANs**

## 2. Image Source

**Dataset:** CelebA (publicly available human face images).

**Preprocessing:**

- Resize to a consistent resolution
- Normalize for training

**Goal:** Generate realistic human face images from random noise using a generative model trained **from scratch** (no fine-tuning from pretrained generative checkpoints).

**Why faces:** Faces have consistent global structure with high local variability, so visual quality and failure modes are easy to observe, and latent space exploration is meaningful.

## 3. Model Architecture

**Family:** Generative Adversarial Network (GAN).

**Baseline:** DCGAN (Radford et al.).

**Optional extension:** WGAN-GP (improved stability vs. vanilla GAN training).

**Components:**

- **Generator:** Transposed convolutions mapping a latent noise vector to an image
- **Discriminator:** Convolutional network classifying real vs. generated images

**Training plan:**

- Train from scratch with adversarial (min–max) objectives
- Monitor stability and convergence (loss curves, sample grids)

**Rationale:** GANs are covered in course material, are a strong fit for natural image generation, and offer a practical balance of difficulty and feasibility.

## 4. Extra Criteria (15%)

Planned extras for full credit on the advanced portion:

| Area | Plan |
|------|------|
| **Hyperparameter tuning** | Learning rate, batch size, latent dimension |
| **Metrics tracking** | Generator/discriminator losses; optional FID |
| **Latent space exploration** | Interpolation between noise vectors; qualitative feature trajectories |
| **Model comparison** | DCGAN vs. WGAN-GP |
| **Gallery output** | Save sample grids across training epochs |

## Repository notes

- **GitHub:** Private repository; collaborators `jdeandria` and `barrieca` must be invited per course instructions.
- **Cadence:** At least one commit per week; prefer small, incremental commits that show process.

## Local layout

```
data/               # CelebA (not committed; download locally)
outputs/samples/  # Saved sample grids each epoch
outputs/logs/       # Loss CSV (local; gitignored except structure)
src/                # Dataset, models, DCGAN / WGAN-GP training, latent interpolation
checkpoints/        # Saved weights (gitignored)
```

## Setup (preview)

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Place the extracted CelebA images under `data/` (for example `data/img_align_celeba/` with `.jpg` files).

### Train DCGAN (baseline)

From the repository root (with your virtual environment activated):

```bash
python -m src.train_dcgan --celeba-root data/img_align_celeba --epochs 5 --batch-size 128
```

This writes per-epoch sample grids to `outputs/samples/`, mean losses to `outputs/logs/dcgan_losses.csv`, and final weights to `checkpoints/`. Use `--checkpoint-every N` to also save periodic checkpoints.

### Train WGAN-GP (extension)

Uses the same generator/critic architecture as DCGAN; default output directory is `outputs/wgan_gp/` so runs do not overwrite DCGAN samples.

```bash
python -m src.train_wgan_gp --celeba-root data/img_align_celeba --epochs 5 --batch-size 64
```

Losses are logged to `outputs/wgan_gp/logs/wgan_gp_losses.csv`. Checkpoints are named `wgan_gp_*_final.pt` under `checkpoints/`.

### Latent interpolation

After you have a trained generator checkpoint:

```bash
python -m src.interpolate_latent --generator-weights checkpoints/generator_final.pt --steps 8
```

**Ideas for later weeks:** FID between real and generated folders, and a small notebook or script to plot DCGAN vs WGAN-GP loss curves side by side.
