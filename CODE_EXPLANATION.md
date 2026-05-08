# Code Explanation

A short walkthrough of each module so the implementation can be reviewed quickly.

## `src/dataset.py` — `CelebAFolderDataset`

A minimal `torch.utils.data.Dataset` that recursively scans a folder for image files (`.jpg`, `.jpeg`, `.png`, `.bmp`), sorts them deterministically, and applies a fixed `torchvision.transforms` pipeline:

1. `Resize(image_size)`
2. `CenterCrop(image_size)`
3. `ToTensor()` — produces a `float` tensor in `[0, 1]`
4. `Normalize(0.5, 0.5)` — shifts the tensor to `[-1, 1]` so it matches the generator's `Tanh` output range.

Choosing a folder-based dataset keeps the loader independent of any specific archive format; for CelebA you point `--celeba-root` at the extracted `img_align_celeba` directory.

## `src/dcgan.py` — DCGAN backbone

Implements the canonical 64×64 DCGAN from Radford et al.

- **`Generator`** maps a latent vector `(B, nz)` (reshaped internally to `(B, nz, 1, 1)`) through five `ConvTranspose2d` blocks with `BatchNorm2d` and `ReLU`, ending with `Tanh` to produce a `(B, 3, 64, 64)` image in `[-1, 1]`.
- **`Discriminator`** mirrors the generator with strided `Conv2d` blocks, `BatchNorm2d` (except on the input), and `LeakyReLU(0.2)`. The final `Conv2d` produces a single logit per image (no sigmoid; the trainer uses `BCEWithLogitsLoss`).
- **`dcgan_weights_init`** applies the paper's initialization (Normal(0, 0.02) on convolutions; Normal(1, 0.02) and zero-bias on BatchNorm), which matters for stability of the baseline.

The same modules are reused for WGAN-GP (the discriminator is treated as a critic, with no sigmoid in either case).

## `src/wgan_gp.py` — gradient penalty

`gradient_penalty(critic, real, fake)` implements the WGAN-GP regularizer:

1. Sample `eps ~ U[0,1]` per example and form `interp = eps * real + (1 - eps) * fake`.
2. Compute critic scores on `interp` and request gradients w.r.t. the input via `torch.autograd.grad(..., create_graph=True)`.
3. Penalize the squared deviation of the gradient norm from 1: `mean((||grad||_2 - 1)^2)`.

`create_graph=True` keeps the penalty differentiable for the critic update.

## `src/train_dcgan.py` — DCGAN trainer

Standard non-saturating GAN training with `BCEWithLogitsLoss`:

- **Discriminator step:** maximize log-prob of real and minimize log-prob of fake. Implemented as `0.5 * (loss_real + loss_fake)` against the constant 1/0 label tensors so the two halves are weighted equally.
- **Generator step:** minimize `BCE(D(G(z)), 1)` (the non-saturating trick — provides stronger gradients early in training than `1 - D(G(z))`).
- **Bookkeeping:** mean per-epoch losses are appended to `outputs/logs/dcgan_losses.csv`. A grid of samples from a **fixed noise** vector is saved every epoch to `outputs/samples/epoch_XXX.png`, so progress is comparable across epochs. Final weights are saved as `checkpoints/generator_final.pt` and `discriminator_final.pt`.
- **CLI hyperparameters:** `--lr`, `--beta1`, `--batch-size`, `--nz`, `--ngf`, `--ndf`, `--epochs`, `--image-size`, `--checkpoint-every`.

## `src/train_wgan_gp.py` — WGAN-GP trainer

Same backbone, different optimization:

- **Critic update** (run `n_critic` times per generator step): minimize `mean(D(fake)) - mean(D(real)) + gp_lambda * gradient_penalty(D, real, fake)`.
- **Generator update** (once per batch): minimize `-mean(D(G(z)))`.
- **Adam** with `lr=1e-4`, `betas=(0.0, 0.9)` per the WGAN-GP paper.
- Logs the **Wasserstein estimate** (`mean(D(real)) - mean(D(fake))`), the generator loss, and the mean gradient-penalty value to `outputs/wgan_gp/logs/wgan_gp_losses.csv`. Sample grids and final weights are written to `outputs/wgan_gp/samples/` and `checkpoints/wgan_gp_*_final.pt`.

## `src/interpolate_latent.py` — latent-space exploration

Loads a saved DCGAN generator, samples two latent vectors `z0` and `z1`, builds `args.steps` linearly interpolated points between them (`alpha * z1 + (1 - alpha) * z0`), runs them through the generator in a single batch, and saves the resulting row of images. Useful for showing that the generator has learned a continuous mapping rather than memorized examples.

## How the pieces fit together

```
CelebAFolderDataset  ->  DataLoader  ->  Generator / Discriminator
                                        ↓
                       (loss + optimizer step in train_*)
                                        ↓
        outputs/samples/*.png  +  outputs/logs/*.csv  +  checkpoints/*.pt
```

A trained generator checkpoint can then be fed to `interpolate_latent.py` for the latent-walk visualization, independently of the training loop.
