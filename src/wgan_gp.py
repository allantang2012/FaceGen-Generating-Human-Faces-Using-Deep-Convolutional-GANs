from __future__ import annotations

import torch


def gradient_penalty(
    critic: torch.nn.Module,
    real: torch.Tensor,
    fake: torch.Tensor,
) -> torch.Tensor:
    """WGAN-GP term along random interpolations between real and fake batches."""
    b = real.size(0)
    eps = torch.rand(b, 1, 1, 1, device=real.device, dtype=real.dtype)
    interp = eps * real + (1.0 - eps) * fake
    interp = interp.detach().requires_grad_(True)

    scores = critic(interp)
    grad_outputs = torch.ones_like(scores)
    grads = torch.autograd.grad(
        outputs=scores,
        inputs=interp,
        grad_outputs=grad_outputs,
        create_graph=True,
        retain_graph=True,
    )[0]
    grads = grads.reshape(b, -1)
    return ((grads.norm(2, dim=1) - 1.0) ** 2).mean()
