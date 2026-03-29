# the code here is mostly copied from this tutorial: https://colab.research.google.com/drive/120kYYBOVa1i0TD85RjlEkFjaWDxSFUx3?usp=sharing

import torch
import numpy as np
import tqdm

import sys
import os

sys.path.append(os.path.abspath("../../.."))
from library.torch_pc_sampler_with_ll import pc_sampler_with_loglik


def pc_sampler2(
    score_model,
    marginal_prob_std,
    diffusion_coeff,
    batch_size=64,
    num_steps=500,
    snr=0.16,
    device="cuda",
    eps=1e-3,
    show_progress=True,
):
    """Generate samples from score-based models with Predictor-Corrector method.

    Args:
    score_model: A PyTorch model that represents the time-dependent score-based model.
    marginal_prob_std: A function that gives the standard deviation
        of the perturbation kernel.
    diffusion_coeff: A function that gives the diffusion coefficient
        of the SDE.
    batch_size: The number of samplers to generate by calling this function once.
    num_steps: The number of sampling steps.
        Equivalent to the number of discretized time steps.
    device: 'cuda' for running on GPUs, and 'cpu' for running on CPUs.
    eps: The smallest time step for numerical stability.

    Returns:
    Samples.
    """
    t = torch.ones(batch_size, device=device)
    init_x = (
        torch.randn(batch_size, score_model.input_channels, 28, 28, device=device)
        * marginal_prob_std(t)[:, None, None, None]
    )
    lls = torch.zeros(batch_size, 3, device=device)
    time_steps = np.linspace(1.0, eps, num_steps)
    step_size = time_steps[0] - time_steps[1]
    x = init_x

    for time_step in tqdm.tqdm(time_steps) if show_progress else time_steps:
        batch_time_step = torch.ones(batch_size, device=device) * time_step
        # Corrector step (Langevin MCMC)
        single_grads, grad = score_model(x, batch_time_step, lls)
        grad_norm = torch.norm(grad.reshape(grad.shape[0], -1), dim=-1).mean()
        noise_norm = np.sqrt(np.prod(x.shape[1:]))
        langevin_step_size = 2 * (snr * noise_norm / grad_norm) ** 2
        dx = langevin_step_size * grad + torch.sqrt(
            2 * langevin_step_size
        ) * torch.randn_like(x)
        # x = x + dx
        # Predictor step (Euler-Maruyama)
        g = diffusion_coeff(batch_time_step)
        single_grads, grad = score_model(x + dx, batch_time_step, lls)
        dx_mean = dx + (g**2)[:, None, None, None] * grad * step_size
        final_dx = dx_mean + torch.sqrt(g**2 * step_size)[
            :, None, None, None
        ] * torch.randn_like(x)

        # print(step_size,dx_final,single_grads[0]*langevin_step_size
        # ito_density_step(step_size, final_dx, )s

        x = x + final_dx
        # ito_density_step(dt,)

    # The last step does not include any noise
    return x + dx_mean


def pc_sampler(*args, **kwargs):
    return pc_sampler_with_loglik(*args, **kwargs)


def pc_trajectory_sampler(
    score_model,
    marginal_prob_std,
    diffusion_coeff,
    batch_size=64,
    num_steps=500,
    snr=0.16,
    device="cuda",
    eps=1e-3,
    show_progress=False,
):
    """Generate samples from score-based models with Predictor-Corrector method.

    Args:
    score_model: A PyTorch model that represents the time-dependent score-based model.
    marginal_prob_std: A function that gives the standard deviation
        of the perturbation kernel.
    diffusion_coeff: A function that gives the diffusion coefficient
        of the SDE.
    batch_size: The number of samplers to generate by calling this function once.
    num_steps: The number of sampling steps.
        Equivalent to the number of discretized time steps.
    device: 'cuda' for running on GPUs, and 'cpu' for running on CPUs.
    eps: The smallest time step for numerical stability.

    Returns:
    Sample trajectories [timestep, batch, channel, x, y] and timesteps []
    """
    t = torch.ones(batch_size, device=device)
    init_x = (
        torch.randn(batch_size, score_model.input_channels, 28, 28, device=device)
        * marginal_prob_std(t)[:, None, None, None]
    )
    time_steps = np.linspace(1.0, eps, num_steps)
    step_size = time_steps[0] - time_steps[1]
    x = init_x
    batch = []
    for time_step in tqdm.tqdm(time_steps) if show_progress else time_steps:
        batch_time_step = torch.ones(batch_size, device=device) * time_step
        # Corrector step (Langevin MCMC)
        grad = score_model(x, batch_time_step)
        grad_norm = torch.norm(grad.reshape(grad.shape[0], -1), dim=-1).mean()
        noise_norm = np.sqrt(np.prod(x.shape[1:]))
        langevin_step_size = 2 * (snr * noise_norm / grad_norm) ** 2
        x = (
            x
            + langevin_step_size * grad
            + torch.sqrt(2 * langevin_step_size) * torch.randn_like(x)
        )

        # Predictor step (Euler-Maruyama)
        g = diffusion_coeff(batch_time_step)
        x_mean = (
            x
            + (g**2)[:, None, None, None] * score_model(x, batch_time_step) * step_size
        )
        x = x_mean + torch.sqrt(g**2 * step_size)[
            :, None, None, None
        ] * torch.randn_like(x)
        batch.append(x)

    # The last step does not include any noise
    return torch.stack(batch), time_steps
