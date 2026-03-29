import torch
import numpy as np
import tqdm


def _ito_density_step_torch(dt, dx, scaled_score, g, f=None, div_f=None):
    """
    Torch version of your JAX ito_density_step.
    dt:        (B,) or ()           time step
    dx:        (B, D)               realized increment
    scaled_score: (B, D)            score s(x_t, t)
    g:         (B,) or ()           diffusion factor
    f:         (B, D) or None       base drift (default 0)
    div_f:     (B,) or () or None   divergence of base drift (default 0)
    returns:   (B,)                 per-sample increment Δℓ
    """
    B = dx.shape[0]

    # Broadcast helpers
    def _to_b1(x):
        if x is None:
            return None
        if x.dim() == 0:
            return x.view(1).expand(B)
        return x

    dt = _to_b1(dt)
    g = _to_b1(g)
    div_f = (
        _to_b1(div_f)
        if div_f is not None
        else torch.zeros(B, device=dx.device, dtype=dx.dtype)
    )
    if f is None:
        f = torch.zeros_like(dx)

    # Δℓ = <dx, s> − <f dt, s> − (div f) dt − ½ g^2 ||s||^2 dt
    # print(scaled_score.shape)
    scaled_score = scaled_score
    # print(scaled_score.shape)
    inner = (dx * scaled_score).sum(dim=1)
    inner -= ((f * dt[:, None]) * scaled_score).sum(dim=1)
    inner -= div_f * dt
    inner -= 0.5 * (g**2) * (scaled_score.pow(2).sum(dim=1)) * dt
    return inner


def resample_cat_systematic(
    weights: torch.Tensor, k: int = 8, g: torch.Generator | None = None
) -> torch.Tensor:
    """
    weights: (B,) log-weights or scores
    k: group size (must divide B)
    returns: (B,) resampled indices, grouped within blocks of size k
    """
    B = weights.numel()
    assert B % k == 0
    groups = B // k
    probs = torch.softmax(weights.view(groups, k), -1)
    cdf = torch.cumsum(probs, -1)

    u0 = torch.rand(groups, 1, generator=g, device=weights.device)
    u = (u0 + torch.arange(k, device=weights.device) / k) % 1
    idx = torch.searchsorted(cdf, u)

    return (idx + (torch.arange(groups, device=weights.device) * k)[:, None]).reshape(
        -1
    )


def pc_sampler_with_loglik(
    # TODO: this is the main part of the experiment code.
    # Use density estimation from the library.
    # Make this run on Cuda.
    score_model,
    marginal_prob_std,
    diffusion_coeff,
    batch_size=64,
    num_steps=500,
    snr=0.16,
    device="cuda",
    eps=1e-3,
    show_progress=True,
    resample=0,
):
    """
    Predictor–Corrector sampler that ALSO accumulates per-sample log-likelihood increments (Itô form).

    Returns:
        x_mean:   final noise-free sample (B, C, H, W)
        loglik:   accumulated Δℓ per sample (B,)
    """
    t = torch.ones(batch_size, device=device)
    # init_x ~ N(0, σ(t)^2 I)
    init_x = (
        torch.randn(batch_size, score_model.input_channels, 28, 28, device=device)
        * marginal_prob_std(t)[:, None, None, None]
    )

    time_steps = np.linspace(1.0, eps, num_steps)
    step_size = float(time_steps[0] - time_steps[1])  # positive
    x = init_x
    loglik = torch.zeros(batch_size, 3, device=device)
    # 4) add terminal prior once at the beginning (t=1)
    with torch.no_grad():
        sigma1 = marginal_prob_std(torch.ones(batch_size, device=device))  # σ(1)
        D = x[0].numel()
        logp_prior = (
            -(init_x.view(batch_size, -1).pow(2).sum(dim=1)) / (2 * sigma1**2)
            - 0.5 * D * np.log(2 * np.pi)
            - D * torch.log(sigma1)
        )
    # add to whichever column corresponds to the score you want
    loglik += logp_prior.to(loglik.dtype)[:, None]
    fkc = torch.zeros(batch_size, device=device)

    iterator = tqdm.tqdm(time_steps) if show_progress else time_steps
    for time_step in iterator:
        batch_time_step = torch.full((batch_size,), float(time_step), device=device)

        # ===== Corrector step (Langevin MCMC) =====
        individual_scores, score, fkc_terms = score_model(
            x, batch_time_step, loglik
        )  # same as 'scaled_score' if you've pre-scaled
        grad = score

        grad_norm = torch.norm(grad.reshape(grad.shape[0], -1), dim=-1).mean()
        noise_norm = np.sqrt(np.prod(x.shape[1:]))
        langevin_step_size = 2 * (snr * noise_norm / (grad_norm + 1e-12)) ** 2  # scalar

        # realize step
        noise = torch.randn_like(x)
        x_new = (
            x
            + langevin_step_size * grad
            + torch.sqrt(torch.tensor(2.0 * langevin_step_size, device=device)) * noise
        )

        # for i, s in enumerate(individual_scores):
        # loglik[:,i] += _ito_density_step_torch(dt, dx, g_corr.view(-1,1)*s.reshape(batch_size, -1), g_corr)
        x = x_new  # advance state

        # ===== Predictor step (Euler–Maruyama for reverse SDE) =====
        g = diffusion_coeff(batch_time_step)  # shape (B,)
        individual_scores, score, fkc_terms = score_model(x, batch_time_step, loglik)
        fkc -= fkc_terms * eps
        x_mean = x + (g**2)[:, None, None, None] * score * step_size
        noise = torch.randn_like(x)
        x_new = x_mean + torch.sqrt((g**2) * step_size)[:, None, None, None] * noise

        # Ito increment for predictor step (VE-SDE base drift f=0):
        dx = (x_new - x).reshape(batch_size, -1)
        dt = torch.full((batch_size,), -step_size, device=device)
        for i, s in enumerate(individual_scores):
            loglik[:, i] += _ito_density_step_torch(
                -dt, dx, s.reshape(batch_size, -1), g
            )

        if 0.5 < time_step < 1 and int(resample) != 0:
            idx = resample_cat_systematic(fkc)
            if (x_new != x_new[idx]).any():
                print("actually resampled!")
            x_new = x_new[idx]
            loglik = loglik[idx]
        print(loglik)
        fkc = torch.zeros_like(fkc)

        #     print("ratio_mean vs target_mean:",
        #   (-(score_model(init_x, torch.ones(batch_size, device=device), loglik)[1].reshape(batch_size,-1) * init_x.reshape(batch_size,-1)).sum(1)
        #    / (init_x.reshape(batch_size,-1).pow(2).sum(1) + 1e-12)).mean().item(),
        #   (1.0 / (marginal_prob_std(torch.ones(batch_size, device=device))**2)).mean().item())

        # if t[0] <= .5:
        #     loglik *= 0
        # loglik -= loglik.min(dim=1)[0].view(-1,1)
        x = x_new  # advance to next time

    # print(loglik)
    # print(loglik.argmax(dim=1))
    return x_mean, loglik
