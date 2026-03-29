# This file contains code for a general purpose implementation of the ito density estimator
# for denoising diffusion SDEs.
# The general log-density equation is:
# delta_ll = <dx, s> - <f*dt, s> - (div f)dt - 1/2 diff_coef ^ 2 ||s||^2 dt

import math
from typing import TypeAlias
from jaxtyping import Float
import torch
from torch import nn

Array: TypeAlias = torch.Tensor


def _multi_batched_sum(x: Float[Array, "B K ..."]) -> Float[Array, "B K"]:
    return x.reshape(x.shape[0], x.shape[1], -1).sum(2)


def full_ito_sde_step(
    dx: Float[Array, "B *shape"],
    scores: Float[Array, "B K *shape"],
    diffusion_coefficient: Float[Array, "B"],
    dt: Float[Array, "B"],
    base_drift_f: Float[Array, "B *shape"] | None = None,
    drift_divergence: Float[Array, "B"] | None = None,
) -> Float[Array, "B K"]:
    """
    Implements the Ito density estimator for most diffusion SDEs.
    dt is assumed to be nonnegative.
    """

    assert (dt > 0).all(), "time-increments are assumed to be positive"
    inner = _multi_batched_sum(dx.unsqueeze(1) * scores)
    # scale = diffusion_coefficient * torch.sqrt(0.5 * dt)  # [B]
    # extra_dims = (1,) * (scores.ndim - 2)
    # scale = scale.view(scale.shape + extra_dims)
    # scale: [B, K, 1, 1, ..., 1]

    # scale = scale[:, None, *([None] * (scores.ndim - 2))] # broadcast to scores
    # inner -= _multi_batched_sum((scores * scale).pow(2))

    inner -= (
        0.5
        * (diffusion_coefficient.pow(2) * dt)[:, None]
        * _multi_batched_sum(scores.pow(2))
    )

    #     _multi_batched_sum(
    #         (
    #             scores
    #             * (diffusion_coefficient * torch.sqrt(0.5 * dt)).expand_as(scores)
    #         ).pow(2)
    #     )
    # )

    if base_drift_f is not None:
        inner -= _multi_batched_sum(base_drift_f.unsqueeze(1) * scores) * dt[:, None]
    if drift_divergence is not None:
        inner -= (drift_divergence * dt)[:, None]

    return inner


if __name__ == "__main__":

    class GaussianScoreOU(nn.Module):
        """
        OU SDE:
            dX_t = theta (c - X_t) dt + sqrt(2 theta) dW_t

        Marginals:
            X_t ~ N(m_t, I_d),  m_t = c (1 - exp(-theta t))

        Score:
            ∇_x log p_t(x) = m_t - x
        """

        def __init__(self, shift: Float[Array, "*shape"], theta: float = 1.0):
            super().__init__()  # type: ignore
            shift = torch.as_tensor(shift, dtype=torch.float32)
            self.register_buffer("shift", shift)
            self.theta = float(theta)

        def mean_t(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
            """
            m_t = c (1 - exp(-theta t)), broadcast over batch and *shape.
            x: (B, *shape)
            t: (B,)
            returns: (B, *shape)
            """
            shift = self.shift
            while shift.dim() < x.dim():
                shift = shift.unsqueeze(0)  # add batch dim

            expand_dims = x.dim() - 1
            t_reshaped = t.view(-1, *([1] * expand_dims))
            return shift * (1.0 - torch.exp(-self.theta * t_reshaped))

        def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
            """
            Score: ∇_x log p_t(x) = m_t - x
            x: (B, *shape)
            t: (B,)
            returns: (B, *shape)
            """
            m_t = self.mean_t(x, t)
            return m_t - x

        def log_pdf(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
            """
            Analytic log p_t(x) for N(m_t, I_d)
            returns: (B,)
            """
            B = x.shape[0]
            d = int(torch.prod(torch.tensor(x.shape[1:], device=x.device)))
            m_t = self.mean_t(x, t)
            diff = x - m_t
            quad = diff.view(B, -1).pow(2).sum(dim=-1)
            log_norm = -0.5 * d * math.log(2.0 * math.pi)
            return log_norm - 0.5 * quad

    def simulate_ou_with_ito(  # type: ignore
        B: int = 128,
        d: int = 4,
        T: float = 1.0,
        N: int = 100,
        theta: float = 1.0,
        shift_value: float = 1.0,
        device: str = "cpu",
    ):
        """
        1) Simulate OU SDE:
            dX_t = theta (c - X_t) dt + sqrt(2 theta) dW_t
        2) Use full_ito_sde_step to estimate log p_T(X_T)
        and compare to analytic log pdf.
        """
        device = torch.device(device)  # type: ignore

        # Shape is (d,) (vector), shift is c
        shift = torch.full((d,), shift_value, device=device)

        score_model = GaussianScoreOU(shift=shift, theta=theta).to(device)

        # Time grid
        t_grid = torch.linspace(0.0, T, N + 1, device=device)  # N+1 points

        # Initial samples X_0 ~ N(0, I_d)
        x0 = torch.randn(B, d, device=device)
        x = x0.clone()

        # For OU: σ = sqrt(2 θ), constant
        sigma_scalar = math.sqrt(2.0 * theta)

        # Divergence of f(x) = theta (c - x) is -theta * d
        div_f_scalar = -theta * d

        # Prepare log-density estimate: we estimate log p_T(x_T)
        # using log p_0(x_0) + sum Δ log p.
        t0 = t_grid[0].expand(B)  # all zeros
        log_p0 = score_model.log_pdf(x0, t0)  # analytic p_0 = N(0, I)

        log_p_est = torch.zeros(B, 1, device=device)  # K=1

        # Simulate forward in time
        noise_norm = torch.sqrt(torch.prod(torch.tensor(x.shape[1:])))
        t = t_grid[0].expand(B)  # start at t=0
        for n in range(N):
            t_next = t_grid[n + 1].expand(B)
            dt = t_next - t  # scalar (same for all batch), but keep as (B,)
            assert (dt > 0).all()

            # ====================corrector step===================
            for _ in range(10):
                score = score_model(x, t)
                grad_norm: Float[Array, "1"] = torch.sqrt(
                    score.flatten(start_dim=1).pow(2).sum(1)
                ).mean()
                langevin_step_size = 2 * (0.16 * noise_norm / grad_norm) ** 2
                dx_corr = langevin_step_size * score + torch.sqrt(
                    2 * langevin_step_size
                ) * torch.randn_like(x)
                x += dx_corr
            # =====================================================
            # Drift f_t(x) = theta (c - x)
            f = theta * (shift - x)

            # # Euler-Maruyama step
            noise = torch.randn_like(x)
            dx = (
                f * dt.unsqueeze(-1)
                + sigma_scalar * torch.sqrt(dt).unsqueeze(-1) * noise
            )
            x_next = x + dx

            # Score s_t(x): (B, d)
            s = score_model(x, t)  # score at current time/location

            # Prepare inputs for Ito estimator
            scores = s.unsqueeze(1)  # (B, 1, d)  -> K=1

            sigma_batch = sigma_scalar * torch.ones(B, device=device)  # (B,)
            dt_batch = dt  # (B,)
            div_f_batch = div_f_scalar * torch.ones(B, device=device)  # (B,)

            # One Itô increment (B, 1)
            delta_log_p = full_ito_sde_step(
                dx=dx,
                scores=scores,
                diffusion_coefficient=sigma_batch,
                dt=dt_batch,
                base_drift_f=f,
                drift_divergence=div_f_batch,
            )

            log_p_est += delta_log_p  # accumulate

            # Move to next step
            x = x_next
            t = t_next

        # Final analytic log density at T
        tT = t_grid[-1].expand(B)
        log_p_T_analytic = score_model.log_pdf(x, tT)  # (B,)

        # Our estimate: log p_T(x_T) ≈ log p_0(x_0) + sum Δ log p
        log_p_T_est = log_p0.unsqueeze(1) + log_p_est  # (B, 1)
        log_p_T_est = log_p_T_est.squeeze(1)

        # Simple diagnostics
        err = (log_p_T_est - log_p_T_analytic).mean().item()
        mae = (log_p_T_est - log_p_T_analytic).abs().mean().item()
        print(f"Mean error (log p_est - log p_analytic): {err:.4f}")
        print(f"Mean absolute error: {mae:.4f}")

        return {
            "x_T": x,  # final samples
            "log_p_T_est": log_p_T_est,  # estimated log densities
            "log_p_T_analytic": log_p_T_analytic,
        }  # type: ignore

    torch.manual_seed(0)  # type: ignore
    _ = simulate_ou_with_ito(  # type: ignore
        B=1024 * 8,
        d=2,
        T=1.0,
        N=10000,
        theta=1.0,
        shift_value=100.0,
        device="cuda",
    )
    # We observe: low (vanishing) mean error -> estimator seems unbiased
