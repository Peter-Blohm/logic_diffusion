from typing import Literal
import jax
import jax.numpy as jnp
from flax import linen as nn


# ====== shared math (closed forms) ======


def _a(t):  # mean scale  a(t) = 1 - t
    return 1.0 - t


def _s2(t, sigma):  # variance schedule s^2(t) = (1-t) sigma^2 + t
    return (1.0 - t) * (sigma**2) + t


def _m(t):  # corner magnitude along each axis at time t (== 2 a(t))
    return 2.0 * _a(t)


def _mu_bar_all(x, t, sigma):
    # posterior mean of component for the "all corners" case, elementwise:
    # μ̄_j = 2 a(t) * tanh( [2 a(t) / s^2(t)] * x_j )
    at = _a(t)  # [bs,1]
    s2t = _s2(t, sigma)  # [bs,1]
    return 2.0 * at * jnp.tanh((2.0 * at / s2t) * x)


def _mu_bar_one_side(x, t, sigma, axis: int = 0):
    at = _a(t)
    s2t = _s2(t, sigma)
    mubar = 2.0 * at * jnp.tanh((2.0 * at / s2t) * x)
    twoa = jnp.squeeze(2.0 * at, axis=-1)  # [bs]
    mubar = mubar.at[..., axis].set(twoa)  # force chosen coord to +2a(t)
    return mubar


def _score(x, t, sigma, which: Literal["all", "one_side"], axis: int = 0):
    s2t = _s2(t, sigma)
    mubar = (
        _mu_bar_all(x, t, sigma)
        if which == "all"
        else _mu_bar_one_side(x, t, sigma, axis)
    )
    return (mubar - x) / s2t


def _loglik_one_side(x, t, sigma, axis: int = 0):
    s2t = _s2(t, sigma)
    mt = _m(t)
    # selected (one-sided) coordinate: N(x_axis | +m, s2)
    x_axis = x[..., axis : axis + 1]  # [bs,1]
    ll_axis = -0.5 * jnp.log(2.0 * jnp.pi * s2t) - ((x_axis - mt) ** 2) / (
        2.0 * s2t
    )  # [bs,1]
    # remaining coordinates: symmetric mixtures
    if x.shape[-1] == 1:
        return ll_axis
    x_rest = jnp.concatenate([x[..., :axis], x[..., axis + 1 :]], axis=-1)  # [bs,k-1]
    xrp = -((x_rest - mt) ** 2) / (2.0 * s2t)
    xrm = -((x_rest + mt) ** 2) / (2.0 * s2t)
    per_dim = (
        -0.5 * jnp.log(2.0 * jnp.pi * s2t) + jnp.logaddexp(xrp, xrm) - jnp.log(2.0)
    )
    return ll_axis + jnp.sum(per_dim, axis=-1, keepdims=True)


def _loglik(x, t, sigma, which: Literal["all", "one_side"], axis: int = 0):
    return (
        _loglik_all(x, t, sigma)
        if which == "all"
        else _loglik_one_side(x, t, sigma, axis)
    )


def _loglik_all(x, t, sigma):
    # log p_t(x) for the "all corners" case, using a stable 1D factorization.
    # per-dim: log[ 0.5 N(x|+m,s2) + 0.5 N(x|-m,s2) ]
    # = -0.5*log(2π s2) + logaddexp( -((x-m)^2)/(2s2), -((x+m)^2)/(2s2) ) - log 2
    s2t = _s2(t, sigma)  # [bs,1]
    mt = _m(t)  # [bs,1]
    xmp = -((x - mt) ** 2) / (2.0 * s2t)
    xmm = -((x + mt) ** 2) / (2.0 * s2t)
    per_dim = (
        -0.5 * jnp.log(2.0 * jnp.pi * s2t) + jnp.logaddexp(xmp, xmm) - jnp.log(2.0)
    )
    return jnp.sum(per_dim, axis=-1, keepdims=True)  # [bs,1]


# ====== plug-and-play modules ======


class HardcodedScore(nn.Module):
    """Drop-in replacement for MLP that outputs the score ∇_x log p_t(x)."""

    num_out: int
    sigma: float
    which: Literal["all", "one_side"] = "all"
    axis: int = 0
    sign: Literal[-1, 1] = 1

    @nn.compact
    def __call__(self, t, x):
        # t: [bs,1], x: [bs,ndim]
        score = (
            _score(x * self.sign, t, self.sigma, self.which, self.axis) * self.sign
        )  # [bs, ndim]
        # sanity: make sure user-set num_out matches ndim
        assert (
            score.shape[-1] == self.num_out
        ), f"num_out={self.num_out} but x has ndim={score.shape[-1]}"
        return score

    @nn.compact
    def likelihood(self, t, x):
        # t: [bs,1], x: [bs,ndim]
        ll = _loglik(x * self.sign, t, self.sigma, self.which, self.axis)  # [bs,1]
        return ll


class HardcodedLogLik(nn.Module):
    """Same interface, but returns log p_t(x) in shape [bs,1]."""

    sigma: float
    which: Literal["all", "one_side"] = "all"

    @nn.compact
    def __call__(self, t, x):
        ll = _loglik(x, t, self.sigma, self.which)  # [bs,1]
        assert (
            self.num_out == 1
        ), f"Set num_out=1 for log-likelihood; got {self.num_out}"
        return ll


@jax.jit
def get_sscore(state, t, x, temp=1):
    return state.apply_fn(state.params, t, x) * temp


@jax.jit
def get_ll(state, t, x):
    return state.apply_fn(state.params, t, x, method=HardcodedScore.likelihood)
