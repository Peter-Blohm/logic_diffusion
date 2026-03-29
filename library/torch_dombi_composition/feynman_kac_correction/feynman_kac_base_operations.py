# This File implements all the base operations for FK correction in diffusion.
# This includes: temperature scaling, products, mixtures, power-norms
# More complex operations, like negation and dombi composition will NOT be implemented in this file.

from typing import TypeAlias
from jaxtyping import Float
import torch
from torch.nn import functional as F

from library.torch_dombi_composition.types import FKC_Composition

Array: TypeAlias = torch.Tensor


def annealing_with_fkc(
    scores: Float[Array, "B *shape"],
    log_likelihoods: Float[Array, "B"],
    exponents: Float[Array, "B"],
    diffusion_coefficient: Float[Array, "B"],
    component_fk_terms: Float[Array, "B"] | None = None,
    # base_drift_f: Float[Array, "B *shape"] | None = None,
    drift_divergence: Float[Array, "B"] | None = None,
) -> FKC_Composition:
    """
    Implements temperature scaling, see Skreta et. al. Prop. D.1
    """
    target_scaled_scores = torch.einsum("B...,B->B...", scores, exponents)
    log_likelihood = log_likelihoods * exponents
    weights = (exponents - 1) * (
        0.5 * exponents * diffusion_coefficient.pow(2) * _batched_sum(scores.pow(2))
    )
    if drift_divergence is not None:
        weights += (exponents - 1) * drift_divergence
    if component_fk_terms is not None:
        weights += exponents * component_fk_terms

    return target_scaled_scores, log_likelihood, weights


def _batched_sum(x: Float[Array, "B ..."]) -> Float[Array, "B"]:
    return x.reshape(x.shape[0], -1).sum(1)


def _multi_batched_sum(x: Float[Array, "B K ..."]) -> Float[Array, "B K"]:
    return x.reshape(x.shape[0], x.shape[1], -1).sum(2)


def annealed_score_product_with_fkc(
    scores: Float[Array, "B K *shape"],
    exponents: Float[Array, "B K"],
    log_likelihoods: Float[Array, "B K"],
    diffusion_coefficient: Float[Array, "B"],
    component_fk_terms: Float[Array, "B K"] | None = None,
    drift_divergence: Float[Array, "B"] | float = 0.0,
) -> FKC_Composition:
    """
    Implements products of K scores, see D.5 in Skreta et. al.
    """
    exponent_sum = exponents.sum(1)  # shape B

    score = torch.einsum("BK...,BK->B...", scores, exponents)  # shape B *shape
    # shape B *shape
    log_likelihood = (log_likelihoods * exponents).sum(1)  # shape B
    fk_terms = 0.5 * diffusion_coefficient.pow(2) * (
        _batched_sum(score.pow(2))
        - (_multi_batched_sum(scores.pow(2)) * exponents).sum(1)
    ) + drift_divergence * (exponent_sum - 1.0)
    if component_fk_terms is not None:
        fk_terms += component_fk_terms.sum(1)  # shape B
    return (score, log_likelihood, fk_terms)


def unweighted_mixture_with_fkc(
    scores: Float[Array, "B K *shape"],
    log_likelihoods: Float[Array, "B K"],
    component_fk_terms: Float[Array, "B K"] | None = None,
) -> FKC_Composition:
    """
    Implements a unweighted mixture of K scores
    """
    responsibilities = F.softmax(log_likelihoods, dim=1)  # shape B K

    score = torch.einsum("BK...,BK->B...", scores, responsibilities)  # shape B *shape
    log_likelihood = torch.logsumexp(log_likelihoods, dim=1)  # shape B
    fk_terms = (component_fk_terms * responsibilities).sum(1)  # shape B

    return score, log_likelihood, fk_terms


def power_norm_composition_with_fkc(
    scores: Float[Array, "B K *shape"],
    log_likelihoods: Float[Array, "B K"],
    inv_temp: Float[Array, "B"],
    diffusion_coefficient: Float[Array, "B"],
    component_fk_terms: Float[Array, "B K"] | None = None,
    drift_divergence: Float[Array, "B"] | float = 0.0,
) -> FKC_Composition:
    """
    Implements the power-norm with the exponent inv_temp
    """
    responsibilities = F.softmax(
        log_likelihoods * inv_temp[:, None], dim=1
    )  # shape B K
    score = torch.einsum("BK...,BK->B...", scores, responsibilities)  # shape [B *shape]

    log_likelihood = (
        torch.logsumexp(log_likelihoods * inv_temp[:, None], dim=1) / inv_temp
    )  # shape [B]
    fk_term = (
        (1 - inv_temp)
        * 0.5
        * diffusion_coefficient.pow(2)
        * (
            _batched_sum(score.pow(2))  # shape [B]
            - (responsibilities * _multi_batched_sum(scores.pow(2))).sum(
                dim=1
            )  # shape [B]
        )
    )

    if component_fk_terms is not None:
        fk_term += (component_fk_terms * responsibilities).sum(1)  # shape [B]

    return score, log_likelihood, fk_term


if __name__ == "__main__":

    B = 256
    K = 5
    shape = (3, 4, 5)
    inv_temp = 3.0
    drift_divergence = None
    drift_divergence = torch.randn(1)

    diffusion_coefficient = torch.randn(1)

    scores = torch.randn((B, K, *shape))
    likelihoods = torch.randn((B, K))
    component_fk_terms = torch.randn((B, K))

    power_score, power_ll, power_fkc = power_norm_composition_with_fkc(
        scores,
        likelihoods,
        inv_temp * torch.ones(B),
        diffusion_coefficient * torch.ones(B),
        component_fk_terms,
    )

    # simulate power norm with: annealing, mixture, annealing

    # anneal:
    annealed_scores, annealed_lls, annealed_fkc = annealing_with_fkc(
        scores.reshape((B * K, *shape)),
        likelihoods.reshape((B * K)),
        torch.ones(B * K) * inv_temp,
        diffusion_coefficient * torch.ones(B * K),
        component_fk_terms.reshape((B * K)),
        drift_divergence=drift_divergence * torch.ones(B * K),
    )
    annealed_scores, annealed_lls, annealed_fkc = (
        annealed_scores.reshape((B, K, *shape)),
        annealed_lls.reshape((B, K)),
        annealed_fkc.reshape((B, K)),
    )

    # mix
    mix_score, mix_ll, mix_fkc = unweighted_mixture_with_fkc(
        annealed_scores, annealed_lls, annealed_fkc
    )
    assert (mix_score.shape, mix_ll.shape, mix_fkc.shape) == (
        torch.Size([B, *shape]),
        torch.Size([B]),
        torch.Size([B]),
    ), "Malformed outputs"
    # un-anneal:
    final_score, final_ll, final_fkc = annealing_with_fkc(
        mix_score,
        mix_ll,
        torch.ones(B) / inv_temp,
        diffusion_coefficient * torch.ones(B),
        mix_fkc,
        drift_divergence=drift_divergence * torch.ones(B),
    )

    print(abs(power_fkc - final_fkc).max())

    assert torch.allclose(
        power_score, final_score, atol=10**-7
    ), "Power Norm Score Inconsistent"
    assert torch.allclose(
        power_ll, final_ll, atol=10**-7
    ), "Power Norm likelihood Inconsistent"
    assert torch.allclose(
        power_fkc, final_fkc, atol=5 * 10**-5
    ), "Power Norm FKC Inconsistent"

    print("==============Power Norm Implementation Consistent=============")
