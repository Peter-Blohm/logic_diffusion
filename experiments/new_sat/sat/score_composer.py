from typing import List


# Hi
import jax
import jax.numpy as jnp
from jax.numpy import ndarray

from sat.propositional_diffusion_models import get_sscore, get_ll

Lit = int
Clause = List[Lit]
CNF = List[Clause]


def binary_composition(
    sd_1: ndarray,
    sd_2: ndarray,
    ll_1: ndarray,
    ll_2: ndarray,
    ell: float,
    bias: float = 0,
) -> (ndarray, ndarray):
    k = jax.nn.softmax(jnp.stack([ell * ll_1 + bias, ell * ll_2]), axis=0)[0][:, None]
    sdlog = sd_2 + k * (sd_1 - sd_2)
    ll = jax.nn.logsumexp(jnp.stack([ell * ll_1, ell * ll_2]), axis=0) / ell
    return sdlog, ll


def multi_composition(
    sds: List[ndarray], lls: List[ndarray], ell: float
) -> (ndarray, ndarray):
    if len(sds) == 1:
        return (
            sds[0],
            lls[0],
            jnp.ones(1),
        )  # no idea if this works, tries to catch nonexistent unary composition
    # logits for soft-(max|min) mixing
    logits = jnp.stack([ell * ll for ll in lls], axis=0)  # (n, B)
    kappas = jax.nn.softmax(logits, axis=0)  # (n, B)

    # Weighted average of scores using kappas
    sd_stack = jnp.stack(sds, axis=0)  # (n, B, D, ...)
    weights = kappas[(...,) + (None,) * (sd_stack.ndim - 3)]  # (n, B, 1, ...)

    sdlog = jnp.sum(weights * sd_stack, axis=0)  # (B, D, ...)
    ll = jax.nn.logsumexp(logits, axis=0) / ell  # (B,)
    # kappas = jax.nn.softmax(jnp.stack([ll * ell for ll in lls]), axis=0)
    # sdlog = jnp.stack([sd * k for sd, k in zip(sds, kappas)]).sum(axis=0)
    # ll = jax.nn.logsumexp(jnp.stack([ll * ell for ll in lls]), axis=0) / ell
    return sdlog, ll, kappas


def conjunction(sds: [ndarray], lls: [ndarray], ell: float) -> (ndarray, ndarray):
    assert ell > 0
    return multi_composition(sds, lls, -ell)

    for lll in lls:
        # print((ll-lll).max())
        assert (
            ll <= lll + 10**-5
        ).all(), "Conjunction is not strictly smaller than minimum"
    # assert ll < lls.max(axis=0)

    return sd, ll
    # return jnp.stack([ell * sd for sd in sds], axis=0).sum(axis=0), jnp.stack([ell * ll for ll in lls], axis=0).sum(axis=0)


def disjunction(sds: [ndarray], lls: [ndarray], ell: float) -> (ndarray, ndarray):
    assert ell > 0
    return multi_composition(sds, lls, ell)


def compose_formula(
    score_models, t, x_t, clauses: CNF, reference, ell: float = 1
) -> (ndarray, ndarray):
    clause_scores = []
    clause_lls = []
    for cl in clauses:
        lit_scores = []
        lit_lls = []

        for lit in cl:
            v = abs(lit)
            sgn = 1 - 2 * (lit < 0)
            model = score_models[v - 1][sgn]
            lit_scores.append(get_sscore(model, t, x_t))
            lit_lls.append(get_ll(model, t, x_t))
        clause_score, clause_ll = disjunction(lit_scores, lit_lls, ell)
        clause_scores.append(clause_score)
        clause_lls.append(clause_ll)

    return conjunction(clause_scores, clause_lls, ell)


def _compose_formula_prob(
    score_models, t, x_t, clauses: CNF, reference, ell: float = 1, gamma: float = 0.5
):
    scores = [get_sscore(model, t, x_t) for model in score_models]
    lls = [get_ll(model, t, x_t) for model in score_models]
    clause_scores = [2 * get_sscore(reference, t, x_t)]
    clause_lls = [get_ll(reference, t, x_t)]
    for cl in clauses:
        lit_scores = []
        lit_lls = []
        for lit in cl:
            v = abs(lit)
            score = scores[v - 1]
            ll = lls[v - 1]
            if lit < 0:
                score = -score * gamma
                ll = -ll * gamma
                
            lit_scores.append(score)
            lit_lls.append(ll)
        clause_score, clause_ll, _ = disjunction(lit_scores, lit_lls, ell)
        clause_scores.append(clause_score)
        clause_lls.append(clause_ll)
    return (
        (
            jnp.sum(jnp.stack(clause_scores, axis=0), axis=0),
            jnp.sum(jnp.stack(clause_lls, axis=0), axis=0),
        ),
        lls,
        clause_lls,
        lls * 0,
    )


# TODO: continue here: find implementation with FKC terms, use in notebook, look at stuck samples.
def compose_formula_with_reference(
    score_models,
    t,
    x_t,
    clauses: CNF,
    reference,
    ell: float = 1,
    method="dombi",
    gamma: float = 0.5,
) -> (ndarray, ndarray):
    assert method == "dombi" or method == "prob"
    if method == "prob":
        return _compose_formula_prob(
            score_models, t, x_t, clauses, reference, ell, gamma
        )
    scores = [get_sscore(model, t, x_t) for model in score_models]
    ref_score = get_sscore(reference, t, x_t)
    lls = [get_ll(model, t, x_t) for model in score_models]
    ref_ll = get_ll(reference, t, x_t)
    clause_kappas = []
    clause_scores = []  # get_sscore(reference, t, x_t)]
    clause_lls = [] # get_ll(reference, t, x_t)]
    for cl in clauses:
        lit_scores = []
        lit_lls = []
        for lit in cl:
            v = abs(lit)
            # model = score_models[v - 1]
            score = scores[v - 1]
            ll = lls[v - 1]
            if lit < 0:
                # The first part of the negation has to be constant for all scores

                score = ref_score + gamma * (ref_score - score)
                ll = ref_ll + gamma * (ref_ll - ll)
            lit_scores.append(score)
            lit_lls.append(ll)
        clause_score, clause_ll, clause_kappa = disjunction(lit_scores, lit_lls, ell)
        clause_scores.append(clause_score)
        clause_lls.append(clause_ll)
        clause_kappas.append(clause_kappa)
    final_score, final_ll, final_kappas = conjunction(clause_scores, clause_lls, ell)

    # for clause in clauses,
    full_k = jnp.zeros((len(x_t), len(score_models)))
    for i, (c, k) in enumerate(zip(clauses, clause_kappas)):
        idxs = (jnp.abs(jnp.array(c)) - 1).astype(int)

        sgns = jnp.array(c) > 0

        # print(final_kappas[i].squeeze())
        # print(jnp.array(k).squeeze())
        # sgns[:,None]

        # print(full_k[:,idxs].shape,((jnp.array(k).squeeze() * sgns[:,None]).T* final_kappas[i]).shape)
        # print(full_k[:,idxs],jnp.array(k), final_kappas[i], full_k[:,idxs] + jnp.array(k) * sgns * final_kappas[i])
        full_k = full_k.at[:, idxs].set(
            full_k[:, idxs] + ((jnp.array(k).squeeze() * 1).T * final_kappas[i])
        )

    return (final_score, final_ll), lls, clause_lls, full_k


# TODO: A version of "compose_formula_with_reference", that can accept literals I force to
