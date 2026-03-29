# This File implements all advanced operations for score composition with FKC
# This includes: ICN style negation, Standard Negation, dombi conjunction and disjunction, as well as an extended DIMACs parsing method.


import functools
import inspect
from typing import Callable, Concatenate, ParamSpec, TypeVar
from jaxtyping import Float
import torch

from library.torch_dombi_composition.feynman_kac_correction.feynman_kac_base_operations import (
    annealed_score_product_with_fkc,
    power_norm_composition_with_fkc,
)
from library.torch_dombi_composition.types import (
    CompositionFn,
    FKC_Composition,
    Lit,
    Formula,
    Array,
    NegationFn,
)


P = ParamSpec("P")
R = TypeVar("R")


def standard_negation_with_fkc(
    scores: Float[Array, "B 2 *shape"],
    log_likelihoods: Float[Array, "B 2"],
    diffusion_coefficient: Float[Array, "B"],
    component_fk_terms: Float[Array, "B 2"] | None = None,
    drift_divergence: Float[Array, "B"] | float = 0.0,
    gammas: Float[Array, "B"] | float = 100.0,
) -> FKC_Composition:
    """
    Implements the standard negation with a reference
    """

    gamma_arr: Float[Array, "B"] = (
        torch.ones(scores.shape[0], device=scores.device) * gammas
    )  # to make typecheckers happy

    exponents = torch.stack((1 + gamma_arr, -gamma_arr), dim=1)
    s, l, f = annealed_score_product_with_fkc(
        scores,
        exponents,
        log_likelihoods,
        diffusion_coefficient,
        component_fk_terms,
        drift_divergence,
    )
    # assert (s == 2 * scores[:, 0] - scores[:, 1]).all()
    # assert (l == 2 * log_likelihoods[:, 0] - log_likelihoods[:, 1]).all()
    return s, l, f


def enforce_temp_sign(func: Callable[P, R], sgn: float = 1.0) -> Callable[P, R]:
    """
    Asserts that the temperature parameter adheres to the specified sgn convention.
    """

    @functools.wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        bound = inspect.signature(func).bind(*args, **kwargs)
        bound.apply_defaults()

        assert (bound.arguments["inv_temp"] > 0) == (
            sgn > 0
        ), f"inv_temp sign needs to be {sgn}"
        return func(*bound.args, **bound.kwargs)

    return wrapper


dombi_disjunction_with_fkc = enforce_temp_sign(power_norm_composition_with_fkc, sgn=1.0)

dombi_conjunction_with_fkc = enforce_temp_sign(
    power_norm_composition_with_fkc, sgn=-1.0
)

# =========================================


def dimacs_composition_with_fkc(
    composition_function: CompositionFn,
    negation_function: NegationFn,
    scores: Float[Array, "B K *shape"],
    log_likelihoods: Float[Array, "B K"],
    inv_temp: Float[Array, "B"],
    diffusion_coefficient: Float[Array, "B"],
    formula: Formula,
    component_fk_terms: Float[Array, "B K"] | None = None,
    drift_divergence: Float[Array, "B"] | float = 0.0,
) -> FKC_Composition:
    """
    Takes a formula in dimacs format and aggregates everything with recursive tree-aggregation.
    This approach is not efficient but easy to understand and general.
    The sign of inv_temp dictates if the formula is in conjunctive form: this happens iff inv_temp < 0.
    """
    # Base case: formula is just a literal
    if isinstance(formula, Lit):
        idx = abs(formula)
        fkc = component_fk_terms
        if formula > 0:  # No negation
            return (
                scores[:, idx, ...],
                log_likelihoods[:, idx, ...],
                (
                    fkc[:, idx, ...]
                    if fkc is not None
                    else torch.zeros(scores.shape[0], device=scores.device)
                ),
            )
        return negation_function(
            scores[:, [0, idx], ...],
            log_likelihoods[:, [0, idx], ...],
            diffusion_coefficient,
            fkc[:, [0, idx], ...] if fkc is not None else None,
            drift_divergence,
        )

    # Recursive case: compose each subformula, stack and then compose the current node.
    child_results = [
        dimacs_composition_with_fkc(
            composition_function,
            negation_function,
            scores,
            log_likelihoods,
            -inv_temp,
            diffusion_coefficient,
            subformula,
            component_fk_terms,
            drift_divergence,
        )
        for subformula in formula
    ]
    children_scores, children_lls, children_fks = zip(*child_results)
    scores_stacked = torch.stack(list(children_scores), dim=1)
    lls_stacked = torch.stack(list(children_lls), dim=1)
    fks_stacked = torch.stack(list(children_fks), dim=1)
    return composition_function(
        scores_stacked,
        lls_stacked,
        inv_temp,
        diffusion_coefficient,
        fks_stacked,
        drift_divergence,
    )


def set_operands(
    func: Callable[Concatenate[CompositionFn, NegationFn, P], R],
    composition_function: CompositionFn,
    negation_function: NegationFn,
) -> Callable[P, R]:
    """
    Wraps func and removes the two arguments composition function and negation function from the signature
    """
    sig = inspect.signature(func)

    @functools.wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        # TODO args binding issue
        bound = sig.bind_partial(*args, **kwargs)
        bound.apply_defaults()
        bound.arguments["composition_function"] = composition_function
        bound.arguments["negation_function"] = negation_function

        return func(*bound.args, **bound.kwargs)

    return wrapper


dombi_dimacs_composition = set_operands(
    dimacs_composition_with_fkc,
    power_norm_composition_with_fkc,
    standard_negation_with_fkc,
)


if __name__ == "__main__":
    print("==============DIMACs Composition Implementation Test....=============")
    from library.torch_dombi_composition.responsibility_composition.tree_responsibility_composition import (
        compose_responsibilities,
        linearize_responsibilities,
        responsibility_weighted_score_composition,
    )

    B = 256
    K = 5
    shape = (3, 4, 5)
    inv_temp = -3.0
    drift_divergence = 0.0
    drift_divergence = torch.randn(1)

    diffusion_coefficient = torch.randn(1)

    scores = torch.randn((B, K, *shape)) * 5
    likelihoods = torch.randn((B, K))
    component_fk_terms = torch.randn((B, K))

    formula: Formula = [[1, 2, 3], [-1, [-2, 1, [3, 4]]], [[3]]]
    # formula: Formula = [[1, [2], 3]]
    # formula: Formula = [[-1]]

    # Test:
    # Check for some formulas if this is equivalent to the responsibility composition
    # Check if some invariants hold: e.g. associativity.
    # Check something in relation to PoE??

    # responsibility method:
    ll_rep, kappa_tree = compose_responsibilities(
        likelihoods=[likelihoods[:, [i]] for i in range(1, K)],
        base_ll=likelihoods[:, [0]],
        inv_temp=inv_temp,
        formula=formula,
    )

    responsibilities = linearize_responsibilities(kappa_tree, K - 1)
    score_rep, fkc_rep = responsibility_weighted_score_composition(
        scores, float(diffusion_coefficient), inv_temp, responsibilities
    )
    ll_rep = ll_rep.squeeze()
    # here:
    score, ll, fkc = dombi_dimacs_composition(
        scores=scores,
        log_likelihoods=likelihoods,
        inv_temp=inv_temp * torch.ones(B),
        diffusion_coefficient=diffusion_coefficient,
        formula=formula,
        component_fk_terms=component_fk_terms,
        drift_divergence=drift_divergence,
    )
    # print(
    #     "score",
    #     scores.sum(),
    #     "ll",
    #     likelihoods[:, formula].sum(),
    #     "fkc",
    #     component_fk_terms[:, formula].sum(),
    # )
    # print(likelihoods[:, 0] * 2 - likelihoods[:, 1] * 1 - ll_rep)

    assert torch.allclose(
        ll_rep, ll
    ), f"Likelihoods Inconsistent: error {(ll_rep-ll).abs().max()}"

    assert torch.allclose(
        score, score_rep, atol=10**-5
    ), f"Scores Inconsistent: error {(score- score_rep).abs().max()}"

    print("WARNING: FKC NOT TESTED")

    print("==============DIMACs Composition Implementations Consistent=============")
