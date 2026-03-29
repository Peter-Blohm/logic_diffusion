from typing import List, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor


Lit = int
Clause = List[Lit]
CNF = List[Clause]


def multi_composition(
    sds: List[Tensor], lls: List[Tensor], ell: float, gs: List[Tensor] | None = None
) -> Tuple[Tensor, Tensor, Tensor]:
    # Weighted average of scores using kappas
    sd_stack = torch.stack(sds, dim=0)  # (n, B, D, ...)

    # logits for soft-(max|min) mixing
    logits = torch.stack([ell * ll for ll in lls], dim=0)  # (n, B)
    kappas = F.softmax(logits, dim=0)  # (n, B)

    # reshape kappas to (n, B, 1, 1, ...) to match sd_stack
    expand_dims = sd_stack.ndim - 2
    alpha = kappas.view(*kappas.shape, *([1] * expand_dims))  # (n, B, 1, ...)

    sdlog = torch.sum(alpha * sd_stack, dim=0)  # (B, D, ...)
    ll = torch.logsumexp(logits, dim=0) / ell  # (B,)
    newg = (1 - ell) * (
        (sdlog**2).sum(dim=(1, 2, 3))
        - (kappas * (sd_stack**2).sum(dim=(2, 3, 4))).sum(dim=0)
    )
    if not gs:
        gs = [torch.zeros_like(ll) for ll in lls]
    g = torch.sum(kappas * torch.stack(gs, dim=0), dim=(0))  # (B, D, ...)

    return sdlog, ll, newg + g


def conjunction(
    sds: List[Tensor], lls: List[Tensor], ell: float, gs: List[Tensor] | None = None
) -> Tuple[Tensor, Tensor]:
    assert ell > 0
    return multi_composition(sds, lls, -ell, gs)


def disjunction(
    sds: List[Tensor], lls: List[Tensor], ell: float, gs: List[Tensor] | None = None
) -> Tuple[Tensor, Tensor]:
    assert ell > 0
    return multi_composition(sds, lls, ell, gs)


def fkc_compose_formula_with_reference(
    scores,
    lls,
    clauses: CNF,
    ref_score,
    ref_ll,
    ell_conj: float = 1,
    ell_disj: float = 1,
    gs=None,
    ref_g=None,
    gamma: float = 1,
) -> Tuple[Tensor, Tensor]:

    if not gs:
        gs = [torch.zeros_like(ll) for ll in lls]
    clause_scores: List[Tensor] = []
    clause_lls: List[Tensor] = []
    clause_gs: List[Tensor] = []

    for cl in clauses:
        lit_scores: List[Tensor] = []
        lit_lls: List[Tensor] = []
        lit_gs: List[Tensor] = []
        for lit in cl:
            v = abs(lit)
            score = scores[v - 1]
            ll = lls[v - 1]
            g = gs[v - 1]
            if lit < 0:
                score = (1 + gamma) * ref_score - gamma * score
                ll = (1 + gamma) * ref_ll - gamma * ll
                g = (
                    gamma * (gamma - 1) * ((score - ref_score) ** 2).sum(dim=(1, 2, 3))
                    + 2 * ref_g
                    - g
                )
            lit_scores.append(score)
            lit_lls.append(ll)
            lit_gs.append(g)
        clause_score, clause_ll, clause_g = disjunction(
            lit_scores, lit_lls, ell_disj, gs=lit_gs
        )
        clause_scores.append(clause_score)
        clause_lls.append(clause_ll)
        clause_gs.append(clause_g)

    return conjunction(clause_scores, clause_lls, ell_conj, gs=clause_gs)


def fkc_poe_formula_with_reference(
    scores,
    lls,
    clauses: CNF,
    # ref_score,
    # ref_ll,
    # ell_conj: float = 1,
    # ell_disj: float = 1,
    gs=None,
    ref_g=None,
    gamma: float = 1,
) -> Tuple[Tensor, Tensor]:

    if not gs:
        gs = [torch.zeros_like(ll) for ll in lls]
    clause_scores: List[Tensor] = []
    clause_lls: List[Tensor] = []
    clause_gs: List[Tensor] = []

    for cl in clauses:
        lit_scores: List[Tensor] = []
        lit_lls: List[Tensor] = []
        lit_gs: List[Tensor] = []
        for lit in cl:
            v = abs(lit)
            score = scores[v - 1]
            ll = lls[v - 1]
            g = gs[v - 1]
            if lit < 0:
                score *= -1 / (1 + gamma)
                ll *= -1 / (1 + gamma)
            lit_scores.append(score)
            lit_lls.append(ll)
            lit_gs.append(g)
        clause_score, clause_ll, clause_g = disjunction(
            lit_scores, lit_lls, 1, gs=lit_gs
        )
        clause_scores.append(clause_score)
        clause_lls.append(clause_ll)
        clause_gs.append(clause_g)

    return torch.sum(torch.stack(clause_scores, dim=1), dim=1), 0, 0
