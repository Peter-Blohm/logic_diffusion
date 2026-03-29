# sampler_checker_sparse.py

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Iterable
import numpy as np
import math

# ---------- SAT basics (DIMACS-style CNF) ----------

Lit = int
Clause = List[Lit]
CNF = List[Clause]

def evaluate_cnf_bool_vec(clauses: CNF, model_bits: np.ndarray) -> bool:
    """
    Fast CNF eval on a boolean vector model_bits of shape (nvars,).
    model_bits[i] is truth value of variable (i+1).
    """
    for cl in clauses:
        ok = False
        for lit in cl:
            v = abs(lit)-1
            val = model_bits[v]
            if (lit > 0 and val) or (lit < 0 and not val):
                ok = True
                break
        if not ok:
            return False
    return True

# ---------- Uniformity metrics (over OCCUPIED bins) ----------

@dataclass
class UniformityReport:
    counts: np.ndarray            # counts per occupied bin (shape (K_occ,))
    n: int                        # total counted samples (sum(counts))
    K: int                        # number of occupied bins (K_occ)
    expected: float               # n/K
    chi2: float                   # chi-square statistic vs uniform over occupied bins
    chi2_pvalue: Optional[float]  # None if SciPy not available or invalid
    chi2_valid: bool              # True if expected >= 5 (rule-of-thumb)
    tv_distance: float            # total variation distance
    l_infty: float                # max absolute prob deviation
    entropy_bits: float           # Shannon entropy (base 2)
    entropy_ratio: float          # H / log2(K)
    perplexity: float             # 2^H

def _safe_chi2_pvalue(chi2: float, df: int) -> Optional[float]:
    try:
        from scipy.stats import chi2 as _chi2
        return float(_chi2.sf(chi2, df))
    except Exception:
        return None

def uniformity_report(counts: np.ndarray) -> UniformityReport:
    counts = np.asarray(counts, dtype=float)
    n = int(counts.sum())
    K = counts.size
    if K == 0:
        raise ValueError("counts is empty")
    expected = n / K if K > 0 else math.nan

    if n == 0:
        chi2 = 0.0
        pval = None
        valid = False
    else:
        with np.errstate(divide='ignore', invalid='ignore'):
            chi2 = float(np.nansum((counts - expected) ** 2 / expected)) if expected > 0 else 0.0
        valid = expected >= 5.0
        pval = _safe_chi2_pvalue(chi2, df=K - 1) if (K > 1 and valid) else None

    # Probabilities
    p = counts / n if n > 0 else np.full(K, 1.0 / K)
    u = 1.0 / K
    tv = 0.5 * float(np.sum(np.abs(p - u)))
    linf = float(np.max(np.abs(p - u)))
    # Entropy
    with np.errstate(divide='ignore', invalid='ignore'):
        terms = np.where(p > 0, -p * np.log2(p), 0.0)
    H = float(np.sum(terms))
    H_ratio = H / math.log2(K) if K > 1 else 1.0
    perplexity = 2.0 ** H
    return UniformityReport(
        counts=counts.astype(int),
        n=n,
        K=K,
        expected=expected,
        chi2=chi2,
        chi2_pvalue=pval,
        chi2_valid=valid,
        tv_distance=tv,
        l_infty=linf,
        entropy_bits=H,
        entropy_ratio=H_ratio,
        perplexity=perplexity,
    )

# ---------- Sparse, streaming analyzer ----------

@dataclass
class SamplerCheckResult:
    d: int
    n_points: int
    n_within_sigma: int
    frac_within_sigma: float
    occupied_bins: int
    occupied_correct_bins: int
    collisions: int
    all_modes_uniformity: UniformityReport               # over occupied bins
    correct_modes_uniformity: Optional[UniformityReport] # over occupied & correct bins
    # optional diagnostics
    metric: str
    sigma: float
    used_float32: bool

def _as_mu_vec(mu: float | np.ndarray, d: int, dtype=np.float32) -> np.ndarray:
    if np.isscalar(mu):
        return np.full(d, float(mu), dtype=dtype)
    mu = np.asarray(mu, dtype=dtype)
    if mu.shape != (d,):
        raise ValueError(f"mu must be scalar or shape ({d},), got {mu.shape}")
    return mu


def _pack_sign_bits(batch_points: np.ndarray) -> list[bytes]:
    """
    Pack sign bits (True for >=0) row-wise into immutable bytes keys.
    Returning bytes ensures hashability for use as dict keys.
    """
    bits = (batch_points >= 0)  # (b, d) boolean
    packed = np.packbits(bits, axis=1, bitorder='little')  # (b, ceil(d/8)) uint8
    return [packed[i].tobytes() for i in range(packed.shape[0])]



def analyze_sampler_sparse(
    points: np.ndarray,
    clauses: CNF,
    mu: float | np.ndarray,
    sigma: float,
    metric: str = "l2",                 # "l2" or "linf"
    true_if_positive: bool = True,      # +μ -> True (default)
    batch_max_mb: float = 64.0,         # memory budget for working arrays
    use_float32: bool = True,           # float32 for speed/memory
) -> SamplerCheckResult:
    """
    Sparse K-free analyzer. Streams points, assigns each to its nearest vertex
    via sign(x), checks within-σ distance to that vertex, and counts only the
    occupied bins. CNF is evaluated only for those occupied bins.
    """
    P = np.asarray(points, dtype=np.float32 if use_float32 else np.float64)
    if P.ndim != 2:
        raise ValueError("points must be (N, d)")
    N, d = P.shape
    mu_vec = _as_mu_vec(mu, d, dtype=P.dtype)

    # Choose batch size to respect memory budget roughly: we'll materialize ~3 arrays of size (b, d)
    bytes_per = P.dtype.itemsize
    b_est = max(1, int((batch_max_mb * (1024**2)) / (bytes_per * d * 3)))
    bsz = min(N, max(16, b_est))

    counts: Dict[bytes, int] = {}
    n_within = 0

    for start in range(0, N, bsz):
        stop = min(N, start + bsz)
        B = P[start:stop]  # (b, d)
        # residual to assigned vertex: nearest vertex has coordinate sign = sign(B_ij)
        # residual = B - sgn*mu; with sgn = +1 if B>=0 else -1
        residual = np.where(B >= 0, B - mu_vec, B + mu_vec)

        if metric == "l2":
            dist = np.sqrt(np.einsum("ij,ij->i", residual, residual, optimize=True))
        elif metric == "linf":
            dist = np.max(np.abs(residual), axis=1)
        else:
            raise ValueError("metric must be 'l2' or 'linf'")

        mask = dist <= float(sigma)
        if not mask.any():
            continue

        B_in = B[mask]
        n_in = int(mask.sum())
        n_within += n_in

        # Pack sign bits of the accepted rows to bin keys
        keys = _pack_sign_bits(B_in)  # list[bytes], hashable

        # Count collisions
        for k in keys:
            counts[k] = counts.get(k, 0) + 1

    # Occupied bins summary
    # Occupied bins summary
    # Occupied bins summary
    occ_keys = list(counts.keys())
    if not occ_keys:
        # No points within σ → return an empty-but-valid result
        occupied_bins = 0
        collisions = 0
        all_unif = UniformityReport(
            counts=np.array([], dtype=int),
            n=0,
            K=0,
            expected=float("nan"),
            chi2=0.0,
            chi2_pvalue=None,
            chi2_valid=False,
            tv_distance=0.0,
            l_infty=0.0,
            entropy_bits=0.0,
            entropy_ratio=1.0,
            perplexity=1.0,
        )
        return SamplerCheckResult(
            d=d,
            n_points=N,
            n_within_sigma=n_within,
            frac_within_sigma=(n_within / N) if N else 0.0,
            occupied_bins=occupied_bins,
            occupied_correct_bins=0,
            collisions=collisions,
            all_modes_uniformity=all_unif,
            correct_modes_uniformity=None,
            metric=metric,
            sigma=float(sigma),
            used_float32=use_float32,
        )

    occ_counts = np.array([counts[k] for k in occ_keys], dtype=int)
    occupied_bins = int(occ_counts.size)
    collisions = int(n_within - occupied_bins)

    # Uniformity over occupied bins
    all_unif = uniformity_report(occ_counts)
    #TODO: check if there is a shift bug

    # Evaluate CNF only on occupied bins
    if len(occ_keys) > 0:
        # number of variables used in the formula (max var id)
        nvars = max((abs(l) for cl in clauses for l in cl), default=0)
        # Build correct-bin mask
        correct_mask = np.zeros(occupied_bins, dtype=bool)
        for i, k in enumerate(occ_keys):
            # unpack sign bits for first nvars dims to build assignment
            packed = np.frombuffer(k, dtype=np.uint8)
            bits_full = np.unpackbits(packed, bitorder='little')[:max(nvars, 0)]
            model_bits = bits_full.astype(bool, copy=False)
            if not true_if_positive:
                model_bits = np.logical_not(model_bits)
            if nvars > model_bits.size:
                # If points have fewer dims than variables (shouldn't happen), pad False
                pad = np.zeros(nvars - model_bits.size, dtype=bool)
                model_bits = np.concatenate([model_bits, pad], axis=0)
            correct_mask[i] = evaluate_cnf_bool_vec(clauses, model_bits)

        # Uniformity over occupied & correct bins
        if correct_mask.any():
            counts_correct = occ_counts.copy()
            counts_correct[~correct_mask] = 0
            # Drop zero bins to report over actually occupied-correct bins
            counts_correct = counts_correct[counts_correct > 0]
            corr_unif = uniformity_report(counts_correct)
            occupied_correct_bins = int(counts_correct.size)
        else:
            corr_unif = None
            occupied_correct_bins = 0
    else:
        corr_unif = None
        occupied_correct_bins = 0

    return SamplerCheckResult(
        d=d,
        n_points=N,
        n_within_sigma=n_within,
        frac_within_sigma=(n_within / N) if N else 0.0,
        occupied_bins=occupied_bins,
        occupied_correct_bins=occupied_correct_bins,
        collisions=collisions,
        all_modes_uniformity=all_unif,
        correct_modes_uniformity=corr_unif,
        metric=metric,
        sigma=float(sigma),
        used_float32=use_float32,
    )

# ---------- Pretty-print helper (for OCCUPIED bins) ----------

def summarize_sparse_check(res: SamplerCheckResult) -> str:
    lines = []
    lines.append(f"d={res.d}, N={res.n_points}, metric={res.metric}, σ={res.sigma}, float32={res.used_float32}")
    lines.append(f"Within σ: {res.n_within_sigma} ({res.frac_within_sigma:.3f})")
    lines.append(f"Occupied bins: {res.occupied_bins} (collisions: {res.collisions})")
    u = res.all_modes_uniformity
    lines.append("Uniformity over OCCUPIED bins:")
    lines.append(f"  expected per bin: {u.expected:.3f} (K_occ={u.K})")
    if u.chi2_valid and u.chi2_pvalue is not None:
        lines.append(f"  χ²={u.chi2:.3f}, p={u.chi2_pvalue:.4g}")
    else:
        lines.append(f"  χ²={u.chi2:.3f} (p-value not reported; expected={u.expected:.3f} < 5)")
    lines.append(f"  TV={u.tv_distance:.4f}, L∞={u.l_infty:.4f}, H={u.entropy_bits:.3f} bits "
                 f"({u.entropy_ratio*100:.1f}% of max), perplexity={u.perplexity:.2f}")

    if res.correct_modes_uniformity is not None:
        cu = res.correct_modes_uniformity
        lines.append(f"Uniformity over OCCUPIED & CORRECT bins (K={cu.K}):")
        if cu.chi2_valid and cu.chi2_pvalue is not None:
            lines.append(f"  χ²={cu.chi2:.3f}, p={cu.chi2_pvalue:.4g}")
        else:
            lines.append(f"  χ²={cu.chi2:.3f} (p-value not reported; expected={cu.expected:.3f} < 5)")
        lines.append(f"  TV={cu.tv_distance:.4f}, L∞={cu.l_infty:.4f}, H={cu.entropy_bits:.3f} bits, "
                     f"perplexity={cu.perplexity:.2f}")
    else:
        lines.append("No occupied bins satisfy the CNF (UNSAT on occupied set).")
    return "\n".join(lines)


import numpy as np

clauses = [
    [1, -2, 3],
    [-1, 2],
    [3],
    [4]
]
d = 4
mu = 2.0
sigma = .5
metric = "l2"

rng = np.random.default_rng(0)
sat_modes = [
    (+1, +1, +1, +1),
    (-1, -1, 1, 1),
    (+1, -1, +1, +1),
    (-1, +1, +1, 1),
]
N_per = 250
points = []
for s in sat_modes:
    center = mu * np.array(s, dtype=float)
    cloud = center + rng.normal(0, 0.0, size=(N_per, d))
    points.append(cloud)
points = np.vstack(points)

res = analyze_sampler_sparse(points, clauses, mu=mu, sigma=sigma, metric=metric)
print(summarize_sparse_check(res))


