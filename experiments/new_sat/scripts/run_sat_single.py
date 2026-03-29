#!/usr/bin/env python3
import argparse
import csv
import fcntl
import itertools
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training import train_state
from jax import random

sys.path.insert(0, "/home/blohmp1/diffusion-composition/experiments/new_sat")

from sat.fuzzy_checker import analyze_sampler_sparse
from sat.propositional_diffusion_models import HardcodedScore
from sat.score_composer import compose_formula_with_reference


METRIC_COLUMNS = [
    "method",
    "lambda",
    "instance",
    "d",
    "n_points",
    "n_corr",
    "n_outside",
    "n_wrong",
    "perplexity",
    "num_modes",
    "max_count",
    "frac_corr_all",
    "frac_corr_within",
]

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one SAT diffusion simulation and write one CSV row.")
    parser.add_argument("--ndim", type=int, required=True, help="Dimension d.")
    parser.add_argument(
        "--instance",
        type=str,
        required=True,
        choices=["majority", "xor", "exactly_one"],
        help="SAT instance family.",
    )
    parser.add_argument("--method", type=str, required=True, choices=["dombi", "prob"], help="Composition method.")
    parser.add_argument("--lamb", type=float, required=True, help="Composition lambda.")

    parser.add_argument("--bs", type=int, default=1024, help="Number of particles.")
    parser.add_argument("--sigma", type=float, default=0.5, help="Model sigma.")
    parser.add_argument("--seed", type=int, default=0, help="PRNG seed.")

    parser.add_argument(
        "--output-csv",
        type=str,
        required=True,
        help="Output CSV path for this single run.",
    )
    return parser.parse_args()


def majority_dimacs(ndim: int, k: int = None):
    if k is None:
        k = ndim // 2 + 1
    return [[c + 1 for c in comb] for comb in itertools.combinations(range(ndim), k)]


def xor_dimacs(n):
    clauses = []
    for bits in itertools.product([0, 1], repeat=n):
        if sum(bits) % 2 == 0:
            clause = [-(i + 1) if bits[i] else (i + 1) for i in range(n)]
            clauses.append(clause)
    return clauses


def exactly_one_dimacs(ndim: int):
    return [[i + 1 for i in range(ndim)]] + [[-(i + 1), -(j + 1)] for i in range(ndim) for j in range(i + 1, ndim)]


INSTANCE_BUILDERS = {
    "majority": majority_dimacs,
    "xor": xor_dimacs,
    "exactly_one": exactly_one_dimacs,
}


def propositional_Gausses(ndim: int, bs: int, sigma: float, key):
    model = HardcodedScore(num_out=ndim, sigma=sigma*1.5, which="all", axis=0)
    key, init_key = random.split(key)
    optimizer = optax.adam(learning_rate=2e-4)
    state = train_state.TrainState.create(
        apply_fn=model.apply,
        params=model.init(init_key, np.ones([bs, 1]), np.zeros([bs, ndim])),
        tx=optimizer,
    )
    base_model = state
    dim_models = []
    for axis in range(ndim):
        model = HardcodedScore(num_out=ndim, sigma=sigma, which="one_side", axis=axis, sign=1)
        key, init_key = random.split(key)
        optimizer = optax.adam(learning_rate=2e-4)
        state = train_state.TrainState.create(
            apply_fn=model.apply,
            params=model.init(init_key, np.ones([bs, 1]), np.zeros([bs, ndim])),
            tx=optimizer,
        )
        dim_models.append(state)
    return base_model, dim_models


def main() -> None:
    args = parse_args()

    if args.method == "prob" and args.lamb != 1:
        raise ValueError("For method='prob', lamb must be 1.")

    bs = args.bs
    sigma = args.sigma
    dt = 1e-3
    t_init = 1.0
    n = int(t_init / dt)

    beta_0 = 0.1
    beta_1 = 20.0
    log_alpha = lambda t: -0.5 * t * beta_0 - 0.25 * t**2 * (beta_1 - beta_0)
    log_sigma = lambda t: jnp.log(t)
    dlog_alphadt = jax.grad(lambda t: log_alpha(t).sum())
    beta = lambda t: (1 + 0.5 * t * beta_0 + 0.5 * t**2 * (beta_1 - beta_0))

    key = random.PRNGKey(args.seed)
    instance = INSTANCE_BUILDERS[args.instance](args.ndim)

    t = t_init * jnp.ones((bs, 1))
    key, ikey = random.split(key, num=2)
    x_gen = jnp.zeros((bs, n + 1, args.ndim))
    x_gen = x_gen.at[:, 0, :].set(random.normal(ikey, shape=(bs, args.ndim)))

    base, dim_models = propositional_Gausses(args.ndim, bs, sigma, key)
    for index in range(n):
        x_t = x_gen[:, index, :]
        key, ikey = random.split(key, num=2)
        (scores, _), _, _, _ = compose_formula_with_reference(
            dim_models,
            t,
            x_t,
            instance,
            base,
            ell=args.lamb,
            method=args.method,
            gamma=1,
        )

        dx = -dt * (dlog_alphadt(t) * x_t - 2 * beta(t) * scores)
        dx += jnp.sqrt(2 * jnp.exp(log_sigma(t)) * beta(t) * dt) * random.normal(ikey, shape=(bs, args.ndim))
        x_gen = x_gen.at[:, index + 1, :].set(x_t + dx)
        t -= dt

    report = analyze_sampler_sparse(x_gen[:, -1, :], instance, 2, 0.75, metric="linf")
    anycorr = report.correct_modes_uniformity

    row = {
        "method": args.method,
        "lambda": args.lamb,
        "instance": args.instance,
        "d": report.d,
        "n_points": report.n_points,
        "n_corr": report.correct_modes_uniformity.n if anycorr else 0,
        "n_outside": report.n_points - report.n_within_sigma,
        "n_wrong": report.n_within_sigma - report.correct_modes_uniformity.n if anycorr else 0,
        "perplexity": report.correct_modes_uniformity.perplexity if anycorr else 0,
        "num_modes": report.correct_modes_uniformity.K if anycorr else 0,
        "max_count": max(report.correct_modes_uniformity.counts) if anycorr else 0,
        "frac_corr_all": report.correct_modes_uniformity.n / report.n_points if anycorr else 0,
        "frac_corr_within": report.correct_modes_uniformity.n / report.n_within_sigma if anycorr else 0,
    }

    out_path = Path(args.output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    file_exists = out_path.exists()
    with out_path.open("a", encoding="utf-8", newline="") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        writer = csv.DictWriter(f, fieldnames=METRIC_COLUMNS, delimiter=";")
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)
        f.flush()
        fcntl.flock(f.fileno(), fcntl.LOCK_UN)

    print(f"Wrote single-run result to {out_path}")


if __name__ == "__main__":
    main()
