# the code here is mostly copied from this tutorial: https://colab.research.google.com/drive/120kYYBOVa1i0TD85RjlEkFjaWDxSFUx3?usp=sharing

import numpy as np
import torch
import functools
import os, json, argparse, datetime as dt, math
from pathlib import Path
import ast
from torchvision.utils import make_grid, save_image

import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

COLOR_MNIST_DIR = Path(__file__).resolve().parent
CHECKPOINTS_DIR = COLOR_MNIST_DIR / "checkpoints"

from library.torch_dombi_composition.sampler.predictor_corrector import (
    CompositionScoreModel,
    pc_sampler_with_likelihoods,
)
from models.score_model import ScoreNet
from models.compositions import (
    DombiDiffusionComposition,
    ProductOfExpertsDiffusionComposition,
)

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")


def marginal_prob_std(t, sigma):
    """Compute the mean and standard deviation of $p_{0t}(x(t) | x(0))$.

    Args:
    t: A vector of time steps.
    sigma: The $\\sigma$ in our SDE.

    Returns:
    The standard deviation.
    """
    t = torch.tensor(t, device=device)
    return torch.sqrt((sigma ** (2 * t) - 1.0) / 2.0 / np.log(sigma))


def diffusion_coeff(t, sigma):
    """Compute the diffusion coefficient of our SDE.

    Args:
    t: A vector of time steps.
    sigma: The $\\sigma$ in our SDE.

    Returns:
    The vector of diffusion coefficients.
    """
    return torch.tensor(sigma**t, device=device)


sigma = 25.0
marginal_prob_std_fn = functools.partial(marginal_prob_std, sigma=sigma)
diffusion_coeff_fn = functools.partial(diffusion_coeff, sigma=sigma)


def cnf_slug(cnf):
    # [[-3, -2], [3, 2], [-1]] -> "(-3_-2)--(3_2)--(-1)"
    return "--".join("(" + "_".join(str(l) for l in clause) + ")" for clause in cnf)


#
# vis and helper functions
#


def convert_colorblind(X):
    X = X.cpu()
    if X.shape[1] == 1:
        return X

    # colorblind_transform = torch.tensor([[0.83, 0.07, 0.35],[0.1, 0.52, 1.0], [0.0, 0.0, 0.0]])
    colorblind_transform = torch.tensor(
        [
            [225 / 255, 190 / 255, 106 / 255],
            [64 / 255, 176 / 255, 166 / 255],
            [0.0, 0.0, 0.0],
        ]
    )
    Xcb = torch.zeros_like(X)
    for i in range(X.shape[0]):
        for x in range(X.shape[2]):
            for y in range(X.shape[3]):
                Xcb[i, :, x, y] = X[i, :, x, y] @ colorblind_transform
    return Xcb


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)
    return p


def cnf_slug(cnf):
    # [[-3, -2], [3, 2], [-1]] -> "(-3_-2)--(3_2)--(-1)"
    return "--".join("(" + "_".join(str(l) for l in clause) + ")" for clause in cnf)


#
# main code
#


def main():
    ap = argparse.ArgumentParser(
        description="Generate + store samples/LLs for a Dombi CNF composition of color MNIST."
    )
    ap.add_argument(
        "--cnf",
        type=str,
        required=True,
        help="Python list-of-lists, e.g. '[[ -3,-2 ],[ 3,2 ],[ -1 ]]'",
    )
    ap.add_argument("--conj_guidance_scale", type=float, default=5e-3)
    ap.add_argument("--disj_guidance_scale", type=float, default=5e-1)
    ap.add_argument("--gamma", type=float, default=10.0)
    ap.add_argument("--batch_size", type=int, default=64 * 16)
    ap.add_argument("--num_steps", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resample", type=int, default=0)  # NEW
    ap.add_argument("--composition", type=str, default="dombi")  # NEW
    # output
    ap.add_argument("--outdir", type=str, default="runs")
    ap.add_argument("--run_name", type=str, default=None)
    ap.add_argument("--nrow", type=int, default=None)
    ap.add_argument("--no_colorblind_grid", action="store_true")
    ap.add_argument("--save_individual_pngs", action="store_true")
    ap.add_argument("--save_fp16", action="store_true")
    ap.add_argument("--tag", type=str, default=None)  # handy for Slurm arrays
    args = ap.parse_args()

    # minimal CNF handling: just literal_eval the string; no bells/whistles
    cnf_list = ast.literal_eval(args.cnf)  # e.g. [[-3,-2],[3,2],[-1]]
    cnf_str = cnf_slug(cnf_list)  # keep exactly what you passed
    # cnf_hash = hashlib.sha1(cnf_str.encode()).hexdigest()[:10]

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    INPUT_CHANNELS = 3

    score_model1 = ScoreNet(
        marginal_prob_std=marginal_prob_std_fn, input_channels=INPUT_CHANNELS
    )
    score_model1 = score_model1.to(device)
    ckpt1 = torch.load(CHECKPOINTS_DIR / "gen_MN1_ckpt_195.pth", map_location=device)
    score_model1.load_state_dict(ckpt1)
    for param in score_model1.parameters():
        param.requires_grad = False

    score_model2 = ScoreNet(
        marginal_prob_std=marginal_prob_std_fn, input_channels=INPUT_CHANNELS
    )
    score_model2 = score_model2.to(device)
    ckpt2 = torch.load(CHECKPOINTS_DIR / "gen_MN2_ckpt_195.pth", map_location=device)
    score_model2.load_state_dict(ckpt2)
    for param in score_model2.parameters():
        param.requires_grad = False

    score_model3 = ScoreNet(
        marginal_prob_std=marginal_prob_std_fn, input_channels=INPUT_CHANNELS
    )
    score_model3 = score_model3.to(device)
    ckpt3 = torch.load(CHECKPOINTS_DIR / "gen_MN3_ckpt_195.pth", map_location=device)
    score_model3.load_state_dict(ckpt3)
    for param in score_model3.parameters():
        param.requires_grad = False
    if args.composition == "dombi":
        dombiComposition = DombiDiffusionComposition(
            [score_model1, score_model2, score_model3],
            cnf_list,
            conj_guidance_scale=args.conj_guidance_scale,
            disj_guidance_scale=args.disj_guidance_scale,
            gamma=args.gamma,
        )
        dombiComposition2 = CompositionScoreModel(
            [score_model1, score_model2, score_model3],
            cnf_list,
            args.conj_guidance_scale,
            args.gamma,
            device=str(device),
        )
    elif args.composition == "poe":
        dombiComposition = ProductOfExpertsDiffusionComposition(
            [score_model1, score_model2, score_model3],
            cnf_list,
            conj_guidance_scale=args.conj_guidance_scale,
            disj_guidance_scale=args.disj_guidance_scale,
            gamma=args.gamma,
        )
    else:
        raise NotImplementedError

    print("@@@@@@@@@@@@@@@@@@@@@@@", args.resample)
    # samples, lls = pc_sampler(
    #     dombiComposition,
    #     marginal_prob_std_fn,
    #     diffusion_coeff_fn,
    #     args.batch_size,
    #     num_steps=args.num_steps,
    #     device=device,
    #     resample=args.resample,
    # )

    samples, lls = pc_sampler_with_likelihoods(
        dombiComposition2,
        marginal_prob_std_fn,
        diffusion_coeff_fn,
        args.batch_size,
        num_steps=args.num_steps,
        device=str(device),
        resample=args.resample,
    )

    samples = samples.clamp(0.0, 1.0).detach()

    ts = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    slurm_id = os.environ.get("SLURM_JOB_ID")
    parts = [
        f"cnf{cnf_str}",
        f"conj{args.conj_guidance_scale:g}",
        f"disj{args.disj_guidance_scale:g}",
        f"g{args.gamma:g}",
        f"bs{args.batch_size}",
        f"steps{args.num_steps}",
        f"seed{args.seed}",
    ]
    if args.tag:
        parts.append(str(args.tag))
    if slurm_id:
        parts.append(f"slurm{slurm_id}")
    # parts.append(ts)
    run_name = args.run_name or "__".join(parts)

    outdir = ensure_dir(Path(args.outdir) / run_name)

    meta = {
        "cnf_str": cnf_str,
        "conj_guidance_scale": args.conj_guidance_scale,
        "disj_guidance_scale": args.disj_guidance_scale,
        "gamma": args.gamma,
        "batch_size": args.batch_size,
        "num_steps": args.num_steps,
        "seed": args.seed,
        "device": str(device),
        "slurm_job_id": slurm_id,
        "tag": args.tag,
        "timestamp": ts,
    }
    (outdir / "meta.json").write_text(json.dumps(meta, indent=2))

    # save tensors
    samples_to_save = samples.cpu().half() if args.save_fp16 else samples.cpu()
    torch.save(
        samples_to_save,
        outdir / ("samples_fp16.pt" if args.save_fp16 else "samples.pt"),
    )
    torch.save(lls, outdir / "lls.pt")

    # also CSV for quick grep
    header = ",".join([f"model{i+1}" for i in range(lls.shape[1])])
    np.savetxt(
        outdir / "lls.csv", lls.cpu().numpy(), delimiter=",", header=header, comments=""
    )

    # grid (colorblind transform by default; disable with flag)
    nrow = args.nrow or int(math.sqrt(samples.shape[0]))
    grid_src = samples if args.no_colorblind_grid else convert_colorblind(samples)
    grid = make_grid(grid_src, nrow=nrow)
    save_image(grid, outdir / "grid.png")

    # optional individual PNGs
    if args.save_individual_pngs:
        imgdir = ensure_dir(outdir / "images")
        for i in range(samples.shape[0]):
            save_image(samples[i], imgdir / f"img_{i:05d}.png")

    print(f"[OK] Saved to: {outdir.resolve()}")


if __name__ == "__main__":
    main()
