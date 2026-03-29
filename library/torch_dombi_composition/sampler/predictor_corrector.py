import torch
import tqdm
from library.torch_dombi_composition.density_estimation.ito_density_estimator import (
    full_ito_sde_step,
)
from library.torch_dombi_composition.types import Array, Formula
from library.torch_dombi_composition.feynman_kac_correction.feynman_kac_dombi_composition import (
    dombi_dimacs_composition,
)
from jaxtyping import Float
from typing import Callable, Sequence, Tuple


def pc_sampler_with_likelihoods(
    score_model: torch.nn.Module,
    diffusion_coeff_fkt_marginal: Callable[
        [Float[Array, "batch_size"]], Float[Array, "batch_size"]
    ],
    diffusion_coeff_fkt: Callable[
        [Float[Array, "batch_size"]], Float[Array, "batch_size"]
    ],
    batch_size: int = 64,
    num_steps: int = 500,
    snr: float = 0.16,
    device: str = "cuda",
    eps: float = 1e-3,
    show_progress: bool = True,
    resample: bool = False,
    corrector_steps: int | None = None,
) -> Tuple[
    Float[Array, "batch_size channels 28 28"], Float[Array, "batch_size num_components"]
]:
    """
    Predictor-Corrector sampler that ALSO accumulates per-sample log-likelihood increments (Itô form).

    Returns:
        x_mean:   final noise-free sample (B, C, H, W)
        loglik:   accumulated Δℓ per sample (B,)
    """
    t = torch.ones(batch_size, device=device)
    # init_x ~ N(0, σ(t)^2 I)
    init_x = torch.einsum(
        "B...,B->B...",
        torch.randn(batch_size, score_model.input_channels, 28, 28, device=device),
        diffusion_coeff_fkt_marginal(t),
    )
    time_steps = torch.linspace(1.0, eps, num_steps)
    step_size = time_steps[0] - time_steps[1]
    x = init_x
    lls = torch.zeros((batch_size, score_model.num_components), device=device)

    noise_norm = torch.sqrt(torch.prod(torch.tensor(x.shape[1:])))
    dx_diffusion = dx_flow = 0
    # full predictor-Corrector Sampling Process
    with torch.no_grad():
        for time_step in tqdm.tqdm(time_steps) if show_progress else time_steps:
            batch_time_step = torch.ones(batch_size, device=device) * time_step
            # Corrector step (Langevin MCMC)
            for _ in range(0 if corrector_steps is None else corrector_steps):
                score, component_scores, _ = score_model(x, batch_time_step, lls)
                grad_norm: Float[Array, "1"] = torch.sqrt(
                    score.flatten(start_dim=1).pow(2).sum(1)
                ).mean()
                langevin_step_size = 2 * (snr * noise_norm / grad_norm) ** 2
                dx_corr = langevin_step_size * score + torch.sqrt(
                    2 * langevin_step_size
                ) * torch.randn_like(x)
                x += dx_corr

            # Predictor step (Euler-Maruyama)
            g = diffusion_coeff_fkt(batch_time_step)
            score, component_scores, _ = score_model(x, batch_time_step, lls)
            dx_flow = torch.einsum(
                "B...,B->B...",
                score,
                g.pow(2) * step_size,
            )
            dx_diffusion = dx_flow + torch.sqrt(g**2 * step_size)[
                :, None, None, None
            ] * torch.randn_like(x)

            lls += full_ito_sde_step(
                dx_diffusion,
                component_scores,
                g,
                step_size,
            )
            # TODO: implement fk_correction

            print(lls[0, :])
            x += dx_diffusion
        # The last step does not include any noise
    return x + dx_flow - dx_diffusion, lls

    # ===================================================================


class CompositionScoreModel(torch.nn.Module):
    """
    Composition of m diffusion models using a Composition Formula
    score_models: list of score_model models
    clauses: CNF object () defining the composition
    guidance_scale" float representing the guidance scaling factor (lambda)
    uses a unweighted mixture as the negation reference
    """

    def __init__(
        self,
        score_models: Sequence[torch.nn.Module],
        clauses: Formula,
        inv_temp: float = 1.0,
        gamma: float = 1.0,
        device: "str" = "cuda",
    ):
        super().__init__()  # type: ignore
        self.score_models = score_models
        self.K = len(self.score_models)
        self.clauses = clauses
        self.inv_temp = inv_temp
        self.gamma = gamma
        self.input_channels = score_models[0].input_channels
        self.num_components = self.K
        self.device = device

    def _responsibilities(self, lls: Float[Array, "B K"]) -> Float[Array, "B K"]:
        """
        returns a softmax vector over the log-likelihoods weighted by lambda
        """
        softmax = torch.nn.Softmax(dim=1)
        return softmax(lls * self.inv_temp)

    def _get_scores(
        self, x: Float[Array, "B *shape"], t: Float[Array, "B"]
    ) -> Float[Array, "B K *shape"]:
        return torch.stack([component(x, t) for component in self.score_models], dim=1)

    def _add_reference(
        self, scores: Float[Array, "B K *shape"], component_lls: Float[Array, "B K"]
    ) -> Tuple[Float[Array, "B K+1 *shape"], Float[Array, "B K+1"]]:
        """
        PREpends a refence score and likelihood to scores and component_lls as the conventional mixture
        """
        responsibilities = self._responsibilities(component_lls)
        ref_score = torch.einsum("BK...,BK->B...", scores, responsibilities)
        ref_likelihood = (
            torch.logsumexp(component_lls * self.inv_temp, dim=1) / self.inv_temp
        )  # shape B
        return torch.cat([ref_score.unsqueeze(1), scores], dim=1), torch.cat(
            [ref_likelihood.unsqueeze(1), component_lls], dim=1
        )

    @torch.no_grad()  # type: ignore
    def forward(
        self,
        x: Float[Array, "B *shape"],
        t: Float[Array, "B"],
        component_lls: Float[Array, "B K"],
    ):
        scores = self._get_scores(x, t)
        ############# DOMBI COMPOSITION HERE #############
        based_scores, based_component_lls = self._add_reference(scores, component_lls)

        score, _, fkc = dombi_dimacs_composition(
            scores=based_scores,
            log_likelihoods=based_component_lls,
            inv_temp=-self.inv_temp * torch.ones(x.shape[0], device=self.device),
            diffusion_coefficient=torch.ones(x.shape[0], device=self.device),
            formula=self.clauses,
            component_fk_terms=None,
            drift_divergence=0.0,
        )
        # TODO: use log-liklihoods
        return score, scores, fkc
