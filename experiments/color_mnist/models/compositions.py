import torch
import torch.nn as nn
import sys
import os

sys.path.append("/u/84/blohmp1/unix/diffusion-composition/diffusion-composition/")

sys.path.append(os.path.abspath("../../.."))
from library.torch_dombi_composition.types import Formula
from library.torch_score_composer import (
    fkc_compose_formula_with_reference,
    fkc_poe_formula_with_reference,
    multi_composition,
)
from library.torch_dombi_composition.feynman_kac_correction.feynman_kac_dombi_composition import (
    dombi_dimacs_composition,
)

##### BASELINES #####


class BinaryDiffusionComposition(nn.Module):
    """
    Composition of m diffusion models using 2 y variables
    score_models: list of score_model models
    classifier: classifier for classifier guidance
    y_1, y_2: int defining the composition
    guidance_scale" float representing the guidance scaling factor
    """

    def __init__(self, score_models, classifier, y_1, y_2, guidance_scale=1.0):
        super().__init__()
        self.score_models = score_models
        self.m = len(self.score_models)
        self.classifier = classifier
        self.y_1 = y_1
        self.y_2 = y_2
        self.guidance_scale = guidance_scale

        self.input_channels = score_models[0].input_channels

    def classifier_grad(self, x, t):
        x_tmp = torch.clone(x).requires_grad_(True).to(x.device)
        t.requires_grad_(False)
        cls_logprobs_x_t = self.classifier(x_tmp, t)

        grd = torch.zeros(
            (x.shape[0], self.m, self.m), device=x.device
        )  # same shape as cls_logprobs_x_t
        grd[:, self.y_1 - 1, self.y_2 - 1] = 1.0  # column of Jacobian to compute
        cls_logprobs_x_t.backward(gradient=grd, retain_graph=True)
        grad = x_tmp.grad
        grad.requires_grad_(False)

        return grad

    def forward(self, x, t):
        cls_grad = self.classifier_grad(x, t)
        with torch.no_grad():
            scores = []
            for score_model in self.score_models:
                scores.append(score_model(x, t))

            cls_logprobs_x_t = self.classifier(x, t)

            mixture_score = torch.zeros_like(scores[0], device=x.device)
            for i in range(self.m):
                mixture_score += torch.mul(
                    scores[i],
                    torch.sum(torch.exp(cls_logprobs_x_t), dim=2)[:, i].view(
                        -1, 1, 1, 1
                    ),
                )

            composition_score = mixture_score + self.guidance_scale * cls_grad
            return composition_score


class ConditionalDiffusionComposition(nn.Module):
    """
    Composition of m diffusion models using 2 y variables
    score_models: list of score_model models
    classifier: classifier for classifier guidance
    y_1, y_2: int defining the composition
    guidance_scale" float representing the guidance scaling factor
    """

    def __init__(
        self, binary_diffusion, conditional_classifier, y_3, guidance_scale=1.0
    ):
        super().__init__()
        self.binary_diffusion = binary_diffusion
        self.conditional_classifier = conditional_classifier
        self.m = binary_diffusion.m
        self.y_1 = binary_diffusion.y_1
        self.y_2 = binary_diffusion.y_2
        self.y_3 = y_3
        self.guidance_scale = guidance_scale

        self.input_channels = binary_diffusion.input_channels

    def classifier_grad(self, x, t):
        x_tmp = torch.clone(x).requires_grad_(True).to(x.device)
        t.requires_grad_(False)
        cls_logprobs_x_t = self.conditional_classifier(
            x_tmp, t, [self.y_1] * x.shape[0], [self.y_2] * x.shape[0]
        )

        grd = torch.zeros(
            (x.shape[0], self.m), device=x.device
        )  # same shape as cls_logprobs_x_t
        grd[:, self.y_3 - 1] = 1.0  # column of Jacobian to compute
        cls_logprobs_x_t.backward(gradient=grd, retain_graph=True)
        grad = x_tmp.grad
        grad.requires_grad_(False)

        return grad

    def forward(self, x, t):
        binary_score = self.binary_diffusion(x, t)
        cls_grad = self.classifier_grad(x, t)
        return binary_score + cls_grad * self.guidance_scale


##### OUR APPROACH, with FKC #####


class DombiDiffusionComposition(nn.Module):
    """
    Composition of m diffusion models using a CNF clause
    score_models: list of score_model models
    clauses: CNF object () defining the composition
    guidance_scale" float representing the guidance scaling factor (lambda)
    uses a unweighted mixture as the negation reference
    """

    def __init__(
        self,
        score_models,
        clauses: Formula,
        conj_guidance_scale=1.0,
        disj_guidance_scale=1.0,
        gamma: float = 1.0,
        negation_reference="",
    ):
        super().__init__()
        self.score_models = score_models
        self.m = len(self.score_models)
        self.clauses = clauses
        self.conj_guidance_scale = conj_guidance_scale
        self.disj_guidance_scale = disj_guidance_scale
        self.gamma = gamma
        self.input_channels = score_models[0].input_channels

    def _responsibilities(self, lls):
        """
        returns a softmax vector over the log-likelihoods weighted by lambda
        """
        softmax = torch.nn.Softmax(dim=1)
        return softmax(lls * self.guidance_scale)

    def forward(self, x, t, lls: list[torch.Tensor]):
        cls_grad = 1
        with torch.no_grad():
            scores = []
            for score_model in self.score_models:
                scores.append(score_model(x, t))
            scores[0] *= 1
            ############# DOMBI COMPOSITION HERE #############
            reference_score, reference_ll, reference_g = multi_composition(
                scores, [lls[:, i] for i in range(3)], 1
            )
            # reference_score, reference_ll = scores[0], lls[:,0]
            # reference_score, reference_ll = reference_score, reference_ll
            # reference_score = (scores[0]+scores[1]+scores[2])/3
            # reference_ll = (lls[:,0] + lls[:,1] + lls[:,2])/3
            # combo_score, combo_ll, combo_fkc = fkc_compose_formula_with_reference(
            #     scores,
            #     [lls[:, i] for i in range(3)],
            #     self.clauses,
            #     reference_score,
            #     reference_ll,  # *torch.log(torch.Tensor([1/self.guidance_scale]).to(device = lls.device)),
            #     self.conj_guidance_scale,
            #     self.conj_guidance_scale,  # TODO: previously, this was another value!
            #     ref_g=reference_g,
            #     gamma=1,  # TODO: previously, this was gamma
            # )

            score, ll, fkc = dombi_dimacs_composition(
                scores=torch.stack([reference_score] + scores, dim=1),
                log_likelihoods=torch.stack(
                    [reference_ll] + [lls[:, i] for i in range(3)], dim=1
                ),
                inv_temp=-self.conj_guidance_scale * torch.ones_like(reference_ll),
                diffusion_coefficient=torch.ones_like(reference_ll),
                formula=self.clauses,
                component_fk_terms=None,
                drift_divergence=0.0,
            )
            # print(abs(score - combo_score).max())
            # print(abs(ll - combo_ll).max())
            # RESULT: ==============for gamma=1, and conjunctive and disjunctive guidance being equal implementations are THE SAME!!!!!!!!!================
            # composition_score = reference_score + self.guidance_scale * cls_grad
            return scores, score, fkc


class ProductOfExpertsDiffusionComposition(nn.Module):
    """
    Composition of m diffusion models using a CNF clause
    score_models: list of score_model models
    clauses: CNF object () defining the composition
    guidance_scale" float representing the guidance scaling factor (lambda)
    uses a unweighted mixture as the negation reference
    """

    def __init__(
        self,
        score_models,
        clauses: float,
        conj_guidance_scale=1.0,
        disj_guidance_scale=1.0,
        gamma: float = 1.0,
        negation_reference="",
    ):
        super().__init__()
        self.score_models = score_models
        self.m = len(self.score_models)
        self.clauses = clauses
        self.conj_guidance_scale = conj_guidance_scale
        self.disj_guidance_scale = disj_guidance_scale
        self.gamma = gamma
        self.input_channels = score_models[0].input_channels

    def _responsibilities(self, lls):
        """
        returns a softmax vector over the log-likelihoods weighted by lambda
        """
        softmax = torch.nn.Softmax(dim=1)
        return softmax(lls * self.guidance_scale)

    def forward(self, x, t, lls: list[torch.Tensor] | None = None):
        cls_grad = 1
        with torch.no_grad():
            scores = []
            for score_model in self.score_models:
                scores.append(score_model(x, t))
            scores[0] *= 1
            ############# POE COMPOSITION HERE #############
            # reference_score, reference_ll, reference_g = multi_composition(scores,[lls[:,i] for i in range(3)],1)
            # reference_score, reference_ll = scores[0], lls[:,0]
            # reference_score, reference_ll = reference_score, reference_ll

            # NO G terms
            # reference_score = (scores[0]+scores[1]+scores[2])
            # reference_ll = (lls[:,0] + lls[:,1] + lls[:,2])
            combo_score, combo_ll, combo_fkc = fkc_poe_formula_with_reference(
                scores,
                [lls[:, i] for i in range(3)],
                self.clauses,
                #    reference_score,
                #    reference_ll,#*torch.log(torch.Tensor([1/self.guidance_scale]).to(device = lls.device)),
                #    self.conj_guidance_scale,
                #    self.disj_guidance_scale,
                gamma=self.gamma,
            )

            # composition_score = reference_score + self.guidance_scale * cls_grad
            return scores, combo_score, combo_fkc
