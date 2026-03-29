from typing import Callable, Sequence, Tuple, TypeAlias, Union
from jaxtyping import Float
import torch

Lit: TypeAlias = int
Array: TypeAlias = torch.Tensor

# Some Types that give alternating Disjunction, Conjunction with list nesting
Formula: TypeAlias = Union[Lit, Sequence["Formula"]]
Annotated_Formula: TypeAlias = Union[
    Lit, Sequence[Tuple[Float[Array, "B"], "Annotated_Formula"]]
]

FKC_Composition: TypeAlias = Tuple[
    Float[Array, "B *shape"], Float[Array, "B"], Float[Array, "B"]
]
Composition: TypeAlias = Tuple[
    Float[Array, "B *shape"], Float[Array, "B"], Float[Array, "B"] | None
]


NegationFn: TypeAlias = Callable[
    [
        Float[Array, "B 2 *shape"],  # scores
        Float[Array, "B 2"],  # log_likelihoods
        Float[Array, "B"],  # diffusion_coefficient
        Float[Array, "B 2"] | None,  # component_fk_terms
        Float[Array, "B"] | float,  # drift_divergence
    ],
    FKC_Composition,
]


CompositionFn: TypeAlias = Callable[
    [
        Float[Array, "B K *shape"],  # scores
        Float[Array, "B K"],  # log_likelihoods
        Float[Array, "B"],  # inv_temp
        Float[Array, "B"],  # diffusion_coefficient
        Float[Array, "B K"] | None,  # component_fk_terms
        Float[Array, "B"] | float,  # drift_divergence
    ],
    FKC_Composition,
]
