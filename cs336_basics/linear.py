from __future__ import annotations

import torch
from torch import Tensor, nn
from jaxtyping import Float


class Linear(nn.Module):
    """
    Minimal Linear layer without bias.

    Interface mirrors nn.Linear (minus bias):
      - in_features: size of the last input dim
      - out_features: size of the last output dim
      - device / dtype: optional parameter placement & dtype

    Parameters
    ----------
    W : (out_features, in_features)
        Stored in memory as W (not transposed) for conventional ordering.

    Expected forward shapes
    -----------------------
    x : (..., in_features) -> y : (..., out_features)

    Notes
    -----
    - Initialize weights with trunc_normal_ as per assignment note.
    - Do not use nn.Linear or torch.nn.functional.linear.
    - You should implement the actual matmul in forward().
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)

        factory_kwargs: dict[str, object] = {}
        if device is not None:
            factory_kwargs["device"] = device
        if dtype is not None:
            factory_kwargs["dtype"] = dtype

        # Store as W (out_features, in_features)
        W = torch.empty((self.out_features, self.in_features), **factory_kwargs)
        # Truncated normal initialization
        nn.init.trunc_normal_(W, mean=0.0, std=0.02, a=-0.04, b=0.04)
        self.W = nn.Parameter(W)

    def forward(self, x: Float[Tensor, "... d_in"]) -> Float[Tensor, "... d_out"]:  # type: ignore[override]
        """
        Apply linear transform without bias.

        Implement y = x @ W.T, reducing over the last input dim (d_in)
        and producing the last output dim (d_out). All leading batch
        dimensions should be preserved.
        """
        return x @ self.W.t()
