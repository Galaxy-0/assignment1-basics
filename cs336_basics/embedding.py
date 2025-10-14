from __future__ import annotations

import torch
from torch import Tensor, nn
from jaxtyping import Float, Int


class Embedding(nn.Module):
    """
    Minimal embedding layer implemented from scratch.

    Mirrors the core interface of nn.Embedding (without extras like padding_idx):
      - num_embeddings: vocabulary size
      - embedding_dim: output vector dimension (d_model)
      - device / dtype: optional parameter placement & dtype

    Parameters
    ----------
    W : (num_embeddings, embedding_dim)
        The learnable embedding matrix, stored with d_model as the final dimension.

    Expected forward shapes
    -----------------------
    token_ids : (...,) LongTensor -> embeddings : (..., embedding_dim)

    Notes
    -----
    - Do not use nn.Embedding or torch.nn.functional.embedding.
    - Initialize weights with trunc_normal_ as specified.
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.num_embeddings = int(num_embeddings)
        self.embedding_dim = int(embedding_dim)

        factory_kwargs: dict[str, object] = {}
        if device is not None:
            factory_kwargs["device"] = device
        if dtype is not None:
            factory_kwargs["dtype"] = dtype

        # Embedding weight matrix with shape (vocab_size, d_model)
        W = torch.empty((self.num_embeddings, self.embedding_dim), **factory_kwargs)
        nn.init.trunc_normal_(W, mean=0.0, std=0.02, a=-0.04, b=0.04)
        self.W = nn.Parameter(W)

    def forward(self, token_ids: Int[Tensor, " ..."]) -> Float[Tensor, " ... d_model"]:  # type: ignore[override]
        # Ensure integer indexing dtype
        if token_ids.dtype != torch.long:
            token_ids = token_ids.long()
        return self.W[token_ids]

