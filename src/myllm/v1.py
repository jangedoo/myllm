import math

import torch

from myllm.nn import rope


class TransformerWithRoPE(torch.nn.Module):
    def __init__(self, embed_dim: int, nheads: int, dim_ff: int, theta: float):
        super().__init__()
        self.norm1 = torch.nn.LayerNorm(embed_dim)
        self.norm2 = torch.nn.LayerNorm(embed_dim)
        self.mha = rope.RoPEAttention(embed_dim=embed_dim, nheads=nheads, theta=theta)
        self.ff = torch.nn.Sequential(
            torch.nn.Linear(embed_dim, dim_ff),
            torch.nn.GELU(),
            torch.nn.Linear(dim_ff, embed_dim),
        )

    def forward(self, x, key_padding_mask=None, attn_mask=None, position_ids=None):
        residual = x
        x = self.norm1(x)
        attn_output = self.mha(
            x,
            key_padding_mask=key_padding_mask,
            attn_mask=attn_mask,
            position_ids=position_ids,
        )
        x = residual + attn_output

        residual = x
        return residual + self.ff(self.norm2(x))

    @classmethod
    def get_position_ids(cls, bs, seq_len, device, key_padding_mask=None):
        return rope.get_position_ids(
            bs=bs, seq_len=seq_len, device=device, key_padding_mask=key_padding_mask
        )


class MyLLMV1(torch.nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        nlayers: int,
        nheads: int,
        dim_ff: int,
        padding_idx: int | None,
        rope_theta: float,
        init_std: float = 0.02,
        tie_embeddings: bool = True,
    ):
        super().__init__()
        self.init_std = init_std
        self.nlayers = nlayers
        self.embed_dim = embed_dim

        self.emb_layer = torch.nn.Embedding(
            num_embeddings=vocab_size, embedding_dim=embed_dim, padding_idx=padding_idx
        )
        self.transformers = torch.nn.ModuleList(
            [
                TransformerWithRoPE(
                    embed_dim=embed_dim, nheads=nheads, dim_ff=dim_ff, theta=rope_theta
                )
                for _ in range(nlayers)
            ]
        )
        self.norm = torch.nn.LayerNorm(embed_dim)
        self.lm_head = torch.nn.Linear(
            in_features=embed_dim, out_features=vocab_size, bias=False
        )

        # initialize all modules (done recursively)
        self.apply(self._init_weights)

        # residual output projections
        self._scale_residual_projections()

        if tie_embeddings:
            # tie the embedding weights to lm_head weights after weight initialization
            self.lm_head.weight = self.emb_layer.weight

    def _init_weights(self, module: torch.nn.Module):
        if isinstance(module, torch.nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.init_std)

        elif isinstance(module, torch.nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.init_std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

        elif isinstance(module, torch.nn.LayerNorm) and module.elementwise_affine:
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)

    def _scale_residual_projections(self):
        residual_std = self.init_std / math.sqrt(2 * self.nlayers)
        for block in self.transformers:
            # output of attention branch, before adding residual
            torch.nn.init.normal_(block.mha.out_proj.weight, mean=0.0, std=residual_std)
            # output of FFN
            torch.nn.init.normal_(block.ff[-1].weight, mean=0.0, std=residual_std)

    def forward(self, token_ids, padding_mask=None, position_ids=None, **kwargs):
        bs, seq_len = token_ids.shape
        embeddings = self.emb_layer(token_ids)  # bs, seq_len, embed_dim

        if position_ids is None:
            position_ids = TransformerWithRoPE.get_position_ids(
                bs=bs,
                seq_len=seq_len,
                key_padding_mask=padding_mask,
                device=embeddings.device,
            )
        causal_mask = torch.nn.Transformer.generate_square_subsequent_mask(
            seq_len, device=token_ids.device, dtype=torch.bool
        )
        out = embeddings
        for transformer in self.transformers:
            out = transformer(
                out,
                position_ids=position_ids,
                key_padding_mask=padding_mask,
                # attn_mask=causal_mask,
                attn_mask=None,  # since we are using sdpa, no need to pass it explicitly
            )
        out = self.norm(out)
        out = self.lm_head(out)  # bs, seq_len, vocab_size
        return out

    @property
    def num_params(self):
        return sum(p.numel() for p in self.parameters())

    # many of these properties have been adapted from nanoChat
    # Ref: https://github.com/karpathy/nanochat/blob/92d63d4e8bb4df75c3b71618f31ddde2378b2bcd/nanochat/gpt.py

    @property
    def num_matmul_params(self):
        """
        Number of parameters that participate in matrix multiplication
        each of these parameters cost 2 FLOPs/param to the forward pass.
        """
        return sum(
            m.weight.numel() for m in self.modules() if isinstance(m, torch.nn.Linear)
        )

    def get_estimated_flops(self, seq_len: int):
        """
        Estimated FLOPs per token for the model while training (forward + backward)
        """
        num_flops_per_token = 6 * self.num_matmul_params
        num_flops_per_token += self.nlayers * self.embed_dim * seq_len  # attn flops

        return num_flops_per_token
