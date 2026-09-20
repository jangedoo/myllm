import einops
import torch


def apply_rope(x, cos, sin):
    """
    x: [B, num_heads, seq_len, head_dim]
    cos/sin: [B, 1, seq_len, head_dim // 2]
    """
    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]

    rotated_even = x_even * cos - x_odd * sin
    rotated_odd = x_even * sin + x_odd * cos

    # [B, num_heads, seq_len, head_dim/2, 2]
    x_rotated = torch.stack([rotated_even, rotated_odd], dim=-1)

    # [B, num_heads, seq_len, head_dim]
    return x_rotated.flatten(-2)


def get_position_ids(bs, seq_len, device, key_padding_mask=None):
    if key_padding_mask is not None:
        # masked tokens should not be counted for positions
        # e.g. basic position_ids=[0, 1, 2, 3] mask=[True, True, False, False]
        # position_ids should be = [0, 0, 0, 1]
        valid = ~key_padding_mask
        position_ids = valid.long().cumsum(dim=-1) - 1
        position_ids = position_ids.clamp_min(0)
    else:
        position_ids = torch.arange(seq_len, device=device)
        # right now position ids is a vector but we need to repeat it 'bs' times
        # position_ids = position_ids.unsqueeze(0).expand(bs, -1)
        position_ids = einops.repeat(position_ids, "seq_len -> bs seq_len", bs=bs)
    return position_ids


class RotaryEmbedding(torch.nn.Module):
    def __init__(self, head_dim: int, theta: float = 10_000.0):
        super().__init__()

        assert head_dim % 2 == 0
        inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))

        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, position_ids, dtype):
        """
        position_ids: [B, seq_len]
        Returns:
            cos: [B, 1, seq_len, head_dim/2]
            sin: [B, 1, seq_len, head_dim/2]
        """
        angles = position_ids.float().unsqueeze(-1) * self.inv_freq
        cos = angles.cos().unsqueeze(1)
        sin = angles.sin().unsqueeze(1)
        return cos.to(dtype), sin.to(dtype)


class RoPEAttention(torch.nn.Module):
    def __init__(self, embed_dim: int, nheads: int, theta: float = 10_000.0):
        super().__init__()
        assert embed_dim % nheads == 0

        self.embed_dim = embed_dim
        self.nheads = nheads
        self.head_dim = embed_dim // nheads

        assert self.head_dim % 2 == 0

        # self.q_proj = torch.nn.Linear(embed_dim, embed_dim)
        # self.k_proj = torch.nn.Linear(embed_dim, embed_dim)
        # self.v_proj = torch.nn.Linear(embed_dim, embed_dim)

        # fused qkv proj
        self.qkv_proj = torch.nn.Linear(embed_dim, 3 * embed_dim)

        self.out_proj = torch.nn.Linear(embed_dim, embed_dim)
        self.rope = RotaryEmbedding(head_dim=self.head_dim, theta=theta)

    def split_heads(self, x: torch.Tensor):
        # bs = x.shape[0]
        # x.view(bs, -1, self.nheads, self.head_dim).transpose(1, 2)
        return einops.rearrange(
            x,
            "bs seq_len (nheads head_dim) -> bs nheads seq_len head_dim",
            nheads=self.nheads,
        )

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
        attn_mask: torch.Tensor | None = None,
        position_ids=None,
    ):
        bs, seq_len, embed_dim = x.shape

        if position_ids is None:
            position_ids = get_position_ids(
                bs=bs,
                seq_len=seq_len,
                device=x.device,
                key_padding_mask=key_padding_mask,
            )

        cos, sin = self.rope(position_ids, dtype=x.dtype)

        qkv = self.qkv_proj(x)  # bs, seq_len, 3 * embed_dim
        q, k, v = einops.rearrange(
            qkv,
            "bs seq_len (qkv nheads head_dim) -> qkv bs nheads seq_len head_dim",
            qkv=3,
            nheads=self.nheads,
        )

        # q = self.split_heads(self.q_proj(x))
        # k = self.split_heads(self.k_proj(x))
        # v = self.split_heads(self.v_proj(x))

        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        if key_padding_mask is None and attn_mask is None:
            # fast path causal attention without explicit masks
            out = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, is_causal=True, dropout_p=0.0
            )
        else:
            # SDPA mask semantics: True = attend, False = masked out
            # unlike the key_padding_mask and attn_mask inputs which assume True=mask, False=attend
            if attn_mask is None:
                sdpa_attn_mask = torch.ones(
                    seq_len, seq_len, dtype=torch.bool, device=x.device
                ).tril()
            else:
                # flip the mask so that it works with SDPA
                sdpa_attn_mask = attn_mask.logical_not()

            if key_padding_mask is not None:
                # key_padding_mask = [bs, seq_len]
                # True = padding, so invert for SDPA
                valid_keys = key_padding_mask.logical_not()
                sdpa_attn_mask = (
                    sdpa_attn_mask[None, None, :, :] & valid_keys[:, None, None, :]
                )
            out = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, attn_mask=sdpa_attn_mask, is_causal=False, dropout_p=0.0
            )

        out = out.transpose(1, 2)  # bs, seq_len, nheads, head_dim
        out = out.contiguous().view(bs, seq_len, embed_dim)
        return self.out_proj(out)

        # scale = self.head_dim**-0.5
        # # we can now try torch's flex attention etc. here but for now, let's implement ourselves
        # scores = q @ k.transpose(-2, -1) * scale  # [bs, nheads, seq_len, seq_len]

        # if attn_mask is None:
        #     attn_mask = torch.nn.Transformer.generate_square_subsequent_mask(
        #         seq_len, device=x.device, dtype=torch.bool
        #     )

        # scores = scores.masked_fill(attn_mask[None, None, :, :], float("-inf"))

        # if key_padding_mask is not None:
        #     scores = scores.masked_fill(
        #         key_padding_mask[:, None, None, :], float("-inf")
        #     )

        # attn_weights = torch.softmax(scores, dim=-1)
        # out = attn_weights @ v  # bs, nheads, seq_len, head_dim
        # out = out.transpose(1, 2)  # bs, seq_len, nheads, head_dim
        # out = out.contiguous().view(bs, seq_len, embed_dim)  # bs, seq_len, embed_dim
        # out = self.out_proj(out)
        # return out
