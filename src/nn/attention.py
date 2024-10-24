from typing import Optional
from torch import nn
import torch

class MultiheadAttention(nn.Module):
    def __init__(self, cfg) -> None:
        super(MultiheadAttention, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim = cfg.n_embd,
            num_heads = cfg.n_head,
            dropout = cfg.dropout,
            batch_first = True,
            bias = cfg.bias
        )

    def forward(
        self,
        query : torch.FloatTensor,
        kv_cross : torch.FloatTensor,
        padding_mask : Optional[torch.BoolTensor] = None,
        att_mask : Optional[torch.BoolTensor] = None,
        return_attn_weights : bool = False
    ) -> torch.FloatTensor:
        att, w = self.multihead_attn(
            query,
            kv_cross,
            kv_cross,
            attn_mask = att_mask,
            key_padding_mask = padding_mask
        )
        if return_attn_weights:
            return (att, w)
        return att