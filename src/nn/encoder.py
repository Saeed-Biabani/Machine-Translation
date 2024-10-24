from typing import Optional
from .transformer import TransformerEncoder
from .embedding import Embedding
from torch import nn
import torch

class Encoder(nn.Module):
    def __init__(self, cfg) -> None:
        super(Encoder, self).__init__()
        self.pad_idx = cfg.pad_idx
        self.emb = Embedding(
            n_embd = cfg.n_embd,
            n_vocab = cfg.n_src_vocab,
            pad_idx = cfg.pad_idx,
            max_positions = cfg.max_src_positions,
            dropout = cfg.dropout,
            norm_eps = cfg.norm_eps
        )
        self.encoder = TransformerEncoder(cfg)

    def forward(
        self,
        input_ids : torch.LongTensor,
        padding_mask : Optional[torch.ByteTensor] = None
    ) -> torch.FloatTensor:
        if padding_mask is not None:
            padding_mask = padding_mask == self.pad_idx
        
        input_embedding = self.emb(input_ids)
        return self.encoder(
            input_embedding = input_embedding,
            padding_mask = padding_mask
        )