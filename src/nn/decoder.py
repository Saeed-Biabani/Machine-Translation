from typing import Optional
from .transformer import TransformerDecoder
from .embedding import Embedding
from torch import nn
import torch

class Decoder(nn.Module):
    def __init__(self, cfg) -> None:
        super(Decoder, self).__init__()
        self.pad_idx = cfg.pad_idx
        self.emb = Embedding(
            n_embd = cfg.n_embd,
            n_vocab = cfg.n_trg_vocab,
            pad_idx = cfg.pad_idx,
            max_positions = cfg.max_trg_positions,
            dropout = cfg.dropout,
            norm_eps = cfg.norm_eps
        )
        self.trdec = TransformerDecoder(cfg)
        self.out = nn.Linear(
            cfg.n_embd,
            cfg.n_trg_vocab,
            bias = False
        )

    def forward(
        self,
        encoder_hidden_states : torch.FloatTensor,
        input_ids : torch.LongTensor,
        input_padding_mask : Optional[torch.ByteTensor] = None,
        encoder_padding_mask : Optional[torch.ByteTensor] = None
    ) -> torch.FloatTensor:
        if input_padding_mask is not None:
            input_padding_mask = input_padding_mask == self.pad_idx
        if encoder_padding_mask is not None:
            encoder_padding_mask = encoder_padding_mask == self.pad_idx

        input_embedding = self.emb(input_ids)
        hidden_states = self.trdec(
            encoder_hidden_states = encoder_hidden_states,
            input_embedding = input_embedding,
            input_padding_mask = input_padding_mask,
            encoder_padding_mask = encoder_padding_mask
        )
        return self.out(hidden_states)