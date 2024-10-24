from .attention import MultiheadAttention
from torch.nn import functional as nnf
from typing import Optional
from torch import nn
import torch

class ffnReLU(nn.Module):
    def __init__(self, cfg) -> None:
        super(ffnReLU, self).__init__()
        self.up = nn.Linear(
            cfg.n_embd,
            cfg.n_embd * cfg.ffn_ratio
        )
        self.down = nn.Linear(
            cfg.n_embd * cfg.ffn_ratio,
            cfg.n_embd
        )
        

    def forward(
        self,
        hidden_states : torch.FloatTensor
    ) -> torch.FloatTensor:
        hidden_states = self.up(hidden_states)
        hidden_states = nnf.silu(hidden_states)
        return self.down(hidden_states)

class EncoderLayer(nn.Module):
    def __init__(self, config) -> None:
        super(EncoderLayer, self).__init__()
        self.att_norm = nn.LayerNorm(config.n_embd, eps = config.norm_eps)
        self.attn = MultiheadAttention(config)

        self.ffn_norm = nn.LayerNorm(config.n_embd, eps = config.norm_eps)
        self.ffn = ffnReLU(config)

        self.drop = nn.Dropout(config.dropout)

    def forward(
        self,
        hidden_states : torch.FloatTensor,
        padding_mask : Optional[torch.BoolTensor] = None
    ) -> torch.FloatTensor:
        hidden_states_norm = self.att_norm(hidden_states)
        attention = self.attn(
            query = hidden_states_norm,
            kv_cross = hidden_states_norm,
            padding_mask = padding_mask
        )
        hidden_states = hidden_states + self.drop(attention)
        
        hidden_states_norm = self.ffn_norm(hidden_states)
        ffn_out = self.ffn(hidden_states_norm)
        hidden_states = hidden_states + self.drop(ffn_out)
        return hidden_states

class TransformerEncoder(nn.Module):
    def __init__(self, config) -> None:
        super(TransformerEncoder, self).__init__()

        self.transformer_enc = nn.ModuleList(
            [EncoderLayer(config) for _ in range(config.n_encoder_layer)]
        )
        self.norm = nn.LayerNorm(config.n_embd, eps = config.norm_eps)

    def forward(
        self,
        input_embedding : torch.FloatTensor,
        padding_mask : Optional[torch.BoolTensor] = None
    ) -> torch.FloatTensor:
        hidden_states = input_embedding
        for block in self.transformer_enc:
            hidden_states = block(
                hidden_states,
                padding_mask = padding_mask
            )
        return self.norm(hidden_states)

class DecoderLayer(nn.Module):
    def __init__(self, config) -> None:
        super(DecoderLayer, self).__init__()
        self.att_norm_1 = nn.LayerNorm(config.n_embd, eps = config.norm_eps)
        self.attn_1 = MultiheadAttention(config)

        self.att_norm_2 = nn.LayerNorm(config.n_embd, eps = config.norm_eps)
        self.attn_2 = MultiheadAttention(config)

        self.ffn_norm = nn.LayerNorm(config.n_embd, eps = config.norm_eps)
        self.ffn = ffnReLU(config)

        self.drop = nn.Dropout(config.dropout)

    def forward(
        self,
        encoder_hidden_states : torch.FloatTensor,
        hidden_states : torch.FloatTensor,
        padding_mask : Optional[torch.BoolTensor] = None,
        encoder_padding_mask : Optional[torch.BoolTensor] = None
    ) -> torch.FloatTensor:
        _, l, _ = hidden_states.shape
        att_mask = torch.triu(
            torch.ones(l, l,device = hidden_states.device), 1
        ).bool()
        
        hidden_states_norm = self.att_norm_1(hidden_states)
        attention = self.attn_1(
            query = hidden_states_norm,
            kv_cross = hidden_states_norm,
            padding_mask = padding_mask,
            att_mask = att_mask
        )
        hidden_states = hidden_states + self.drop(attention)


        hidden_states_norm = self.att_norm_2(hidden_states)
        attention = self.attn_2(
            query = hidden_states_norm,
            kv_cross = encoder_hidden_states,
            padding_mask = encoder_padding_mask
        )
        hidden_states = hidden_states + self.drop(attention)
        
        hidden_states_norm = self.ffn_norm(hidden_states)
        ffn_out = self.ffn(hidden_states_norm)
        hidden_states = hidden_states + self.drop(ffn_out)
        return hidden_states

class TransformerDecoder(nn.Module):
    def __init__(self, config) -> None:
        super(TransformerDecoder, self).__init__()

        self.transformer_enc = nn.ModuleList(
            [DecoderLayer(config) for _ in range(config.n_decoder_layer)]
        )
        self.norm = nn.LayerNorm(config.n_embd, eps = config.norm_eps)

    def forward(
        self,
        encoder_hidden_states : torch.FloatTensor,
        input_embedding : torch.FloatTensor,
        input_padding_mask : Optional[torch.BoolTensor] = None,
        encoder_padding_mask : Optional[torch.BoolTensor] = None
    ) -> torch.FloatTensor:
        hidden_states = input_embedding
        for block in self.transformer_enc:
            hidden_states = block(
                encoder_hidden_states = encoder_hidden_states,
                hidden_states = hidden_states,
                padding_mask = input_padding_mask,
                encoder_padding_mask = encoder_padding_mask
            )
        return self.norm(hidden_states)