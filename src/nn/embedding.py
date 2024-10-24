from torch import nn
import torch
import math

class TokenEmbedding(nn.Embedding):
    def __init__(
        self,
        vocab_size : int,
        embed_size : int,
        pad_idx : int
    ) -> None:
        super(TokenEmbedding, self).__init__(
            vocab_size,
            embed_size,
            padding_idx = pad_idx
        )

class SinPosEmbedding(nn.Module):
    def __init__(
        self,
        n_embd : int,
        max_positions : int
    ) -> None:
        super(SinPosEmbedding, self).__init__()
        self.d_model = n_embd
        self.max_len = max_positions

        pe = torch.zeros(self.max_len, self.d_model).float()
        pe.require_grad = False

        position = torch.arange(0, self.max_len).float().unsqueeze(1)
        div_term = (
            torch.arange(
                0, self.d_model, 2
            ).float() * -(
                math.log(10000.0) / self.d_model
            )
        ).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def extra_repr(self) -> str:
        return f"d_model={self.d_model}, max_len={self.max_len}"

    def forward(self, x : torch.FloatTensor) -> torch.FloatTensor:
        return self.pe[:, :x.size(1)]

class Embedding(nn.Module):
    def __init__(
        self,
        n_embd : int,
        n_vocab : int,
        pad_idx : int,
        max_positions : int,
        dropout : float,
        norm_eps : float
    ) -> None:
        super(Embedding, self).__init__()
        self.n_embd = n_embd
        self.token = TokenEmbedding(
            vocab_size = n_vocab,
            embed_size = n_embd,
            pad_idx = pad_idx
        )
        self.position = SinPosEmbedding(
            n_embd = n_embd,
            max_positions = max_positions
        )

        self.norm = nn.LayerNorm(n_embd, norm_eps)
        self.dropout = nn.Dropout(p = dropout)

    def forward(self, input_ids : torch.LongTensor) -> torch.FloatTensor:
        hidden_states = self.token(input_ids) + self.position(input_ids)
        return self.dropout(self.norm(hidden_states))