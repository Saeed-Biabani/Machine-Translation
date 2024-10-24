from torch import nn
import torch

class TokenDrop(nn.Module):
    def __init__(
        self,
        prob : float = 0.1,
        unk_token : int = 1,
        n_specials : int = 4
    ) -> None:
        self.prob = prob
        self.n_specials = n_specials
        self.unk_token = unk_token

    def __call__(
        self,
        sample : torch.LongTensor
    ) -> torch.LongTensor:
        mask = torch.bernoulli(self.prob * torch.ones_like(sample)).long()

        can_drop = (sample >= self.n_specials).long()
        mask = mask * can_drop

        replace_with = (self.unk_token * torch.ones_like(sample)).long()

        sample_out = (1 - mask) * sample + mask * replace_with

        return sample_out