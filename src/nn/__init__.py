from torch.distributions import Categorical
from torch.nn import functional as nnf
from dataclasses import dataclass
from ..dataset import MetaData
from .encoder import Encoder
from .decoder import Decoder
from typing import Optional
from torch import nn
import torch

@dataclass
class TransOutput:
    encoder_hidden_states : Optional[torch.FloatTensor] = None
    loss : Optional[torch.FloatTensor] = None
    logits : torch.FloatTensor = None

@dataclass
class NetConfig:
    n_src_vocab : int = None
    n_trg_vocab : int = None

    pad_idx : int = None
    sos_idx : int = None
    eos_idx : int = None

    max_src_positions : int = None
    max_trg_positions : int = None

    norm_eps : float = 1e-6
    dropout : float = 0.1
    ffn_ratio : int = 4
    n_head : int = 8
    n_embd : int = 256
    bias : bool = True
    n_encoder_layer : int = 4
    n_decoder_layer : int = 4

class TransNet(nn.Module):
    def __init__(self, cfg) -> None:
        super(TransNet, self).__init__()
        self.cfg = cfg
        self.encoder = Encoder(cfg)
        self.decoder = Decoder(cfg)
    
    def translate(
        self,
        input_ids : torch.LongTensor,
        temp : float = 0.1,
        device : str = 'cuda',
        max_generation : int = 64
    ) -> torch.LongTensor:
        sos_token = self.cfg.sos_idx * torch.ones(1, 1).long()
        log_tokens = [sos_token]

        with torch.no_grad():
            with torch.amp.autocast('cuda'):
                encoder_hidden_states = self.apply_encoder(input_ids.to(device))

            for _ in range(max_generation):
                input_tokens = torch.cat(log_tokens, 1)

                data_pred = self.apply_decoder(
                    encoder_hidden_states = encoder_hidden_states,
                    input_ids = input_tokens.to(device),
                )

                dist = Categorical(logits = data_pred[:, -1] / temp)
                next_tokens = dist.sample().reshape(1, 1)

                log_tokens.append(next_tokens.cpu())

                if next_tokens.item() == self.cfg.eos_idx:
                    break

        return torch.cat(log_tokens, 1)
    
    def apply_encoder(
        self,
        input_ids : torch.LongTensor,
        padding_mask : Optional[torch.ByteTensor] = None
    ) -> torch.FloatTensor:
        return self.encoder(
            input_ids = input_ids,
            padding_mask = padding_mask
        )

    def apply_decoder(
        self,
        encoder_hidden_states : torch.FloatTensor,
        input_ids : torch.LongTensor,
        input_padding_mask : Optional[torch.ByteTensor] = None,
        encoder_padding_mask : Optional[torch.ByteTensor] = None
    ) -> torch.FloatTensor:
        return self.decoder(
            encoder_hidden_states = encoder_hidden_states,
            input_ids = input_ids,
            input_padding_mask = input_padding_mask,
            encoder_padding_mask = encoder_padding_mask
        )

    def forward(
        self,
        src_ids : torch.LongTensor,
        trg_ids : torch.LongTensor,
        src_padding_mask : Optional[torch.ByteTensor] = None,
        trg_padding_mask : Optional[torch.ByteTensor] = None,
        labels : torch.LongTensor = None
    ) -> TransOutput:
        encoder_hidden_states = self.encoder(
            input_ids = src_ids,
            padding_mask = src_padding_mask
        )
        decoder_output = self.decoder(
            encoder_hidden_states = encoder_hidden_states,
            input_ids = trg_ids,
            input_padding_mask = trg_padding_mask,
            encoder_padding_mask = src_padding_mask,
        )

        loss = None
        if labels != None:
            loss = (nnf.cross_entropy(
                decoder_output.transpose(1, 2),
                labels, reduction = 'none'
            ) * trg_padding_mask).mean()

        return TransOutput(
            loss = loss,
            encoder_hidden_states = encoder_hidden_states,
            logits = decoder_output
        )
    @staticmethod
    def from_pretrained(
        path : str,
        device : str = 'cpu',
        return_metadata : bool = True
    ) -> nn.Module:
        data = torch.load(
            path,
            weights_only = False,
            map_location = device
        )
        model = TransNet(
            NetConfig(
                **data['model_config']
            )
        )
        model.load_state_dict(
            data['model_weights']
        )
        model.eval()
        if return_metadata:
            return (model, MetaData(**data['metadata']))
        return model