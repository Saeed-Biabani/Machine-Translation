from hazm import Normalizer, word_tokenize
from typing import Mapping
from .vocab import Vocab
import torch
import re

class Tokenizer:
    def __init__(self):
        _patterns = [r"\'", r"\"", r"\.", r"<br \/>", r",", r"\(", r"\)", r"\!", r"\?", r"\;", r"\:", r"\s+"]
        _replacements = [" '  ", "", " . ", " ", " , ", " ( ", " ) ", " ! ", " ? ", " ", " ", " "]
        self._patterns_dict = list((re.compile(p), r) for p, r in zip(_patterns, _replacements))

    def __call__(self, line):
        line = line.lower()
        for pattern_re, replaced_str in self._patterns_dict:
            line = pattern_re.sub(replaced_str, line)
        return line.strip().split(' ')

class AutoTokenizer:
    def __init__(
        self,
        tokenizer,
        vocab : Vocab
    ):
        self.tokenizer = tokenizer
        self.vocab = vocab
    
    def __call__(
        self,
        input : tuple[str]
    ) -> Mapping[str, torch.Tensor]:
        tokens = [self.tokenizer(item) for item in input]
        
        max_length = len(max(tokens, key = len)) + 1
        bs = len(input)

        batch = torch.zeros((bs, max_length))

        for indx, item in enumerate(tokens):
            encoded = self.vocab.encode(item + ['[EOS]'])
            batch[indx, :len(encoded)] = torch.LongTensor(encoded)
        
        sos_axis = torch.zeros((bs, 1)).fill_(self.vocab['[SOS]']).long()
        batch = torch.cat((sos_axis, batch), dim = 1)

        padding_mask = (~(batch == self.vocab['[PAD]'])).int()
        
        return {
            "input_ids" : batch.long(),
            "padding_mask" : padding_mask,
        }
    
    def decode(
        self,
        input : tuple[int],
        ignore_special : bool = True
    ) -> str:
        input = input.cpu().numpy()
        unt = self.vocab.decode(input, ignore_special)
        return ' '.join(unt)

def src_tokenizer(path) -> AutoTokenizer:
    vocab = Vocab()
    vocab.loadVocab(path)
    
    tokenizer = Tokenizer()
    
    return AutoTokenizer(tokenizer, vocab)

def trg_tokenizer(path) -> AutoTokenizer:
    vocab = Vocab()
    vocab.loadVocab(path)
    
    normalizer = Normalizer()
    tokenizer = lambda x : word_tokenize(normalizer.normalize(x))
    
    return AutoTokenizer(tokenizer, vocab)

def loadTokenizer(type_, path) -> AutoTokenizer:
    if type_ == 'en':
        return src_tokenizer(path)
    elif type_ == 'fa':
        return trg_tokenizer(path)
    else:
        raise NotImplementedError("type_ must be one of ['en', 'fa']")