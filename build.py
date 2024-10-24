from hazm import word_tokenize, Normalizer
from src.dataset import ZeroDataset
from src.tokenizer import Tokenizer
from src.vocab import Vocab
import pprint
import pickle
import tqdm
import os

root = 'Dataset/machine_translation_daily_dialog_en_fa/data/'

df = ZeroDataset.load_data(root = root, split = 'train')

# building src vocab
tokenizer = Tokenizer()
tokens = [
    tokenizer(item) for item in tqdm.tqdm(
        df['en'].to_list(),
        colour = 'magenta'
    )
]
src_max_len = len(max(tokens, key = len))

specials = ['[PAD]', '[UNK]', '[SOS]', '[EOS]']

vocab = Vocab()
vocab.initVocab(
    tokens,
    min_freq = 2,
    specials = specials 
);
src_vocab_path = os.path.join(root, "src_vocab.pkl")
vocab.saveVocab(src_vocab_path)
print(f"Num src Tokens : {len(vocab)}")

# building trg vocab
normalizer = Normalizer()
tokenizer = lambda x : word_tokenize(normalizer.normalize(x))

tokens = [
    tokenizer(item) for item in tqdm.tqdm(
        df['fa'].to_list(),
        colour = 'magenta'
    )
]
trg_max_len = len(max(tokens, key = len))

specials = ['[PAD]', '[UNK]', '[SOS]', '[EOS]']

vocab = Vocab()
vocab.initVocab(
    tokens,
    min_freq = 2,
    specials = specials 
);
trg_vocab_path = os.path.join(root, "trg_vocab.pkl")
vocab.saveVocab(trg_vocab_path)
print(f"Num trg Tokens : {len(vocab)}")

info_path = os.path.join(root, 'info.pkl')
with open(info_path, 'wb') as f:
    data = {
        'pad_idx' : specials.index('[PAD]'),
        'sos_idx' : specials.index('[SOS]'),
        'eos_idx' : specials.index('[EOS]'),
        'src_vocab_path' : src_vocab_path,
        'trg_vocab_path' : trg_vocab_path,
        'max_src_positions' : src_max_len + 8,
        'max_trg_positions' : trg_max_len + 8,
    }
    pickle.dump(data, f)

with open(os.path.join(root, 'info.pkl'), 'rb') as f:
    data = pickle.load(f)
    pprint.pprint(data, sort_dicts = False)