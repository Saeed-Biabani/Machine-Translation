from torch.utils.data import Dataset
from typing import Mapping, Optional
from dataclasses import dataclass
import pandas as pd
import pathlib
import random
import pickle
import os

class ZeroDataset(Dataset):
    def __init__(
        self,
        root : str,
        split : str = 'train',
        dataset : Optional[pd.DataFrame] = None
    ) -> None:
        super(ZeroDataset, self).__init__()
        self.root = root
        if dataset is not None:
            self.df = dataset
        elif dataset is None:
            self.df = ZeroDataset.load_data(root, split = split)

    @staticmethod
    def load_data(
        root : str,
        split : str
    ) -> pd.DataFrame:
        files = list(pathlib.Path(root).glob(f'*{split}*.parquet'))
        dfs = []
        for fname in files:
            dfs.append(pd.read_parquet(fname))
        return pd.concat(dfs).sample(frac = 1.).reset_index()

    def train_test_split(
        self,
        test_size : float = 0.1
    ) -> Mapping[str, Dataset]:
        val_data = pd.DataFrame()
        train_data = pd.DataFrame()

        n_val_sampels = int(len(self.df) * test_size)
        total_slices = set(range(len(self.df)))

        val_slices = set(random.sample(list(total_slices), n_val_sampels))
        other_ = total_slices - val_slices

        val_data = pd.concat([val_data, self.df.iloc[list(val_slices)]]).sample(frac = 1.)
        train_data = pd.concat([train_data, self.df.iloc[list(other_)]]).sample(frac = 1.)

        return {
            'train' : ZeroDataset(root = self.root, dataset = train_data.reset_index()),
            'test' : ZeroDataset(root = self.root, dataset = val_data.reset_index())
        }

    def __len__(self) -> int:
        return len(self.df)
    
    def __getitem__(self, index : int) -> Mapping[str, str]:
        row = self.df.iloc[index]
        return {
            'input' : row['en'],
            'output' : row['fa']
        }

@dataclass
class MetaData:
    src_vocab_path : str = None
    trg_vocab_path : str = None

    max_src_positions : int = None
    max_trg_positions : int = None

    sos_idx : int = None
    pad_idx : int = None
    eos_idx : int = None

def loadInfo(root : str) -> MetaData:
    fname = os.path.join(root, 'info.pkl')
    with open(fname, 'rb') as f:
        data = pickle.load(f)
        return MetaData(**data)