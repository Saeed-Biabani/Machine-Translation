from src.schedule import get_linear_schedule_with_warmup
from src.misc import countParams, updateLog, Visualizer
from src.trainutils import TrainOneEpoch, TestOneEpoch
from src.dataset import ZeroDataset, loadInfo
from src.tokenizer import loadTokenizer
from torch.utils.data import DataLoader
from src.nn import TransNet, NetConfig
from src.tracker import Tracker
from config import TrainConfig
import pandas as pd
import torch
import pprint

train_cfg = TrainConfig()
pprint.pprint(train_cfg)

dataset_info = loadInfo(train_cfg.root)
pprint.pprint(dataset_info)

src_tokenizer = loadTokenizer('en', dataset_info.src_vocab_path)
trg_tokenizer = loadTokenizer('fa', dataset_info.trg_vocab_path)

train_ds = ZeroDataset(
    root = train_cfg.root,
    split = 'train'
); train_ldr = DataLoader(
    train_ds,
    batch_size = train_cfg.batch_size,
    shuffle = True
)

test_ds = ZeroDataset(
    root = train_cfg.root,
    split = 'test'
); test_ldr = DataLoader(
    test_ds,
    batch_size = train_cfg.batch_size,
    shuffle = True
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

net_cfg = NetConfig(
    n_src_vocab = len(src_tokenizer.vocab),
    n_trg_vocab = len(trg_tokenizer.vocab),
    max_src_positions = dataset_info.max_src_positions,
    max_trg_positions = dataset_info.max_trg_positions,
    pad_idx = dataset_info.pad_idx,
    sos_idx = dataset_info.sos_idx,
    eos_idx = dataset_info.eos_idx
)
model = TransNet(net_cfg).to(device)
print(model)
print(f"Trainable Parameters : {countParams(model):,}")

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr = train_cfg.lr,
    weight_decay = train_cfg.weight_decay
)
lr_scheduler = get_linear_schedule_with_warmup(
    optimizer,
    train_cfg.n_epochs * 0.05,
    train_cfg.n_epochs
)

log = {
    "epoch" : [],
    "train_loss" : [],
    "test_loss" : [],
    "lr" : [],
}

visualizer = Visualizer(
    patterns = ['loss', 'lr']
)

tracker = Tracker(
    model,
    monitor = train_cfg.monitor_value,
    delta = train_cfg.monitor_delta,
    mode = train_cfg.monitor_mode
)

template = {
    'metadata' : dataset_info.__dict__
}

for epoch in range(1, train_cfg.n_epochs + 1):
    train_log = TrainOneEpoch(
        model = model,
        optimizer = optimizer,
        src_tokenizer = src_tokenizer,
        trg_tokenizer = trg_tokenizer,
        ldr = train_ldr,
        epoch = epoch,
        device = device
    )
    updateLog(train_log, log)

    test_log = TestOneEpoch(
        model = model,
        src_tokenizer = src_tokenizer,
        trg_tokenizer = trg_tokenizer,
        ldr = test_ldr,
        epoch = epoch,
        device = device
    )
    updateLog(test_log, log)

    tracker.step(test_log, epoch)

    log['lr'].append(lr_scheduler.get_last_lr()[0])
    log['epoch'].append(epoch)

    visualizer(log)

    lr_scheduler.step()
    
    tracker.at_epoch_end()
    
    tracker.restore_best_weights(
        template = template,
        fname = train_cfg.save_as
    )
pd.DataFrame(log).to_csv('trainLog.csv', index = False)