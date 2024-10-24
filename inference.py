from src.tokenizer import loadTokenizer
from src.dataset import ZeroDataset
from src.misc import countParams
from src.nn import TransNet
import random
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model, metadata = TransNet.from_pretrained('model.pth')
model = model.to(device)

src_tokenizer = loadTokenizer('en', metadata.src_vocab_path)
trg_tokenizer = loadTokenizer('fa', metadata.trg_vocab_path)

train_ds = ZeroDataset(
    root = 'Dataset/machine_translation_daily_dialog_en_fa/data/',
    split = 'val'
)

sample = train_ds[random.randint(0, len(train_ds))]

text = sample['input']
# text = "i need to go shopping."
enc = src_tokenizer([text])

generatd = model.translate(
    input_ids = enc['input_ids'].to(device)
)
gen_text = trg_tokenizer.decode(generatd[0], ignore_special = True)
print(f"Input : {text}")
print(f"Prediction : {gen_text}")