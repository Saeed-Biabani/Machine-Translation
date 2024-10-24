<p align="center">
  <h1 align="center">Machine Translation</h1>
  <p align="center">Transformer Based Model For Machine Translation, English To Persian</p>
</p>


#### **Quick Links**
- [Dependencies](#Dependencies)
- [Getting Started](#Getting-Starte)
- [Architecture](#Architecture)
- [Modules](#Modules)
- [Dataset](#Dataset)
- [Training](#Training)
- [Prediction](#Prediction)
- [License](#License)

## Dependencies
- Install Dependencies `$ pip install -r requirements.txt`
- Download Pretrained Weights [Here](https://huggingface.co/ordaktaktak/Machine-Translation)

## Getting Started
- Project Structure
```
```

## Architecture
<p align="center">
  <div align="center"><img src="assets/machine-translation.png" height = 500 ></div>
  <p align="center"> Fig. 1. Proposed Model Architecture </p>
</p>

## Modules
<p align="justify">

  <div align="justify">

  <p align="justify"><strong><a href = 'https://arxiv.org/pdf/1706.03762'>Positional Encoding:</a></strong> Since our model contains no recurrence and no 
    convolution, in order for the model to make use of the order of the sequence, we must inject some information about the relative or absolute position of the 
    tokens in the sequence. In this work, we use sine and cosine functions of different frequencies:</p>
  <p align="center"> $\ PE_{(pos, 2i)} = \sin({\frac{pos}{10000^\frac{2i}{d_{model}}}})$ </p>
  <p align="center"> $\ PE_{(pos, 2i+1)} = \cos({\frac{pos}{10000^\frac{2i}{d_{model}}}})$ </p>
  
  </div>

  <div align="justify">

  <p align="justify"><strong><a href = 'https://arxiv.org/pdf/1706.03762v7'>Multi-Head Attention:</a></strong> Multi-head Attention is a module for attention 
    mechanisms which runs through an attention mechanism several times in parallel. The independent attention outputs are then concatenated and linearly 
    transformed into the expected dimension. Intuitively, multipleattention heads allows for attending to parts of the sequence differently (e.g. longer-term 
    dependencies versus shorter-term dependencies).</p>
  <p align="center"> $\ MultiHead(Q, K, V) = Concat(head_1, ..., head_h)W^0$ </p>
  <p align="center"> where $\ head_i = Attention(Q{W_i}^Q, K{W_i}^K, V{W_i}^V)$</p>
  <p align="center"> $\ Attention(Q, K, V) = softmax(\frac{QK^T}{\sqrt{d_k}})V$</p>
  <p align="justify"> above $\ W$ are all learnable parameter matrices.</p>
  
  </div>

</p>

## Dataset
<div align="center">
  We Use <strong>machine_translation_daily_dialog_en_fa</strong> DataSet For Train Our Model That You Can Find It <a href = 'https://huggingface.co/datasets/ordaktaktak/machine_translation_daily_dialog_en_fa'>
    Here
  </a>
</div>

## Training
<p align="center">
  <div align="center"><img src="assets/lossCurve.png"></div>
</p>

## 🛡️ License <a name="license"></a>
Project is distributed under [MIT License](https://github.com/Saeed-Biabani/Machine-Translation/blob/main/LICENSE)
