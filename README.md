# MiniGPT, Character-Level Language Model from Scratch

A from-scratch implementation of a GPT-style decoder-only Transformer in PyTorch, trained on raw text data at the character level. Designed to demonstrate how large language models work internally, from tokenization and embeddings to multi-head self-attention and autoregressive generation.

---

## Overview

This project implements the core architecture of a GPT language model, built entirely with PyTorch primitives. It trains on any plain-text corpus and learns to generate coherent text by predicting one character at a time. The architecture mirrors the original "Attention Is All You Need" decoder block design, including:

- Character-level tokenization
- Token and positional embeddings
- Multi-head causal self-attention
- Position-wise feed-forward layers
- Layer normalization with residual connections
- Autoregressive text generation

---

## Architecture

| Component            | Details                                      |
|----------------------|----------------------------------------------|
| Model type           | Decoder-only Transformer (GPT-style)         |
| Embedding dimension  | 384                                          |
| Attention heads      | 6                                            |
| Transformer blocks   | 6                                            |
| Context length       | 256 tokens                                   |
| Dropout              | 0.2                                          |
| Optimizer            | AdamW                                        |
| Learning rate        | 3e-4                                         |

---

## Requirements

- Python 3.8+
- PyTorch 2.0+
- CUDA (optional, but recommended for training speed)

Install dependencies:

```bash
pip install torch
```

---

## Usage

### 1. Prepare your dataset

Place your training corpus as a plain text file named `input.txt` in the project root (or pass `--input path`). The results below use [Tiny Shakespeare](https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt). The model will automatically build its vocabulary from the characters present in the file.

```
project/
├── gpt_lm.py
└── input.txt
```

### 2. Train the model

```bash
python gpt_lm.py
```

Training runs for 5,000 iterations by default, printing train/validation loss every 500 steps. On completion, the model generates a 500-character sample to stdout.

### 3. Train a smaller model on a CPU

The defaults are sized for a GPU. Every size is a flag, so a small model trains on a laptop:

```bash
python gpt_lm.py --n-embd 128 --n-head 4 --n-layer 4 --block-size 128 \
  --max-iters 2000 --eval-interval 250 --eval-iters 50
```

---

## Results

One CPU run of the small config above on Tiny Shakespeare (1.1 MB of text, 0.82M parameters, seed 1337):

```
step 0:    train loss 4.1340, val loss 4.1329
step 500:  train loss 2.2995, val loss 2.3118
step 1000: train loss 1.8445, val loss 1.9486
step 1500: train loss 1.6827, val loss 1.8460
step 1999: train loss 1.6158, val loss 1.8005
```

A sample after those 2,000 steps:

```
Which thert is to to the denamperects thie,
It with honour untraiship like your and to him.

Cit deather:
One, brous to too not he witneesent the in's
```

It has learned line lengths, speaker headings and common short words, and still invents most longer ones. I have not run the full default config (384 dimensions, 6 layers, 5,000 steps), which needs a GPU.

---

## Hyperparameters

All hyperparameters are defined at the top of `gpt_lm.py` and can be adjusted directly:

| Parameter       | Default | Description                          |
|-----------------|---------|--------------------------------------|
| `batch_size`    | 32      | Number of sequences per training step |
| `block_size`    | 256     | Maximum context length (tokens)      |
| `max_iters`     | 5000    | Total training iterations            |
| `eval_interval` | 500     | Steps between loss evaluations       |
| `learning_rate` | 3e-4    | AdamW learning rate                  |
| `n_embd`        | 384     | Embedding dimension                  |
| `n_head`        | 6       | Number of attention heads            |
| `n_layer`       | 6       | Number of Transformer blocks         |
| `dropout`       | 0.2     | Dropout probability                  |

---

## Project Structure

```
project/
├── gpt_lm.py       # Full model definition and training loop
└── input.txt       # Training corpus (user-provided)
```

---

## How It Works

1. **Tokenization**, Characters are mapped to integer indices using a simple character-level vocabulary built from the input file.
2. **Embedding**, Each token is embedded into a 384-dimensional vector. Positional embeddings are added to encode sequence order.
3. **Transformer Blocks**, 6 stacked decoder blocks, each containing multi-head causal self-attention and a feed-forward network, with layer normalization and residual connections.
4. **Training**, The model is trained via cross-entropy loss on next-character prediction using AdamW, with periodic evaluation on a held-out validation split (10% of data).
5. **Generation**, Text is generated autoregressively using multinomial sampling from the softmax output distribution.

---

## License

This project is released under the MIT License. See `LICENSE` for details.

---

## Acknowledgements

Architecture and training design inspired by Andrej Karpathy's [nanoGPT](https://github.com/karpathy/nanoGPT) and the original paper [*Attention Is All You Need*](https://arxiv.org/abs/1706.03762) by Vaswani et al.
