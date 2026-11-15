import argparse

import torch
import torch.nn as nn
from torch.nn import functional as F

# Hyperparameters. The defaults are the GPU config, pass smaller values to train on a CPU.
parser = argparse.ArgumentParser()
parser.add_argument("--input", default="input.txt")
parser.add_argument("--batch-size", type=int, default=32)
parser.add_argument("--block-size", type=int, default=256)   # Also known as context length
parser.add_argument("--max-iters", type=int, default=5000)
parser.add_argument("--eval-interval", type=int, default=500)
parser.add_argument("--eval-iters", type=int, default=200)
parser.add_argument("--lr", type=float, default=3e-4)
parser.add_argument("--n-embd", type=int, default=384)       # Dimension of the vector embeddings
parser.add_argument("--n-head", type=int, default=6)         # Number of Self-Attention heads
parser.add_argument("--n-layer", type=int, default=6)        # Number of decoder blocks
parser.add_argument("--dropout", type=float, default=0.2)
parser.add_argument("--sample-tokens", type=int, default=500)
args = parser.parse_args()

batch_size = args.batch_size
block_size = args.block_size
max_iters = args.max_iters
eval_interval = args.eval_interval
learning_rate = args.lr
device = "cuda" if torch.cuda.is_available() else "cpu"
eval_iters = args.eval_iters
n_embd = args.n_embd
n_head = args.n_head
n_layer = args.n_layer
dropout = args.dropout

torch.manual_seed(1337)

# Loading raw data
with open(args.input, "r", encoding="utf-8") as f:
    data = f.read()

# making the Vocab List
chars = sorted(list(set(data)))
vocab_size = len(chars)

# Creating a map between characters and integers (This is our simple character tokenizer)
stoi = {ch:i for i,ch in enumerate(chars)}
itos = {i:ch for i,ch in enumerate(chars)}
encode = lambda x: [stoi[c] for c in x]
decode = lambda l: ''.join([itos[i] for i in l])

# Train and test splits
data = torch.tensor(encode(data), dtype=torch.long)
n = int(0.9*len(data)) # First 90%data will we the train data and the rest 10% will be the test data!
train_data = data[:n]
val_data = data[n:]

# data loading
def get_batch(split):

    data = train_data if split=='train' else val_data
    ix = torch.randint(len(data)-block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y =x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train','val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


class Head(nn.Module):
    """Head for self-attention"""

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self , x):

        B, T, C = x.shape
        k = self.key(x)     # (B,T,hs)
        q = self.query(x)   # (B,T,hs)

        # Computing attention scores (affinities)
        wei = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5   # (B,T,hs) @ (B,hs,T) -> (b,T,T)
        wei = wei.masked_fill(self.tril[:T,:T] == 0, float('-inf')) # (B,T,T)
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)

        # Weighted sum
        v = self.value(x)
        out = wei @ v # (B,T,T) @ (B,T,hs) -> (B,T,hs)
        return out
    
class MultiHeadAttention(nn.Module):
    """Multiple heads of self attention in parallel"""

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size*num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out
    
class FeedForward(nn.Module):
    """ A Linear layer followed by some non linearity"""

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4*n_embd),
            nn.ReLU(),
            nn.Linear(4*n_embd, n_embd),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)
    
class Block(nn.Module):
    
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffed = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffed(self.ln2(x))
        return x
    
class GPTLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

        self.apply(self._init_weights)

      # init used in GPT for stable weights
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) #(T,C)
        x = tok_emb + pos_emb # (B,T,C)
        x = self.blocks(x)    # (B,T,C)
        x = self.ln_f(x)      # (B,T,C)
        logits = self.lm_head(x) # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss
             
    def generate(self, idx, max_new_tokens):

        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, loss = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1) # (B,1)
            idx = torch.cat((idx, idx_next), dim=1) # (B,T+1)
        return idx
    
model = GPTLanguageModel()
model.to(device)

# Number of parameters in the Model
print(sum(p.numel() for p in model.parameters())/1e6, 'M parameters')

# Optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# Training Loop
for iter in range(max_iters):

    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    xb, yb = get_batch('train')

    logits, loss = model(xb, yb)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Generate from model
context = torch.zeros((1,1), dtype=torch.long, device=device)
print(decode(model.generate(context, max_new_tokens=args.sample_tokens)[0].tolist()))