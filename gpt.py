import torch

# https://pytorch.org/docs/stable/nn.html
import torch.nn as nn

# https://pytorch.org/docs/stable/nn.functional.html
from torch.nn import functional as f


# hyperparameters
BATCH_SIZE = 32  # how many independent sequences we will process in parallel
BLOCK_SIZE = 8  # maximum context length for predictions
MAX_ITERS = 3000
EVAL_ITERS = 200
EVAL_INTERVAL = 300
LEARNING_RATE = 1e-2
N_EMBEDS = 32
N_HEAD = 6
N_LAYER = 6
DROPOUT = 0.2
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


torch.manual_seed(1337)


with open('./data/STAR WARS THE PHANTOM MENACE.txt', 'r', encoding='utf-8') as f:
    text = f.read()


# get all the unique characters that occur in this text
chars = sorted(list(set(text)))  # set gets the unique chars, turn into list, then sort
vocab_size = len(chars)

# create mapping between characters and integers
atoi = { ch:i for i, ch in enumerate(chars) }
encode = lambda a: [atoi[c] for c in a]  # given a string, output a list of integers

itoa = { i:ch for i, ch in enumerate(chars) }
decode = lambda l: ''.join([itoa[i] for i in l])  # given a list of integers, output a string


# create train and test splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))  # first 90% will be training data, the rest for validation
train_data = data[:n]
val_data = data[n:]


# generate a small batch of data of inputs x and targets y
def get_batch(split):
    data = train_data if split == 'train' else val_data
    # generates 0 to data - block, shape of [x, y, z] with batch size amount of entries
    i_x = torch.randint(len(data) - BLOCK_SIZE, (BATCH_SIZE,)) 
    
    # torch.stack joins a sequences of matrices (tensors) 
    x = torch.stack([data[i:i + BLOCK_SIZE] for i in i_x])  # gets input for block
    y = torch.stack([data[i + 1:i + BLOCK_SIZE] for i in i_x])  # gets target (next value) for block
    x, y = x.to(DEVICE), y.to(DEVICE)
    return x, y

@torch.no_grad()  # disable gradient descent calculation to reduce memory consumption (no back prop)
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(EVAL_ITERS)
        for k in range(EVAL_ITERS):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


# One head of Self-Attention
class Head(nn.Module):

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(N_EMBEDS, head_size, bias=False)
        self.query = nn.Linear(N_EMBEDS, head_size, bias=False)
        self.value = nn.Linear(N_EMBEDS, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)  # (B, T, head size)
        q = self.query(x)  # (B, T, head size)

        # compute attention scores
        wei = q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5  # (B, T, head size) @ (B, head size, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))  # (B, T, T)
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)

        v = self.value(x)
        out = wei @ v  # (B, T, T) @ (B, T, hs) -> (B, T, hs)
        return out