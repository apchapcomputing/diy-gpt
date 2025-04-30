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