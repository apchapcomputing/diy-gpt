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


class BigramLanguageModel(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        # nn.Embedding: A simple lookup table that stores embeddings of a fixed dictionary and size.
        self.token_embedding_table = nn.Embedding(vocab_size, N_EMBEDS)
        self.position_embedding_table = nn.Embedding(BLOCK_SIZE, N_EMBEDS)
        self.lm_head = nn.Linear(N_EMBEDS, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx and targets are both (B,T) tensors of integers
        token_embeddings = self.token_embedding_table(idx)  # (B,T,C)
        position_embeddings = self.position_embedding_table(torch.arange(T, device=DEVICE))  # (T,C)
        x = token_embeddings + position_embeddings  # (B,T,C)
        logits = self.lm_head(x)  # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B,T) array of indices in the current context
        for _ in range(max_new_tokens):
            logits, loss = self(idx)  # get the predictions
            logits = logits[:, -1, :]  # focus only on the last step, becomes (B,C)
            probs = F.softmax(logits, dim=-1)  # apply softmax to get probabilities, (B,C)
            idx_next = torch.multinomial(probs, num_samples=1)  # sample from the distribution, (B,1)
            idx = torch.cat((idx, idx_next), dim=1)  # append sampled index to the running sequence, (B,T+1)
        
        return idx

model = BigramLanguageModel(vocab_size)
m = model.to(DEVICE)

optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

for i in range(MAX_ITERS):

    # every once in a while evaluate the loss on train and val sets
    if i % EVAL_INTERVAL == 0:
        losses = estimate_loss()
        print(f"step {i}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    xb, yb = get_batch('train')  # sample a batch of data

    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# generate from the model
context = torch.zeros((1,1), dtype=torch.long, DEVICE=DEVICE)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))

