from pathlib import Path
import torch
import torch.nn as nn
from torch.nn import functional as F

def load_data():
    with open(Path(__file__).parent / "input.txt", 'r') as f:
        return f.read()

torch.manual_seed(1337)
text = load_data()
unique_text = sorted(set(text))

# Hyperparams
BATCH_SIZE = 32
HEAD_SIZE = 32
BLOCK_SIZE = 8
N_EMBD = 32 # Embedding dimension
EVAL_ITERS = 200
LEARNING_RATE = 1e-3
LEARNING_STEPS = 5000
VOCAB_SIZE = len(unique_text)

def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - BLOCK_SIZE, (BATCH_SIZE,))
    x = torch.stack([data[i:i+BLOCK_SIZE] for i in ix]) # 
    y = torch.stack([data[i+1:i+BLOCK_SIZE+1] for i in ix])
    return x, y

@torch.no_grad()
def estimate_loss(model):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(EVAL_ITERS)
        for k in range(EVAL_ITERS):
            X, Y = get_batch(split)
            _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

class Head(nn.Module):
    def __init__(self, head_size) -> None:
        super().__init__()
        self.head_size = head_size
        self.key = nn.Linear(N_EMBD, self.head_size, bias=False)
        self.value = nn.Linear(N_EMBD, self.head_size, bias=False)
        self.query = nn.Linear(N_EMBD, self.head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(BLOCK_SIZE, BLOCK_SIZE)))
    
    def forward(self, x):
        B, T, C = x.shape

        k = self.key(x)
        q = self.query(x)
        v = self.value(x)

        wei = q @ k.transpose(-1,-2) # -> (B, T, T)
        wei = wei / (self.head_size) ** 0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        out = wei @ v
        return out

class

class BigramLanguageModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.token_embedding_table = nn.Embedding(VOCAB_SIZE, N_EMBD) # (65 unique text, embedding dimension)
        self.position_embedding_table = nn.Embedding(BLOCK_SIZE, N_EMBD) # (BLOCK_SIZE positions in X, embedding dimension))
        self.sa_head = Head(HEAD_SIZE)
        self.lm_head = nn.Linear(N_EMBD, VOCAB_SIZE) # output -> (embedding dimension, choose 1 out of 65 unique text)
    
    def forward(self, x, target=None):
        B, T = x.shape
        token_embedding = self.token_embedding_table(x) # (B, T, N_EMBD)
        position_embedding = self.position_embedding_table(torch.arange(T)) # (T, N_EMBD)
        x = token_embedding + position_embedding
        x = self.sa_head(x)
        logit = self.lm_head(x) # (B,T,VOCAB_SIZE)
        
        if target == None:
            loss = None
        else:
            B, T, C = logit.shape
            logit = logit.view(B*T, C)
            target = target.view(B*T)
            loss = F.cross_entropy(logit, target)
        return logit, loss
        
    def generate(self, x, max_token):
        for _ in range(max_token):
            x_cond = x[:, -BLOCK_SIZE:]
            logit, _ = self(x_cond)
            logit = logit[:, -1, :]
            prob = F.softmax(logit, dim=-1)
            sample = torch.multinomial(prob, num_samples=1)
            x = torch.cat((x, sample), dim=1)
        return x

def train(model):
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    for _ in range(LEARNING_STEPS):
        optimizer.zero_grad(set_to_none=True)
        x, y = get_batch('train')
        _, loss = model(x, y)
        loss.backward()
        optimizer.step()
    print("Loss is ", estimate_loss(model))

itos= {ch: i for ch, i in enumerate(unique_text)}
stoi= {i: ch for ch, i in enumerate(unique_text)}

encode = lambda s:[stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

data = torch.tensor(encode(text), dtype=torch.long)

n = int(0.9 * len(data))
train_data = data[:n] 
val_data = data[n:] 

bi = BigramLanguageModel()

train(bi)

x = encode("Emily is ")
idx = torch.tensor([x])
out = bi.generate(idx, 200)[0].tolist()
print(decode(out))

