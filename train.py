import torch
import torch.nn as nn
from torch.nn import functional as F

import gpt_config as config
import data_utils as data_utils
from gpt_language_model import GPTLanguageModel

def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = data_utils.train_data if split == 'train' else data_utils.val_data
    ix = torch.randint(len(data) - config.block_size, (config.batch_size,))
    x = torch.stack([data[i:i+config.block_size] for i in ix])
    y = torch.stack([data[i+1:i+config.block_size+1] for i in ix])
    x, y = x.to(config.device), y.to(config.device)
    return x, y

@torch.no_grad()
def estimate_loss(model):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(config.eval_iters)
        for k in range(config.eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

def train_model():
    model = GPTLanguageModel()
    m = model.to(config.device)
    print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

    for iter in range(config.max_iters):

        # every once in a while evaluate the loss on train and val sets
        if iter % config.eval_interval == 0 or iter == config.max_iters - 1:
            losses = estimate_loss(model=model)
            print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

        # sample a batch of data
        xb, yb = get_batch('train')

        # evaluate the loss
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    
    torch.save(model.state_dict(), "model/friendsGPT.pth")
    print("Model Saved")

if __name__ == "__main__":
    train_model()