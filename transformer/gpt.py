import torch
import torch.nn as nn
from torch.nn import functional as F
from model import GPTLanguageModel, loadModel, saveModel,block_size, batch_size,device
from pathlib import Path
import os

################
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# Run time parameters
input_file_name = 'panTadeusz.txt'
model_folder = 'panTadeusz_model'
max_iters = 500
saved_model_name = model_folder+'.pth'
mode = 'train'
# mode='load'
# learning params
eval_interval = 500
eval_iters = 200
learning_rate = 3e-5
################



# Paths
current_dir = os.path.dirname(os.path.abspath(__file__))
input_file_path = os.path.join(current_dir, input_file_name)
saved_model_path = os.path.join(current_dir, model_folder, saved_model_name)

print("Runnunig on: ",device)
print("Input file path: ",input_file_path)
print("Saved model path: ",saved_model_path)
################
# Load the input
with open(input_file_path, 'r', encoding='utf-8') as f:
    text = f.read()

# instantiate the model
vocabulary=[]
if Path(saved_model_path).exists():
    print("Found a model and loading it for further trainning.")
    model,vocabulary  = loadModel(saved_model_path)
    vocab_size=len(vocabulary)
else:
    # here are all the unique characters that occur in this text
    vocabulary = sorted(list(set(text)))
    vocab_size = len(vocabulary)
    print("Number of possible tokens: "+str(vocab_size)) 
    # create a mapping from characters to integers
    print("Creating a new model.")
    model= GPTLanguageModel(vocab_size)   

# encode and decode functions for strings to tensors 
print("Vocabulary size: ",vocab_size)
stoi = {ch: i for i, ch in enumerate(vocabulary)}
itos = {i: ch for i, ch in enumerate(vocabulary)}
encode = lambda s: [stoi[c] for c in s]  # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l])  # decoder: take a list of integers, output a string

# Train and test splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))  # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]

def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y
################

if(mode=='train'):
    m = model.to(device)
    # create a PyTorch optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    # print the number of parameters in the model
    print(sum(p.numel() for p in m.parameters()) / 1e6, 'M parameters')
    print("The number of learing steps:",max_iters)
    for iter in range(max_iters):

        # every once in a while evaluate the loss on train and val sets
        if iter % eval_interval == 0 or iter == max_iters - 1:
            losses = estimate_loss()
            print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
            # Save the model
            saveModel(model, vocabulary, model_folder, saved_model_name)

        # sample a batch of data
        xb, yb = get_batch('train')

        # evaluate the loss
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    saveModel(model, vocabulary, model_folder, saved_model_name)
            
# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(model.generate(context, max_new_tokens=500)[0].tolist()))

