import torch
import torch.nn as nn
from torch.nn import functional as F
from model import GPTLanguageModel
from model import device

################
inputFile='pan-tadeusz'
learing_iters=1500

with open(inputFile+'.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)
# create a mapping from characters to integers
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]  # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l])  # decoder: take a list of integers, output a string


# Later, to load the model
loaded_model = GPTLanguageModel(vocab_size)  # Make sure to instantiate the model with the same parameters
loaded_model.load_state_dict(torch.load('gpt_'+inputFile+str(learing_iters)+'.pth'))
loaded_model = loaded_model.to(device)
loaded_model.eval()  # Make sure to call .eval() if you're using the model for inference

# Now you can use the loaded_model for inference
context = torch.zeros((1, 1), dtype=torch.long, device=device)
while(True):
    print(decode(loaded_model.generate(context, max_new_tokens=500)[0].tolist()))