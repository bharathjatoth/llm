'''
This is the total transformer model i.e., the decoder model of the Chat GPT-3 mentioned in the video by karpathy 
Rough code used for the notebook transformer decoder GPT3
Two files bigram.pyand transformer_karpathy.py are two files where in all the understanding is being coded and presented
Kindly refer these two files and the video if any more doubts araise
'''
# with open(r'C:\Users\bhara\Downloads\input.txt','r',encoding='utf-8') as f:
#     text = f.read()
#
# # print(len(text),text[:100])
#
# #we see the len of the text and all the chars as we  are building char by char
# chars = sorted(list(set(text)))
# vocab_size = len(chars)
# print(''.join(chars))
# print(vocab_size)
#
# #create a mapping from chars to int
# # we can also use sentencecode or tiktoken encoder to encode the words
# stoi = { ch:i for i,ch in enumerate(chars) }
# itos = { i:ch for i,ch in enumerate(chars) }
# encode = lambda s: [stoi[c] for c in s] #encoder: take the string and get the chars
# decode = lambda l: ''.join([itos[i] for i in l])
# #we can use any way of the encoder or decoder based on out availability and the pre loaded too
# print(encode('bharath is great')) # these are the simplest encoder and decoder that can be done
# print(decode(encode('bharath is great'))) #next assignment build these using the sentensecode or tike token
#
# import torch
# #we are going to make embeddings of all the text data using the torch library
# data = torch.tensor(encode(text),dtype=torch.long)
# print(data.shape,data.dtype)
# # print(text[:100])
# # print(data[:100])
#
# #split into train and validation data
# n = int(0.9*len(data))
# train_data = data[:n]
# val_data = data[n:]
#
# #start plugging these text into the transformer
# # we train the transformer with the chunks of train data as giving the whole data at once
# #is problamatic so we go with the block_size refer the paper
# # we also have the batch size i.e., we blocks in each batch and give the examples to the transformer
# block_size = 8
# print(train_data[:block_size+1])
# #we train on all the examples given as transformer to see on all the examples
# '''
# tranf will see all the text till the context of one char
# transformer will get to use to predict the chunk of block size
# if the transf gets more size we chunk it as it will not predict more than the block size
# '''
# x = train_data[:block_size]
# y = train_data[1:block_size+1]
# for t in range(block_size):
#     context = x[:t+1]
#     target = y[t]
#     print(f"when the input is {context} the target is : {target}")
#
# '''
# batches are used to make the gpu busy and make the processing easy and simple instead of
# giving the data all at once
# the following code gives the batch dimesion
# '''
# torch.manual_seed(1337) #to get the random chunks
# batch_size = 4 #how many independant sequences will we process in parallel
# block_size = 8
#
# def get_batch(split):
#     #generate small batches of data
#     data = train_data if split == 'train' else val_data
#     ix = torch.randint(len(data) - block_size, (batch_size,))
#     x = torch.stack([data[i:i+block_size] for i in ix])
#     y = torch.stack([data[i+1:i+block_size+1] for i in ix])
#     return x,y
# xb,yb = get_batch('train')
# print(f'inputs shape are {xb.shape} and {xb}')
# print(f'target are {yb.shape} and {yb}')
# '''now the transformer will get the xb 32 exmples of one batch of 4*8 as input and predict
# the output
# '''
# for b in range(batch_size):
#     for t in range(block_size):
#         context = xb[b, :t+1]
#         target = yb[b,t]
#         print(f"when input is {context.tolist()} the target is {target}")
#
# print(xb) # out input to the transformer
#
# import torch.nn as nn
# from torch.nn import functional as F
# torch.manual_seed(1337)
#
# class BigramLanguageModel(nn.Module):
#     def __init__(self,vocab_size):
#         super().__init__()
#         #each token directly reads off the logits for the next token from a lookup table
#         self.token_embedding_table = nn.Embedding(vocab_size,vocab_size)
#
#     def forward(self,idx,targets=None):
#         logits = self.token_embedding_table(idx) # 4 * 8 * 65
#         #but pytorch needs inputs in b*c*t format
#         # i.e., in 4*65*8 shape so reshape and feed it
#         if targets is None:
#             loss = None
#         else:
#             B,T,C = logits.shape
#             logits = logits.view(B*T,C) #32*65
#             targets = targets.view(B*T)
#             loss = F.cross_entropy(logits,targets)
#
#         return logits,loss
#
#     def generate(self,idx,max_new_tokens):
#         #idx is (B,T) array fo indices
#         #take (B,T) and generate (B,T+1) or any dimension
#         for _ in range(max_new_tokens):
#             #get the predictions
#             logits,loss = self(idx)
#             #focus only on last time step
#             logits = logits[:,-1,:] #Becomes (B,C) we pluck out T from B,T,C
#             #apply softmax to get probabilities
#             probs = F.softmax(logits,dim=1) #(B,C)
#             #sample from the distribution
#             idx_next = torch.multinomial(probs,num_samples=1) # give one sample
#             #append sampled index to running sequence
#             idx = torch.cat((idx,idx_next),dim=1) #(B,T+1)
#
#         return idx
#
# m = BigramLanguageModel(vocab_size)
# logits,loss = m(xb,yb)
# print(logits.shape,loss)
#
# idx = torch.zeros((1,1), dtype=torch.long)
# print(decode(m.generate(idx,max_new_tokens=100)[0].tolist()))
#
# #training
# optimizer = torch.optim.AdamW(m.parameters(),lr=1e-3)
#
# batch_size = 32
# for steps in range(10000):
#     #sample a new batch of data
#     xb,yb = get_batch('train')
#     #evaluate the loss
#     logits,loss = m(xb,yb)
#     optimizer.zero_grad(set_to_none=True)#zero gradients
#     loss.backward()
#     optimizer.step()
#
# print(loss.item())
# idx = torch.zeros((1,1), dtype=torch.long)
# print(decode(m.generate(idx,max_new_tokens=100)[0].tolist()))

'''
Down here we made the file more as a script of 122 lines where we introduced a function which will give the loss automatically
and if u have a gpu will get it and calculate it on a simple Bigram model we have not yet added softmax or a neural net or anything
In the transformer paper we apply layer norm after the attention but in now a days we
apply the transformation first then we do the attention - prenorm formulation
'''

import torch
import torch.nn as nn
from torch.nn import functional as F

# hyperparameters
batch_size = 32 # how many independent sequences will we process in parallel?
block_size = 4 # what is the maximum context length for predictions?
max_iters = 5000
eval_interval = 100
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 32
n_head = 8
n_layer = 2
dropout = 0.2
# ------------

torch.manual_seed(1337)

# wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
with open(r'C:\Users\bhara\Downloads\input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)
# create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

# Train and test splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data)) # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]

class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        '''
        Here we are using a decoder only block where the key query and value come from random and we make the tril version
        but in original encoder to decoder network paper we apply the cross attention where in
        the key and value come from the encoder block to every decoder block and query is generated from the decoder block
        this is very useful when we don't want to have the generate own text or for a machine translation
        This is the only change from the original transformer
        This is exactly done in GPT decoder only
        for chat gpt we have 2 states pretraining (decoder only network and using subword chunks of data rather than our model)
        dmodel in paper = n_embd in out model nanoGPT focus on the pretraining stage
        and finetuning(but it generates documents but not question answering as it is decoder only network so we use finetuning
        it is actually to do align 2nd stage with 3 steps:

        )
        :param x:
        :return:
        '''
        B,T,C = x.shape
        k = self.key(x)   # (B,T,C)
        q = self.query(x) # (B,T,C)
        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2,-1) * C**-0.5 # (B, T, C) @ (B, C, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei) #randomly add so that it shuts communication between them
        # perform the weighted aggregation of the values
        v = self.value(x) # (B,T,C)
        out = wei @ v # (B, T, T) @ (B, T, C) -> (B, T, C)
        return out

class MultiHeadAttention(nn.Module):
    # The next thing is multi headed attention
    '''
    What is multiheaded attention? multiple self- attentions in parallel
    It is easy to be created in pytorch
    num_heads - no of heads we need to run in parallel
    head_size - head size of each head
    '''
    def __init__(self,num_heads,head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd,n_embd)
        self.dropout = nn.Dropout(dropout)
    def forward(self,x):
        #concatinate all the heads
        return torch.cat([h(x) for h in self.heads], dim=-1)
# data loading
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

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


class FeedForward(nn.Module):
    '''
    a simple feed  forward linear layer followed by a non-linearity
    it is on a token level every token does it
    '''
    def __init__(self,n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd,4*n_embd),
            nn.ReLU(),
            nn.Linear(4*n_embd,n_embd),
            nn.Dropout(dropout)
        )
    def forward(self,x):
        return self.net(x)

class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embd, n_head):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


# super simple bigram model
class BigramLanguageModel(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size,n_embd)
        '''
        now we have 4 heads run in paralled with each block size of 8 that is 32 vectors stacked up
        '''
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        # self.sa_heads = MultiHeadAttention(4, n_embd//4)
        # self.ffwd = FeedForward(n_embd)
        self.ln_f = nn.LayerNorm(n_embd) # last final layer in the block before linear and softmax
        self.lm_head = nn.Linear(n_embd,vocab_size)

    def forward(self, idx, targets=None):
        B,T = idx.shape
        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx)  # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))  # (T,C)
        x = tok_emb + pos_emb  # (B,T,C)
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)  # (B,T,vocab_size)
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            #crop_idx so that we get it to the last block_size tokens
            #so that we will never pass no more than the block_size elements
            idx_cond = idx[:,-block_size:]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx

model = BigramLanguageModel(vocab_size)
m = model.to(device)

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):

    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))
