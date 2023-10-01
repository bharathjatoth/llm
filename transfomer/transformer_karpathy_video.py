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
import torch.nn as nn
from torch.nn import functional as F
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
#
# '''
# Down here we made the file more as a script of 122 lines where we introduced a function which will give the loss automatically
# and if u have a gpu will get it and calculate it on a simple Bigram model we have not yet added softmax or a neural net or anything
# '''
#
# import torch
# import torch.nn as nn
# from torch.nn import functional as F
#
# # hyperparameters
# batch_size = 32 # how many independent sequences will we process in parallel?
# block_size = 8 # what is the maximum context length for predictions?
# max_iters = 3000
# eval_interval = 300
# learning_rate = 1e-2
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# eval_iters = 200
# # ------------
#
# torch.manual_seed(1337)
#
# # wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
# with open('input.txt', 'r', encoding='utf-8') as f:
#     text = f.read()
#
# # here are all the unique characters that occur in this text
# chars = sorted(list(set(text)))
# vocab_size = len(chars)
# # create a mapping from characters to integers
# stoi = { ch:i for i,ch in enumerate(chars) }
# itos = { i:ch for i,ch in enumerate(chars) }
# encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
# decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string
#
# # Train and test splits
# data = torch.tensor(encode(text), dtype=torch.long)
# n = int(0.9*len(data)) # first 90% will be train, rest val
# train_data = data[:n]
# val_data = data[n:]
#
# # data loading
# def get_batch(split):
#     # generate a small batch of data of inputs x and targets y
#     data = train_data if split == 'train' else val_data
#     ix = torch.randint(len(data) - block_size, (batch_size,))
#     x = torch.stack([data[i:i+block_size] for i in ix])
#     y = torch.stack([data[i+1:i+block_size+1] for i in ix])
#     x, y = x.to(device), y.to(device)
#     return x, y
#
# @torch.no_grad()
# def estimate_loss():
#     out = {}
#     model.eval()
#     for split in ['train', 'val']:
#         losses = torch.zeros(eval_iters)
#         for k in range(eval_iters):
#             X, Y = get_batch(split)
#             logits, loss = model(X, Y)
#             losses[k] = loss.item()
#         out[split] = losses.mean()
#     model.train()
#     return out
#
# # super simple bigram model
# class BigramLanguageModel(nn.Module):
#
#     def __init__(self, vocab_size):
#         super().__init__()
#         # each token directly reads off the logits for the next token from a lookup table
#         self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)
#
#     def forward(self, idx, targets=None):
#
#         # idx and targets are both (B,T) tensor of integers
#         logits = self.token_embedding_table(idx) # (B,T,C)
#
#         if targets is None:
#             loss = None
#         else:
#             B, T, C = logits.shape
#             logits = logits.view(B*T, C)
#             targets = targets.view(B*T)
#             loss = F.cross_entropy(logits, targets)
#
#         return logits, loss
#
#     def generate(self, idx, max_new_tokens):
#         # idx is (B, T) array of indices in the current context
#         for _ in range(max_new_tokens):
#             # get the predictions
#             logits, loss = self(idx)
#             # focus only on the last time step
#             logits = logits[:, -1, :] # becomes (B, C)
#             # apply softmax to get probabilities
#             probs = F.softmax(logits, dim=-1) # (B, C)
#             # sample from the distribution
#             idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
#             # append sampled index to the running sequence
#             idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
#         return idx
#
# model = BigramLanguageModel(vocab_size)
# m = model.to(device)
#
# # create a PyTorch optimizer
# optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
#
# for iter in range(max_iters):
#
#     # every once in a while evaluate the loss on train and val sets
#     if iter % eval_interval == 0:
#         losses = estimate_loss()
#         print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
#
#     # sample a batch of data
#     xb, yb = get_batch('train')
#
#     # evaluate the loss
#     logits, loss = model(xb, yb)
#     optimizer.zero_grad(set_to_none=True)
#     loss.backward()
#     optimizer.step()
#
# # generate from the model
# context = torch.zeros((1, 1), dtype=torch.long, device=device)
# print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))


#the mathemetical trick is in self-attention
# we want to make each token to speak with each other
# example 5th location token should not speak with 6,7,8 beacause these are future token
# we need the 5th token to speak with only with the 1,2,3,4 tokens as these are the past tokens
# so the easiest way to communicate with past is to do an average of the preceeding tokens
import torch
#
# torch.manual_seed(1337)
# B,T,C = 4,8,2
# x = torch.randn(B,T,C)
# print(x.shape)
#
# #bow bagofwords averaging up things of previous words
# #we get the average of the previous in the xbow of the x torch
# xbow = torch.zeros((B,T,C))
# for b in range(B):
#     for t in range(T):
#         xprev = x[b,:t+1] #shape(t,C)
#         xbow[b,t] = torch.mean(xprev, 0)
# print(x[0],xbow[0])
#
# #version 1 as an example
# #a toy example using the matrix multiplication
# torch.manual_seed(42)
# a = torch.tril(torch.ones(3,3))
# a = a/torch.sum(a,1,keepdim=True)
# b = torch.randint(0,10,(3,2)).float()
# c = a@b # 3*3 * 3*2 = 3*2 matrix
# #now we get the average of the matrix with the help of tril and averaging the a matrix
# print(f"a = {a}")
# print(f"b = {b}")
# print(f"c = {c}")
#
# #now we vectorize and see what we get
# #now we get the information of the tokens with the preceeding only
# #second version
# wei = torch.tril(torch.ones(T,T))
# wei = wei/wei.sum(1,keepdim = True)
# xbow2 = wei@x # (B,T,T) * (B,T,C) --> (B,T,C)
# print(torch.allclose(xbow,xbow2))
# print(xbow,xbow2)
#
# #third version of the same
# tril = torch.tril(torch.ones(T,T))
# wei = torch.zeros((T,T))
# wei = wei.masked_fill(tril ==0,float('-inf'))
# print(wei)
# wei = F.softmax(wei,dim=1)#softmax is also a normalization operation i.e., e^all row elements/mask
# print(wei)
# xbow3 = wei@x
# print(torch.allclose(xbow,xbow3))

#attention single headed implementation (single head perfom self-attention)
#version 4: self:attention using key,query and value:
'''
key : what I have
query: what I am looking for
value: what info I can reveal to
'''
torch.manual_seed(1337)
B,T,C = 4,8,32 #batch,time(block),channels
x = torch.randn(B,T,C)

head_size = 16
key = nn.Linear(C,head_size,bias=False)
query = nn.Linear(C,head_size,bias=False)
value = nn.Linear(C,head_size,bias=False)
'''
here key query and value come from the 
attention: 
self-attention: key, query and value come from same source so the nodes are self-attended the nodes talk to each other
cross-attention: queries are produced from x, key and value come from encoder this is encode-decoder network here we only produce query
key and value come from inputs and just produce query
'''
k = key(x) # size B,T,16
q = query(x) # size B,T,16
# v = value(x)
wei = q @ k.transpose(-2,-1) # B,T,16 matrix multiply with B,16,T gives B,T,T

tril = torch.tril(torch.ones(T,T))
# wei = torch.zeros((T,T))
'''
The wei line of code "wei = wei.masked_fill(tril == 0 ,float('-inf'))"
can be deleted if we need an encoder block just in case if we are doing for a sentiment analysis task using a transformer
because in sentiment analysis all the information needs to be viewed by all the nodes the above line of code will be in the decoder as the future information should not be seen by the present because we are predicting the future tasks or information
'''
wei = wei.masked_fill(tril == 0 ,float('-inf'))
wei = F.softmax(wei,dim=-1)

v = value(x)
out = wei@v
print(out.shape)
print(wei)

'''
the scaling of dividing by sqrt of headsize is done so that the variance is preserved
what is varience preserved? when we pass the values to softmax it will maximize the values with max i.e., in the given example the values with highest
will given most priority than the values of less priority to avoid this and maintain the variance all over the data set we will divide the q.k transpose * value by the sqrt of headsize
this is explained in the video of karpathy at 1:19:00
the example is as follows
'''
k = torch.randn(B,T,head_size)
q = torch.randn(B,T,head_size)
wei = q @ k.transpose(-2,-1) * head_size**-0.5

print(k.var(),q.var(),wei.var())
#example we are speaking above
print(torch.softmax(torch.tensor([0.1, -0.2, 0.3, -0.2, 0.5]), dim=-1))
print(torch.softmax(torch.tensor([0.1, -0.2, 0.3, -0.2, 0.5])*9, dim=-1))#it is maximizing the highest value so we are dividing with the sqrt of the headsize

#The next thing is multi headed attention
'''
What is multiheaded attention? multiple self- attentions in parallel
'''




