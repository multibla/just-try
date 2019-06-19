#!/usr/bin/env python
# coding: utf-8

# # 8. Convolutional Neural Networks

# I recommend you take a look at these material first.

# * http://web.stanford.edu/class/cs224n/lectures/cs224n-2017-lecture13-CNNs.pdf
# * http://www.aclweb.org/anthology/D14-1181
# * https://github.com/Shawn1993/cnn-text-classification-pytorch
# * http://cogcomp.org/Data/QA/QC/

# In[58]:


import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F
import nltk
import random
import numpy as np
from collections import Counter, OrderedDict
import nltk
import re
from copy import deepcopy
import pdb

flatten = lambda l: [item for sublist in l for item in sublist]
random.seed(1024)


# In[2]:


USE_CUDA = torch.cuda.is_available()

FloatTensor = torch.cuda.FloatTensor if USE_CUDA else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if USE_CUDA else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if USE_CUDA else torch.ByteTensor


# In[3]:


def getBatch(batch_size, train_data):
    random.shuffle(train_data)
    sindex = 0
    eindex = batch_size
    while eindex < len(train_data):
        batch = train_data[sindex: eindex]
        temp = eindex
        eindex = eindex + batch_size
        sindex = temp
        yield batch
    
    if eindex >= len(train_data):
        batch = train_data[sindex:]
        yield batch


# In[110]:


def pad_to_batch(batch):
    x,y = zip(*batch)
    max_x = max([s.size(1) for s in x])
    x_p = []
    for i in range(len(batch)):
        if x[i].size(1) < max_x:
            x_p.append(torch.cat([x[i], Variable(LongTensor([word2index['<PAD>']] * (max_x - x[i].size(1)))).view(1, -1)], 1))
        else:
            x_p.append(x[i])
    return torch.cat(x_p), torch.cat(y).view(-1)


# In[20]:


def prepare_sequence(seq, to_index):
    idxs = list(map(lambda w: to_index[w] if to_index.get(w) is not None else to_index["<UNK>"], seq))
    return Variable(LongTensor(idxs))


# ## Data load & Preprocessing

# ### TREC question dataset(http://cogcomp.org/Data/QA/QC/)

# Task involves
# classifying a question into 6 question
# types (whether the question is about person,
# location, numeric information, etc.)

# In[53]:


data = open('../dataset/train_5500.label.txt', 'r', encoding='latin-1').readlines()


# In[54]:


data = [[d.split(':')[1][:-1], d.split(':')[0]] for d in data]


# In[61]:


X, y = list(zip(*data))
X = list(X)


# ### Num masking 

# It reduces the search space. ex. my birthday is 12.22 ==> my birthday is ##.##

# In[62]:


for i, x in enumerate(X):
    X[i] = re.sub('\d', '#', x).split()


# ### Build Vocab 

# In[63]:


vocab = list(set(flatten(X)))


# In[64]:


len(vocab)


# In[31]:


len(set(y)) # num of class


# In[94]:


word2index={'<PAD>': 0, '<UNK>': 1}

for vo in vocab:
    if word2index.get(vo) is None:
        word2index[vo] = len(word2index)
        
index2word = {v:k for k, v in word2index.items()}

target2index = {}

for cl in set(y):
    if target2index.get(cl) is None:
        target2index[cl] = len(target2index)

index2target = {v:k for k, v in target2index.items()}


# In[95]:


X_p, y_p = [], []
for pair in zip(X,y):
    X_p.append(prepare_sequence(pair[0], word2index).view(1, -1))
    y_p.append(Variable(LongTensor([target2index[pair[1]]])).view(1, -1))
    
data_p = list(zip(X_p, y_p))
random.shuffle(data_p)

train_data = data_p[: int(len(data_p) * 0.9)]
test_data = data_p[int(len(data_p) * 0.9):]


# ### Load Pretrained word vector

# you can download pretrained word vector from here https://github.com/mmihaltz/word2vec-GoogleNews-vectors 

# In[41]:


import gensim


# In[43]:


model = gensim.models.KeyedVectors.load_word2vec_format('../dataset/GoogleNews-vectors-negative300.bin', binary=True)


# In[48]:


len(model.index2word)


# In[96]:


pretrained = []

for key in word2index.keys():
    try:
        pretrained.append(model[word2index[key]])
    except:
        pretrained.append(np.random.randn(300))
        
pretrained_vectors = np.vstack(pretrained)


# ## Modeling 

# <img src="../images/08.cnn-for-text-architecture.png">
# <center>borrowed image from http://www.aclweb.org/anthology/D14-1181</center>

# In[117]:


class  CNNClassifier(nn.Module):
    
    def __init__(self, vocab_size, embedding_dim, output_size, kernel_dim=100, kernel_sizes=(3,4,5), dropout=0.5):
        super(CNNClassifier,self).__init__()

        self.embedding_static = nn.Embedding(vocab_size, embedding_dim)
        self.embedding_nonstatic = nn.Embedding(vocab_size, embedding_dim)
        self.convs = nn.ModuleList([nn.Conv2d(2, kernel_dim, (K, embedding_dim)) for K in kernel_sizes])

        # kernal_size = (K,D) 
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(len(kernel_sizes) * kernel_dim, output_size)
    
    
    def init_weights(self, pretrained_word_vectors, is_static=True):
        self.embedding_nonstatic.weight = nn.Parameter(torch.from_numpy(pretrained_word_vectors).float())
        self.embedding_static.weight = nn.Parameter(torch.from_numpy(pretrained_word_vectors).float())
        if is_static:
            self.embedding_static.weight.requires_grad = False


    def forward(self, inputs, is_training=False):
#        inputs = self.embedding(inputs).unsqueeze(1) # (B,1,T,D)
        inputs_static = self.embedding_static(inputs).unsqueeze(1)
        inputs_nonstatic = self.embedding_nonstatic(inputs).unsqueeze(1)
        inputs = torch.cat((inputs_static, inputs_nonstatic), 1)

        inputs = [F.relu(conv(inputs)).squeeze(3) for conv in self.convs] #[(N,Co,W), ...]*len(Ks)
        inputs = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in inputs] #[(N,Co), ...]*len(Ks)

        concated = torch.cat(inputs, 1)

        if is_training:
            concated = self.dropout(concated) # (N,len(Ks)*Co)
        out = self.fc(concated) 
        return F.log_softmax(out,1)


# ## Train 

# It takes for a while if you use just cpu.

# In[145]:


EPOCH = 5
BATCH_SIZE = 50
KERNEL_SIZES = [3,4,5]
KERNEL_DIM = 100
LR = 0.001


# In[146]:


model = CNNClassifier(len(word2index), 300, len(target2index), KERNEL_DIM, KERNEL_SIZES)
model.init_weights(pretrained_vectors) # initialize embedding matrix using pretrained vectors

if USE_CUDA:
    model = model.cuda()
    
loss_function = nn.CrossEntropyLoss()
#optimizer = optim.Adam(model.parameters(), lr=LR)
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LR)


# In[147]:


for epoch in range(EPOCH):
    losses = []
    for i,batch in enumerate(getBatch(BATCH_SIZE, train_data)):
        inputs,targets = pad_to_batch(batch)
        model.zero_grad()
        preds = model(inputs, True)
        loss = loss_function(preds, targets)
        losses.append(loss.item())
        loss.backward()
        
        #for param in model.parameters():
        #    param.grad.data.clamp_(-3, 3)
        
        optimizer.step()
        
        if i % 100 == 0:
            print("[%d/%d] mean_loss : %0.2f" %(epoch, EPOCH, np.mean(losses)))
            losses = []


# ## Test 

# In[150]:


accuracy = 0


# In[151]:

# TODO: input needs padding when senstence is shorter than kernal size
n = 0
print(len(test_data))
for test in test_data:
    try:
       pred = model(test[0]).max(1)[1]
       pred = pred.item()
       target = test[1].item()
       if pred == target:
           accuracy += 1
       n += 1
    except:
       print(test[0])
       print(n)
print(accuracy/len(test_data) * 100)
