#!/usr/bin/env python
# coding: utf-8

# # 4. Word Window Classification and Neural Networks 

# I recommend you take a look at these material first.

# * http://web.stanford.edu/class/cs224n/lectures/cs224n-2017-lecture4.pdf
# * https://en.wikipedia.org/wiki/Named-entity_recognition

# In[1]:


import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F
import nltk
import random
import numpy as np
from collections import Counter
flatten = lambda l: [item for sublist in l for item in sublist]
from sklearn_crfsuite import metrics
import pdb

random.seed(1024)

# You also need <a href="http://sklearn-crfsuite.readthedocs.io/en/latest/index.html">sklearn_crfsuite</a> latest version for print confusion matrix

# In[2]:


print(torch.__version__)
print(nltk.__version__)


# In[3]:


USE_CUDA = torch.cuda.is_available()
gpus = [0]
torch.cuda.set_device(gpus[0])

FloatTensor = torch.cuda.FloatTensor if USE_CUDA else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if USE_CUDA else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if USE_CUDA else torch.ByteTensor


# In[4]:


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


# In[5]:


def prepare_sequence(seq, word2index):
    idxs = list(map(lambda w: word2index[w] if word2index.get(w) is not None else word2index["<UNK>"], seq))
    return Variable(LongTensor(idxs))

def prepare_word(word, word2index):
    return Variable(LongTensor([word2index[word]]) if word2index.get(word) is not None else LongTensor([word2index["<UNK>"]]))

def prepare_tag(tag,tag2index):
    return Variable(LongTensor([tag2index[tag]]))


# ## Data load and Preprocessing 

# CoNLL-2002 Shared Task: Language-Independent Named Entity Recognition <br>
# https://www.clips.uantwerpen.be/conll2002/ner/

# In[6]:


corpus = nltk.corpus.conll2002.iob_sents()


# In[7]:

pdb.set_trace()

data = []
for cor in corpus:
    sent, _, tag = list(zip(*cor))
    data.append([sent, tag])


# In[8]:


print(len(data))
print(data[0])


# ### Build Vocab

# In[9]:


sents,tags = list(zip(*data))
vocab = list(set(flatten(sents)))
tagset = list(set(flatten(tags)))


# In[10]:


word2index={'<UNK>' : 0, '<DUMMY>' : 1} # dummy token is for start or end of sentence
for vo in vocab:
    if word2index.get(vo) is None:
        word2index[vo] = len(word2index)
index2word = {v:k for k, v in word2index.items()}

tag2index = {}
for tag in tagset:
    if tag2index.get(tag) is None:
        tag2index[tag] = len(tag2index)
index2tag={v:k for k, v in tag2index.items()}


# ### Prepare data

# <center>Example : Classify 'Paris' in the context of this sentence with window length 2</center>

# <img src="../images/04.window-data.png">

# <center>borrowed image from http://web.stanford.edu/class/cs224n/lectures/cs224n-2017-lecture4.pdf</center>

# In[11]:


WINDOW_SIZE = 2
windows = []


# In[12]:


for sample in data:
    dummy = ['<DUMMY>'] * WINDOW_SIZE
    window = list(nltk.ngrams(dummy + list(sample[0]) + dummy, WINDOW_SIZE * 2 + 1))
    windows.extend([[list(window[i]), sample[1][i]] for i in range(len(sample[0]))])


# In[13]:


windows[0]


# In[14]:


len(windows)


# In[15]:


random.shuffle(windows)

train_data = windows[:int(len(windows) * 0.9)]
test_data = windows[int(len(windows) * 0.9):]


# ## Modeling 

# <img src="../images/04.window-classifier-architecture.png">
# <center>borrowed image from http://web.stanford.edu/class/cs224n/lectures/cs224n-2017-lecture4.pdf</center>

# In[16]:


class WindowClassifier(nn.Module): 
    def __init__(self, vocab_size, embedding_size, window_size, hidden_size, output_size):

        super(WindowClassifier, self).__init__()
        
        self.embed = nn.Embedding(vocab_size, embedding_size)
        self.h_layer1 = nn.Linear(embedding_size * (window_size * 2 + 1), hidden_size)
        self.h_layer2 = nn.Linear(hidden_size, hidden_size)
        self.o_layer = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        self.softmax = nn.LogSoftmax(dim=1)
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, inputs, is_training=False): 
        embeds = self.embed(inputs) # BxWxD
        pdb.set_trace()
        concated = embeds.view(-1, embeds.size(1)*embeds.size(2)) # Bx(W*D)
        h0 = self.relu(self.h_layer1(concated))
        if is_training:
            h0 = self.dropout(h0)
        h1 = self.relu(self.h_layer2(h0))
        if is_training:
            h1 = self.dropout(h1)
        out = self.softmax(self.o_layer(h1))
        return out


class LstmCrfClassifier(nn.Module):
    def __init(self, vocab_size, embedding_size, hidden_size, num_layers, output_size):
        super(LstmCrfClassifier, self).__init__()

        self.embed = nn.Embedding(vocab_size, embedding_size)
        self.lstm = nn.LSTM(input_size=embedding_size, hidden_size=hidden_size, num_layers=num_layers, bidirectional=True)
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, inputs, is_training=False):
        embeds = self.embed(inputs)
        bilstm, (cn, hn) = self.lstm(embeds)
        output = self.linear(bilstm)
        return output


# In[20]:


BATCH_SIZE = 128
EMBEDDING_SIZE = 50 # x (WINDOW_SIZE*2+1) = 250
HIDDEN_SIZE = 300
EPOCH = 3
LEARNING_RATE = 0.001
LAYER_SIZE = 2

# ## Training 

# It takes for a while if you use just cpu.

# In[22]:


model = WindowClassifier(len(word2index), EMBEDDING_SIZE, WINDOW_SIZE, HIDDEN_SIZE, len(tag2index))
if USE_CUDA:
    model = model.cuda()
loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)


# In[23]:


for epoch in range(EPOCH):
    losses = []
    for i,batch in enumerate(getBatch(BATCH_SIZE, train_data)):
        x,y=list(zip(*batch))
        inputs = torch.cat([prepare_sequence(sent, word2index).view(1, -1) for sent in x])
        targets = torch.cat([prepare_tag(tag, tag2index) for tag in y])
        model.zero_grad()
        preds = model(inputs, is_training=True)
        loss = loss_function(preds, targets)
        #losses.append(loss.data.tolist()[0])
        losses.append(loss.item())
        loss.backward()
        optimizer.step()

        if i % 1000 == 0:
            print("[%d/%d] mean_loss : %0.2f" %(epoch, EPOCH, np.mean(losses)))
            losses = []


# ## Test 

# In[24]:


for_f1_score = []


# In[25]:


accuracy = 0
for test in test_data:
    x, y = test[0], test[1]
    input_ = prepare_sequence(x, word2index).view(1, -1)

    i = model(input_).max(1)[1]
    pred = index2tag[i.data.tolist()[0]]
    for_f1_score.append([pred, y])
    if pred == y:
        accuracy += 1

print(accuracy/len(test_data) * 100)


# This high score is because most of labels are 'O' tag. So we need to measure f1 score.

# ### Print Confusion matrix 

# In[26]:


y_pred, y_test = list(zip(*for_f1_score))


# In[27]:


sorted_labels = sorted(
    list(set(y_test) - {'O'}),
    key=lambda name: (name[1:], name[0])
)


# In[28]:


sorted_labels


# In[29]:


y_pred = [[y] for y in y_pred] # this is because sklearn_crfsuite.metrics function flatten inputs
y_test = [[y] for y in y_test]


# In[30]:


print(metrics.flat_classification_report(
    y_test, y_pred, labels = sorted_labels, digits=3
))


# ### TODO

# * use max-margin objective function http://pytorch.org/docs/master/nn.html#multilabelmarginloss

# In[ ]:
