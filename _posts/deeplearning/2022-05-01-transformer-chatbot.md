---
title:  "[논문 리뷰] 트랜스포머 코드 구현 및 챗봇 적용"
excerpt: "트랜스포머 논문을 직접 구현해보자!!"

categories:
  - NLP
tags:
  - [DL]

toc: true
toc_sticky: true
---

# 데이터 불러오기    

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import urllib.request
import time
import tensorflow_datasets as tfds
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset,DataLoader

urllib.request.urlretrieve("https://raw.githubusercontent.com/songys/Chatbot_data/master/ChatbotData.csv", filename="ChatBotData.csv")
train_data = pd.read_csv('ChatBotData.csv')
train_data.head()
```
# 데이터 전처리    

```python
questions = []
for sentence in train_data['Q']:
    sentence = re.sub(r"([?.!,])", r" \1 ", sentence)
    sentence = sentence.strip()
    questions.append(sentence)

answers = []
for sentence in train_data['A']:
    sentence = re.sub(r"([?.!,])", r" \1 ", sentence)
    sentence = sentence.strip()
    answers.append(sentence)

tokenizer = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
    questions + answers, target_vocab_size=2**13)

START_TOKEN, END_TOKEN = [tokenizer.vocab_size], [tokenizer.vocab_size + 1]
VOCAB_SIZE = tokenizer.vocab_size + 2

MAX_LENGTH = 40

def tokenize_and_filter(inputs, outputs):
  tokenized_inputs, tokenized_outputs = [], []

  for (sentence1, sentence2) in zip(inputs, outputs):
    
    sentence1 = START_TOKEN + tokenizer.encode(sentence1) + END_TOKEN
    sentence2 = START_TOKEN + tokenizer.encode(sentence2) + END_TOKEN

    tokenized_inputs.append(sentence1)
    tokenized_outputs.append(sentence2)

  tokenized_inputs = tf.keras.preprocessing.sequence.pad_sequences(
      tokenized_inputs, maxlen=MAX_LENGTH, padding='post')
  tokenized_outputs = tf.keras.preprocessing.sequence.pad_sequences(
      tokenized_outputs, maxlen=MAX_LENGTH, padding='post')

  return tokenized_inputs, tokenized_outputs

questions, answers = tokenize_and_filter(questions, answers)
```
# 트랜스포머 모델 구현

## Encoder

```python
class Encoder(nn.Module):
  def __init__(self,input_dim,hid_dim,n_layers,n_heads,pf_dim,dropout):
    super().__init__()

    self.hid_dim = hid_dim
    self.embedding_layer = nn.Embedding(input_dim,hid_dim)
    self.scale = torch.sqrt(torch.FloatTensor([hid_dim])).to(device)
    self.dropout = nn.Dropout(dropout)
    self.layers = nn.ModuleList([EncoderLayer(hid_dim,n_heads,pf_dim,dropout) for _ in range(n_layers)])

  def forward(self,src,src_mask):
    batch_size = src.shape[0]
    src_len = src.shape[1]

    encoding = torch.zeros(batch_size,src_len,self.hid_dim,device=device)
    encoding.requires_grad = False
    position = torch.arange(0,src_len,device=device).unsqueeze(1)
    _2i = torch.arange(0,self.hid_dim,step=2,device=device)

    encoding[:,:,0::2] = torch.sin(position / (10000 ** (_2i / self.hid_dim))).unsqueeze(0).repeat(batch_size,1,1)
    encoding[:,:,1::2] = torch.cos(position / (10000 ** (_2i / self.hid_dim))).unsqueeze(0).repeat(batch_size,1,1)

    src = self.dropout((self.embedding_layer(src) * self.scale) + encoding)
    for layers in self.layers:
      src = layers(src,src_mask)
      return src
```

## Encoders

```python
class EncoderLayer(nn.Module):
  def __init__(self,hid_dim,n_heads,pf_dim,dropout):
    super().__init__()

    self.self_attention = MultiHeadAttentionLayer(hid_dim,n_heads,dropout)
    self.self_attention_layer_norm = nn.LayerNorm(hid_dim)
    self.positionwise_feedforward_layer = PositionwiseFeedforwardLayer(hid_dim,pf_dim,dropout)
    self.positionwise_feedforward_layer_layer_norm = nn.LayerNorm(hid_dim)
    self.dropout = nn.Dropout(dropout)

  def forward(self,src,src_mask):

    _src,_ = self.self_attention(src,src,src,src_mask)
    src = self.self_attention_layer_norm(src + self.dropout(_src))
    _src = self.positionwise_feedforward_layer(src)
    src = self.positionwise_feedforward_layer_layer_norm(src + self.dropout(_src))

    return src
```

# Multihead Attention

```python
class MultiHeadAttentionLayer(nn.Module):
  def __init__(self,hid_dim,n_heads,dropout):
    super().__init__()

    assert hid_dim % n_heads ==0
    
    self.hid_dim = hid_dim
    self.n_heads = n_heads
    self.head_dim = hid_dim // n_heads

    self.fc_q = nn.Linear(hid_dim,hid_dim)
    self.fc_k = nn.Linear(hid_dim,hid_dim)
    self.fc_v = nn.Linear(hid_dim,hid_dim)

    self.fc_o = nn.Linear(hid_dim,hid_dim)

    self.scale = torch.sqrt(torch.FloatTensor([self.head_dim])).to(device)
    self.dropout = nn.Dropout(dropout)

  def forward(self,query,key,value,mask=None):

    batch_size = query.shape[0]

    Q = self.fc_q(query)
    K = self.fc_k(key)
    V = self.fc_v(value)

    Q = Q.view(batch_size,-1,self.n_heads,self.head_dim).permute(0,2,1,3)
    K = K.view(batch_size,-1,self.n_heads,self.head_dim).permute(0,2,1,3)
    V = V.view(batch_size,-1,self.n_heads,self.head_dim).permute(0,2,1,3)

    energy = torch.matmul(Q,K.permute(0,1,3,2)) / self.scale

    if mask is not None:
      energy = energy.masked_fill(mask == 0,-1e10)

    attention = torch.softmax(energy,dim=-1)

    x = torch.matmul(self.dropout(attention),V)
    x = x.permute(0,2,1,3).contiguous()
    x = x.view(batch_size,-1,self.hid_dim)

    x = self.fc_o(x)

    return x , attention
```

# Positionwise FeedForward Layer

```python
class PositionwiseFeedforwardLayer(nn.Module):
  def __init__(self,hid_dim,pf_dim,dropout):
    super().__init__()

    self.fc_1 = nn.Linear(hid_dim,pf_dim)
    self.fc_2 = nn.Linear(pf_dim,hid_dim)

    self.dropout = nn.Dropout(dropout)

  def forward(self,x):

    x = torch.relu(self.fc_1(x))
    x = self.dropout(x)
    x = self.fc_2(x)

    return x
```

# Decoder

```python
class Decoder(nn.Module):
  def __init__(self,output_dim,hid_dim,n_layers,n_heads,pf_dim,dropout):
    super().__init__()

    self.hid_dim = hid_dim
    self.embedding_layer = nn.Embedding(output_dim,hid_dim)
    self.dropout = nn.Dropout(dropout)
    self.layers = nn.ModuleList([DecoderLayer(hid_dim,n_heads,pf_dim,dropout) for _ in range(n_layers)])
    self.scale = torch.sqrt(torch.FloatTensor([hid_dim])).to(device)

    self.fc_layer = nn.Linear(hid_dim,output_dim)

  def forward(self,trg,enc_src,trg_mask,src_mask):
    batch_size = trg.shape[0]
    trg_len = trg.shape[1]

    encoding = torch.zeros(batch_size,trg_len,self.hid_dim,device=device)
    encoding.requires_grad=False
    position = torch.arange(0,trg_len,device=device).unsqueeze(1)
    _2i = torch.arange(0,self.hid_dim,step=2,device=device)

    encoding[:,:,0::2] = torch.sin(position / (10000 ** (_2i / self.hid_dim))).unsqueeze(0).repeat(batch_size,1,1)
    encoding[:,:,1::2] = torch.cos(position / (10000 ** (_2i / self.hid_dim))).unsqueeze(0).repeat(batch_size,1,1)

    trg = self.dropout(self.embedding_layer(trg)* self.scale + encoding)
    for layers in self.layers:
      trg,attention = layers(trg,enc_src,trg_mask,src_mask)

    output = self.fc_layer(trg)
    
    return output, attention
```

# Decoders

```python
class DecoderLayer(nn.Module):
  def __init__(self,hid_dim,n_heads,pf_dim,dropout):
    super().__init__()

    self.self_attention = MultiHeadAttentionLayer(hid_dim,n_heads,dropout)
    self.self_attention_norm_layer = nn.LayerNorm(hid_dim)
    self.encoder_decoder_self_attention = MultiHeadAttentionLayer(hid_dim,n_heads,dropout)
    self.encoder_decoder_self_attention_norm_layer = nn.LayerNorm(hid_dim)
    self.positionwise_feedforward_layer = PositionwiseFeedforwardLayer(hid_dim,pf_dim,dropout)
    self.positionwise_feedforward_layer_norm_layer = nn.LayerNorm(hid_dim)

    self.dropout = nn.Dropout(dropout)

  def forward(self,trg,enc_src,trg_mask,src_mask):
    _trg,_ = self.self_attention(trg,trg,trg,trg_mask)
    trg = self.self_attention_norm_layer(trg + self.dropout(_trg))
    _trg,attention= self.encoder_decoder_self_attention(trg,enc_src,enc_src,src_mask)
    trg= self.encoder_decoder_self_attention_norm_layer(trg + self.dropout(_trg))
    _trg = self.positionwise_feedforward_layer(trg)
    trg = self.positionwise_feedforward_layer_norm_layer(trg + self.dropout(_trg))

    return trg,attention
```
# Seq2Se2

```python
class Seq2Seq(nn.Module):
  def __init__(self,encoder,decoder,src_pad_idx,trg_pad_idx):
    super().__init__()

    self.encoder = encoder
    self.decoder = decoder
    self.src_pad_idx = src_pad_idx
    self.trg_pad_idx = trg_pad_idx

  def make_src_mask(self,src):
    
    src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
    
    return src_mask
  
  def make_trg_mask(self,trg):
    trg_pad_mask = (trg != self.trg_pad_idx).unsqueeze(1).unsqueeze(2)

    trg_len = trg.shape[1]

    trg_sub_mask = torch.tril(torch.ones((trg_len,trg_len),device=device)).bool()

    trg_mask = trg_pad_mask & trg_sub_mask
    return trg_mask

  def forward(self,src,trg):

    src_mask = self.make_src_mask(src)
    trg_mask = self.make_trg_mask(trg)

    enc_src = self.encoder(src,src_mask)
    output,attention = self.decoder(trg,enc_src,trg_mask,src_mask)

    return output,attention
```

# 학습 스케쥴러 구현

```python
class Scheduler_lr():
  def __init__(self,optimizer,hid_dim,warmup_steps=4000):
    super().__init__()

    self.optimizer = optimizer
    self.hid_dim = hid_dim
    self.warmup_steps = warmup_steps
    self.steps = 0

  def getlr(self):
    arg1 = self.steps ** (-0.5)
    arg2 = self.steps * (self.warmup_steps ** (-1.5))

    return self.hid_dim ** (-0.5) * min(arg1,arg2)

  def update_lr(self):
    self.steps += 1
    lr = self.getlr()

    for param in optimizer.param_groups:
      param["lr"] = lr
  
  def step(self):
    self.update_lr()
```

# 챗봇 구현

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_dim = VOCAB_SIZE
output_dim = VOCAB_SIZE
hid_dim = 256
n_layers = 2
n_heads = 8
pf_dim = 512
dropout = 0.1
src_pad_idx = 0
trg_pad_idx = 0

encoder = Encoder(input_dim,hid_dim,n_layers,n_heads,pf_dim,dropout)
decoder = Decoder(output_dim,hid_dim,n_layers,n_heads,pf_dim,dropout)
model = Seq2Seq(encoder,decoder,src_pad_idx,trg_pad_idx).to(device)

train_tensor = TensorDataset(torch.LongTensor(questions),torch.LongTensor(answers))
train_loader = DataLoader(train_tensor,batch_size=128,shuffle=True,drop_last=True)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(),lr=0.0001)
lr_scheduler = Scheduler_lr(optimizer,hid_dim,4000)

for epoch in range(50):
  avg_loss = 0
  for batch in train_loader:
    src = batch[0].to(device)
    trg = batch[1].to(device)

    output,_ = model(src,trg)
    loss = criterion(output[:,:-1,:].contiguous().view(-1,VOCAB_SIZE),trg[:,1:].contiguous().view(-1))
    optimizer.zero_grad()
    loss.backward()
    lr_scheduler.step()
    optimizer.step()
    avg_loss += loss / len(train_loader)
  print("epoch : {} 일때 loss : {}".format(epoch+1, avg_loss))

def preprocess_sentence(sentence):

  sentence = re.sub(r"([?.!,])", r" \1 ",sentence)
  sentence = sentence.strip()
  return sentence

def chatbot(sentence):
  question = sentence
  sentence = preprocess_sentence(sentence)
  sentence = tokenizer.encode(sentence)
  sentence = START_TOKEN + sentence + END_TOKEN
  sentence = torch.LongTensor(sentence).unsqueeze(0).to(device)
  trg_tokens =  START_TOKEN.copy()
  model.eval()
  for i in range(MAX_LENGTH):
    with torch.no_grad():
    
      trg = torch.LongTensor(trg_tokens).unsqueeze(0).to(device)
      out,_ = model(sentence,trg)
      token = out.argmax(dim=-1)[:,-1].item()
      trg_tokens.append(token)

      if [token] == END_TOKEN:
        break

  print("Input: {}".format(question))
  print("Output: {}".format(tokenizer.decode(trg_tokens[1:-1])))
```
