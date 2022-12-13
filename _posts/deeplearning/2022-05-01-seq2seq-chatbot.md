---
title:  "Seq2Seq2로 챗봇 구현"
excerpt: "Seq2Seq2로 챗봇을 구현 해 보자!!"

categories:
  - NLP
tags:
  - [DL]

toc: true
toc_sticky: true
---

# 데이터 불러오기

```python
import os
import shutil
import zipfile

import pandas as pd
import tensorflow as tf
import urllib3
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import random
import torch.nn.functional as f
```

```python
http = urllib3.PoolManager()
url ='http://www.manythings.org/anki/fra-eng.zip'
filename = 'fra-eng.zip'
path = os.getcwd()
zipfilename = os.path.join(path, filename)
with http.request('GET', url, preload_content=False) as r, open(zipfilename, 'wb') as out_file:       
    shutil.copyfileobj(r, out_file)

with zipfile.ZipFile(zipfilename, 'r') as zip_ref:
    zip_ref.extractall(path)

lines = pd.read_csv('fra.txt', names=['src', 'tar', 'lic'], sep='\t')
del lines['lic']
print('전체 샘플의 개수 :',len(lines))

lines = lines.loc[:, 'src':'tar']
lines = lines[0:60000]
lines.sample(10)
```

# 단어집합 생성

```python
lines.tar = lines.tar.apply(lambda x : '\t '+ x + ' \n')
lines.sample(10)

src_vocab = set()
for line in lines.src: 
    for char in line: 
        src_vocab.add(char)

tar_vocab = set()
for line in lines.tar:
    for char in line:
        tar_vocab.add(char)
src_vocab_size = len(src_vocab)+1
tar_vocab_size = len(tar_vocab)+1
print('source 문장의 char 집합 :',src_vocab_size)
print('target 문장의 char 집합 :',tar_vocab_size)

src_vocab = sorted(list(src_vocab))
tar_vocab = sorted(list(tar_vocab))
print(src_vocab[45:75])
print(tar_vocab[45:75])

src_to_index = dict([(word, i+1) for i, word in enumerate(src_vocab)])
tar_to_index = dict([(word, i+1) for i, word in enumerate(tar_vocab)])
print(src_to_index)
print(tar_to_index)

encoder_input = []
for line in lines.src:
  encoded_line = []
  for char in line:
    encoded_line.append(src_to_index[char])
  encoder_input.append(encoded_line)
print('source 문장의 정수 인코딩 :',encoder_input[:5])

decoder_input = []
for line in lines.tar:
  encoded_line = []
  for char in line:
    encoded_line.append(tar_to_index[char])
  decoder_input.append(encoded_line)
print('target 문장의 정수 인코딩 :',decoder_input[:5])

decoder_target = []
for line in lines.tar:
  timestep = 0
  encoded_line = []
  for char in line:
    if timestep > 0:
      encoded_line.append(tar_to_index[char])
    timestep = timestep + 1
  decoder_target.append(encoded_line)
print('target 문장 레이블의 정수 인코딩 :',decoder_target[:5])

max_src_len = max([len(line) for line in lines.src])
max_tar_len = max([len(line) for line in lines.tar])
print('source 문장의 최대 길이 :',max_src_len)
print('target 문장의 최대 길이 :',max_tar_len)

encoder_input = pad_sequences(encoder_input, maxlen=max_src_len, padding='post')
decoder_input = pad_sequences(decoder_input, maxlen=max_tar_len, padding='post')
decoder_target = pad_sequences(decoder_target, maxlen=max_tar_len, padding='post')

encoder_input = to_categorical(encoder_input)
decoder_input = to_categorical(decoder_input)
```
# 모델 구현

## Encoder

```python
class Encoder(nn.Module):
  def __init__(self):
    super().__init__()

    self.lstm_layer = nn.LSTM(80,256,batch_first=True)
  
  def forward(self,x):
    out,(hidden,cell) = self.lstm_layer(x)

    return hidden,cell
```

## Decoder

```python
class Decoder(nn.Module):
  def __init__(self):
    super().__init__()

    self.lstm_layer = nn.LSTM(105,256,batch_first=True)
    self.fc_layer = nn.Linear(256,105)
  
  def forward(self,input,hidden,cell):
    input = input.unsqueeze(1)
    out, (hidden,cell) = self.lstm_layer(input,(hidden,cell))
    out = self.fc_layer(out)
    out = out.squeeze(1)
    return out ,hidden, cell
```
## Encoder-Decoder

```python
class Seq2Seq(nn.Module):
  def __init__(self,encoder,decoder):
    super().__init__()

    self.encoder = encoder
    self.decoder = decoder

  def forward(self,x_input,y_input,teaching_force_ratio=1):
    batch_size = y_input.shape[0]
    y_input_len = y_input.shape[1]
    outputs = torch.zeros(y_input_len,batch_size,105).to(device)

    hidden,cell = self.encoder(x_input)

    input = y_input[:,0,:]

    for t in range(1,y_input_len):
      output ,hidden,cell = self.decoder(input,hidden,cell)
      outputs[t] = output

      teacher_force = random.random() < teaching_force_ratio

      top1 = output.argmax(dim=1)

      if teacher_force : 
        input = y_input[:,t,:]

      else :
        input = torch.zeros_like(input).scatter_(1,top1.unsqueeze(1),1)
    return outputs

```
# 챗봇 구현

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

enc = Encoder()
dec = Decoder()
model = Seq2Seq(enc,dec).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(),lr=0.001)

encoder_input = torch.FloatTensor(encoder_input)
decoder_input = torch.FloatTensor(decoder_input)
decoder_target = torch.LongTensor(decoder_target)

train_tensor = TensorDataset(encoder_input[:30000],decoder_input[:30000],decoder_target[:30000])
valid_tensor = TensorDataset(encoder_input[30000:50000],decoder_input[30000:50000],decoder_target[30000:50000])
test_tensor = TensorDataset(encoder_input[50000:],decoder_input[50000:],decoder_target[50000:])

test_x_input = encoder_input[50000:]
test_y_input = decoder_input[50000:]
test_y_target = decoder_target[50000:]

model.train()
for epoch in range(40):
  avg_loss = 0
  for batch in train_loader:
    x_input = batch[0]
    y_input = batch[1]
    y_target = batch[2]

    x_input = x_input.to(device)
    y_input = y_input.to(device)
    y_target = y_target.to(device)

    output = model(x_input,y_input,1)
    output = output[1:].view(-1,105)
    y_target = y_target[:,:-1].permute(1,0).reshape([-1])
    loss = criterion(output,y_target)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    avg_loss += loss / len(train_loader)

  print("epoch : {} 일때 loss {}".format(epoch+1,avg_loss))
```

```python
def test_translate(x_input,y_input):
  model.eval()

  with torch.no_grad():
   out = model(x_input,y_input,0)

  sentences = []
  for index in x_input.argmax(dim=2)[0].cpu().numpy():
    if index == 0:
      continue
    else :
      word = index_to_src[index]
      sentences.append(word)

  sentences_tar = []
  for index in y_input.argmax(dim=2)[0].cpu().numpy() :
    if index == 0:
      continue
    else :
      word = index_to_tar[index]
      sentences_tar.append(word)

  sentences_pred = []
  for index in out[1:].squeeze(1).argmax(dim=1).cpu().numpy():
    if index == 0:
      continue
    else :
      word = index_to_tar[index]
      sentences_pred.append(word)

  input_sentence = "".join(sentences)
  target_sentence = "".join(sentences_tar).strip()
  translate_sentence = "".join(sentences_pred).strip()

  return input_sentence,target_sentence,translate_sentence
```
```python
encoder_input = torch.FloatTensor(encoder_input)
decoder_input = torch.FloatTensor(decoder_input)

for seq_index in [3,50,100,300,1001]:
  x_input = encoder_input[seq_index:seq_index+1].to(device)
  y_input = decoder_input[seq_index:seq_index+1].to(device)

  input_sentence,target_sentence,translate_sentence = test_translate(x_input,y_input)
  print(35*"-")
  print("입력 문장 : {}".format(input_sentence))
  print("정답 문장 : {}".format(target_sentence))
  print("번역 문장 : {}".format(translate_sentence))
```