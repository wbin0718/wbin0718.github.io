---
title:  "[논문 리뷰] SKT Brain의 KoBert를 사용한 네이버 영화 댓글 리뷰 감성 분석"
excerpt: "KoBert를 사용하여 네이버 영화 댓글 리뷰에 대한 감성 분석을 해 보자!!"

categories:
  - NLP
tags:
  - [DL]

toc: true
toc_sticky: true
---

# KOBERT 및 라이브러리 불러오기

```python
pip install 'git+https://github.com/SKTBrain/KoBERT.git#egg=kobert_tokenizer&subdirectory=kobert_hf'
!pip install git+https://git@github.com/SKTBrain/KoBERT.git@master
!pip install sentencepiece==0.1.91
import transformers
transformers.__version__
import pandas as pd
import numpy as np
import urllib.request
import os
from tqdm import tqdm
import tensorflow as tf
from transformers import BertTokenizer, TFBertModel

from torch.utils.data import TensorDataset,DataLoader

from kobert_tokenizer import KoBERTTokenizer
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import BertModel
from transformers import BertForSequenceClassification
import gluonnlp as nlp
```

# 데이터 불러오기

```python
urllib.request.urlretrieve("https://raw.githubusercontent.com/e9t/nsmc/master/ratings_train.txt", filename="ratings_train.txt")
urllib.request.urlretrieve("https://raw.githubusercontent.com/e9t/nsmc/master/ratings_test.txt", filename="ratings_test.txt")

train_data = pd.read_table('ratings_train.txt')
test_data = pd.read_table('ratings_test.txt')

train_data = train_data.dropna(how = 'any')
train_data = train_data.reset_index(drop=True)
print(train_data.isnull().values.any())

test_data = test_data.dropna(how = 'any')
test_data = test_data.reset_index(drop=True)
print(test_data.isnull().values.any())
```

# 데이터 전처리

```python
tokenizer = KoBERTTokenizer.from_pretrained("skt/kobert-base-v1")

max_seq_len = 128
def Preprocessing(examples,labels,max_seq_len,tokenizer):

  input_ids = []
  attention_masks = []
  token_type_ids = []
  data_labels = []

  for example, label in tqdm(zip(examples,labels),total=len(examples)):

    encode = tokenizer(example,max_length=max_seq_len,pad_to_max_length=True)
    input_id = encode["input_ids"]
    attention_mask = encode["attention_mask"]
    token_type_id = encode["token_type_ids"]

    assert len(input_id) == max_seq_len
    assert len(attention_mask) == max_seq_len
    assert len(token_type_id) == max_seq_len

    input_ids.append(input_id)
    attention_masks.append(attention_mask)
    token_type_ids.append(token_type_id)
    data_labels.append(label)

  input_ids = torch.tensor(input_ids)
  attention_masks = torch.tensor(attention_masks)
  token_type_ids = torch.tensor(token_type_ids)
  data_labels = torch.tensor(data_labels)

  return (input_ids,attention_masks,token_type_ids),data_labels

train_X,train_y = Preprocessing(train_data["document"],train_data["label"],max_seq_len,tokenizer)
test_X, test_y = Preprocessing(test_data["document"],test_data["label"],max_seq_len,tokenizer)

```

# 모델 생성 및 학습

```python
device = xm.xla_device()
bert = BertModel.from_pretrained("skt/kobert-base-v1").to(device)
input_ids_train = train_X[0]
attention_masks_train = train_X[1]
token_type_ids_train = train_X[2]

input_ids_test = test_X[0]
attention_masks_test = test_X[1]
token_type_ids_test = test_X[2]

train_tensor = TensorDataset(input_ids_train,attention_masks_train,token_type_ids_train,train_y)
test_tensor = TensorDataset(input_ids_test,attention_masks_test,token_type_ids_test,test_y)

train_loader = DataLoader(train_tensor,batch_size=128,shuffle=True,drop_last=True)
test_loader = DataLoader(test_tensor,batch_size=128,shuffle=False)

class BERTClassifier(nn.Module):
  def __init__(self,bert,hidden_size = 768,num_classes=1,dropout=None):
    super().__init__()
    self.bert = bert
    self.classifier = nn.Linear(hidden_size,num_classes)
    self.sigmoid = nn.Sigmoid()

    if dropout :
      self.dropout = nn.Dropout(dropout)
    
  def forward(self,input_ids,attention_masks,token_type_ids):
    _,pooler = self.bert(input_ids,attention_masks,token_type_ids,return_dict=False)

    if self.dropout :
      out = self.dropout(pooler)
      out = self.classifier(out)
      out = self.sigmoid(out)
    else :
      out = self.classifier(pooler)
      out = self.sigmoid(out)

    return out

hidden_size = 768
num_classes = 1
model = BERTClassifier(bert,hidden_size,num_classes,dropout = 0.1).to(device)
optimizer = optim.Adam(model.parameters(),lr=5e-5)
criterion = nn.BCELoss()

for epoch in range(2):
  avg_loss = 0

  model.train()

  for batch in train_loader:

    input_id = batch[0].to(device)
    attention_mask = batch[1].to(device)
    token_type_id = batch[2].to(device)
    target = batch[3].to(device)
    target = target.type(torch.FloatTensor)

    logits = model(input_id,attention_mask,token_type_id)

    loss = criterion(logits.cpu().view(-1),target.cpu())

    avg_loss += loss / len(train_loader)
    
    optimizer.zero_grad()
    loss.backward()
    xm.optimizer_step(optimizer,barrier=True)

  print("epoch : {} 일때 loss : {}".format(epoch+1,avg_loss))

```

# 평가

```python
def test():
  model.eval()
  corrected = 0

  for batch in test_loader:
    input_id = batch[0].to(device)
    attention_mask = batch[1].to(device)
    token_type_id = batch[2].to(device)
    target = batch[3].to(device)

    logits = model(input_id,attention_mask,token_type_id)
    corrected += ((logits.cpu().view(-1) > 0.5) == target.cpu()).sum()

  print("테스트 정확도 : {:.2f}%".format(corrected / len(test_loader.dataset) * 100))

test()

def sentiment_predict(new_sentence):
  encode = tokenizer(new_sentence,max_length=max_seq_len,pad_to_max_length=True)
  input_id = encode["input_ids"]
  attention_mask = encode["attention_mask"]
  token_type_id = encode["token_type_ids"]

  input_id = torch.tensor(input_id).unsqueeze(0)
  attention_mask = torch.tensor(attention_mask).unsqueeze(0)
  token_type_id = torch.tensor(token_type_id).unsqueeze(0)

  input_id = input_id.to(device)
  attention_mask = attention_mask.to(device)
  token_type_id = token_type_id.to(device)

  logits = model(input_id,attention_mask,token_type_id)
  score = logits.cpu().item()

  if score > 0.5:
    print("{:.2f}% 확률로 긍정 리뷰입니다.\n".format(score * 100))
  else :
    print("{:.2f}% 확률로 부정 리뷰입니다.\n".format((1-score) * 100))

sentiment_predict("이 영화 존잼입니다 대박")
```
