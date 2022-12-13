---
title:  "[딥러닝 스터디] custom dataset을 사용한 모델 학습"
excerpt: "이미지 파일 구성과 데이터 로더를 직접 구현해 모델을 학습시키자!!"

categories:
  - DL
tags:
  - 

toc: true
toc_sticky: true
---

```python
import torch
from torchvision import transforms
import cv2
from PIL import Image
from torch.utils.data import DataLoader , Dataset
import torch.nn as nn
import os
import glob
from torchvision import models

import torch
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import matplotlib.pyplot as plt
import random
import numpy as np
```

```python
mnist_train = dsets.MNIST(root='MNIST_data/',
                          train=True,
                          download=True)

mnist_test = dsets.MNIST(root='MNIST_data/',
                         train=False,
                         download=True)
```

```python
path = os.getcwd()
for i in ["zero","one","two","three","four","five","six","seven","eight","nine"]:
  os.makedirs(path+"/파이토치/{}".format(i))

for i in ["zero","one","two","three","four","five","six","seven","eight","nine"]:
  for j in ["train","test"]:
    os.makedirs(path+"/파이토치/{}/{}".format(i,j))

def make_file(train):
  
  if train == True:

    for col,idx in zip(mnist_train.data,mnist_train.targets):
      col = np.array(col)
      mnist_label =  ["zero","one","two","three","four","five","six","seven","eight","nine"]
      index = mnist_label[idx]
      os.chdir(path+"/파이토치/{}/train".format(index))
      length = len(os.listdir(os.getcwd()))
      cv2.imwrite("{}_{}.jpg".format(index,length+1),col)
  
  elif train == False :

    for col,idx in zip(mnist_test.data,mnist_test.targets):
      col = np.array(col)
      mnist_label =  ["zero","one","two","three","four","five","six","seven","eight","nine"]
      index = mnist_label[idx]
      os.chdir(path+"/파이토치/{}/test".format(index))
      length = len(os.listdir(os.getcwd()))
      cv2.imwrite("{}_{}.jpg".format(index,length+1),col)

make_file(train=True)
make_file(train=False)
label_list = ["zero","one","two","three","four","five","six","seven","eight","nine"

def file_list(train):
  trainset= []
  for i in ["zero","one","two","three","four","five","six","seven","eight","nine"]:
    file_list = glob.glob(path+"/파이토치/{}/{}".format(i,train)+"/*.jpg")
    trainset += file_list

  return trainset

train_file_list = file_list("train")
test_file_list = file_list("test")

class Custom_Dataset(Dataset):

  def __init__(self,file_list,transforms,label_list):
    self.file_list = file_list
    self.label_list = label_list
    self.transforms = transforms


  def __len__(self):
    return len(self.file_list)

  def __getitem__(self,index):
    img_path = self.file_list[index]
    img = cv2.imread(img_path,cv2.IMREAD_GRAYSCALE)
    img = Image.fromarray(img)
    img_transformed = self.transforms(img)
    label_idx = img_path.split("/")[3]
    label = self.label_list.index(label_idx)

    return img_transformed, label
```

```python
train_tensor = Custom_Dataset(train_file_list,transforms,label_list)
test_tensor = Custom_Dataset(test_file_list,transforms,label_list)

train_loader =  DataLoader(train_tensor,shuffle=True,batch_size=128,drop_last=True)
test_loader = DataLoader(test_tensor,shuffle=False,batch_size=128)

class CNN(nn.Module):
  def __init__(self):
    super().__init__()

    self.layer1 = nn.Sequential(
        nn.Conv2d(1,32,kernel_size=3,stride=1,padding=1,bias=False),
        nn.BatchNorm2d(32),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2,stride=2,padding=1)
    )

    self.layer2 = nn.Sequential(
        nn.Conv2d(32,64,kernel_size=3,stride=1,padding=1,bias=False),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2,stride=2)
    )

    self.layer3 = nn.Sequential(
        nn.Conv2d(64,128,kernel_size=3,stride=1,padding=1,bias=False),
        nn.BatchNorm2d(128),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2,stride=2,padding=1)
    )

    self.fc1 = nn.Linear(128,256)
    nn.init.xavier_uniform_(self.fc1.weight)
    self.avg_pool = nn.AdaptiveAvgPool2d(output_size=(1,1))

    self.layer4 = nn.Sequential(
        self.fc1,
        nn.ReLU(),
        nn.Dropout(0.2)
    )

    self.fc2 = nn.Linear(256,10)
    nn.init.xavier_uniform_(self.fc2.weight)

  def forward(self,x):
    out = self.layer1(x)
    out = self.layer2(out)
    out = self.layer3(out)
    out = self.avg_pool(out)
    out = out.squeeze()
    out = self.layer4(out)
    out = self.fc2(out)
    return out
```   

![image](https://user-images.githubusercontent.com/104637982/166158658-bd0b09f8-2b1d-44d0-81f3-261f047dee7e.png)   

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

model = CNN().to(device)
optimizer = torch.optim.Adam(model.parameters(),lr=0.01)
criterion = nn.CrossEntropyLoss()

batch_length = len(train_loader)
loss = []
for epoch in range(30):
  
  avg_loss = 0
  
  for data,targets in train_loader:
    
    data = data.to(device)
    targets = targets.to(device)
    hypothesis = model(data)
    cost = criterion(hypothesis,targets)
    
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    avg_loss += cost / batch_length
  loss.append(avg_loss)
  print("Epoch : {} loss 값 : {}".format(epoch+1,avg_loss))

def train():
  correct = 0
  model.eval()
  with torch.no_grad():
    for data,targets in train_loader:
      data = data.to(device)
      targets = targets.to(device)
      outputs = model(data)
      _,predicted = torch.max(outputs,1)
      correct += predicted.eq(targets.data.view_as(predicted)).sum()
  length = len(train_loader.dataset)
  print("train데이터에서 정확도 : {}/{} {:.2f}%".format(correct,length,correct / length * 100))

def test():
  correct = 0
  model.eval()
  with torch.no_grad():
    for data,targets in test_loader:
      data = data.to(device)
      targets = targets.to(device)
      outputs = model(data)
      _,predicted = torch.max(outputs,1)
      correct += predicted.eq(targets.data.view_as(predicted)).sum()
  length = len(test_loader.dataset)
  print("test데이터에서 정확도 : {}/{} {:.2f}%".format(correct,length,correct / length * 100))

train()
test()

```

# 인셉션 모듈 추가   

![image](https://user-images.githubusercontent.com/104637982/166158697-bd5c8e3b-5756-486a-8f9f-5d451d868f91.png)   


```python
def conv_1(input_dim,output_dim):
  model = nn.Sequential(
      nn.Conv2d(input_dim,output_dim,1,1),
      nn.ReLU(),
  )
  return model

def conv_1_3(input_dim,mid_dim,output_dim):
  model = nn.Sequential(
      nn.Conv2d(input_dim,mid_dim,1,1),
      nn.ReLU(),
      nn.Conv2d(mid_dim,output_dim,kernel_size=3,stride=1,padding=1),
      nn.ReLU(),
  )
  return model

def conv_1_5(input_dim,mid_dim,output_dim):
  model = nn.Sequential(
          nn.Conv2d(input_dim,mid_dim,1,1),
          nn.ReLU(),
          nn.Conv2d(mid_dim,output_dim,5,1,2),
          nn.ReLU()
      )
  return model

def max_3_1(input_dim,output_dim):
  model = nn.Sequential(
      nn.MaxPool2d(kernel_size=3,stride=1,padding=1),
      nn.Conv2d(input_dim,output_dim,1,1),
      nn.ReLU(),
  )
  return model

class inception_module(nn.Module):
  def __init__(self,input_dim,output_dim_1,mid_dim_3,output_dim_3,mid_dim_5,output_dim_5,pool_dim):
    super().__init__()

    self.conv_1 = conv_1(input_dim,output_dim_1)

    self.conv_1_3 = conv_1_3(input_dim,mid_dim_3,output_dim_3)

    self.conv_1_5 = conv_1_5(input_dim,mid_dim_5,output_dim_5)

    self.max_3_1 = max_3_1(input_dim,pool_dim)

  def forward(self,x):
    out_1 = self.conv_1(x)
    out_2 = self.conv_1_3(x)
    out_3 = self.conv_1_5(x)
    out_4 = self.max_3_1(x)

    output = torch.cat([out_1,out_2,out_3,out_4],1)
    return output
  
class Cnn_Inception(nn.Module):
  def __init__(self):
    super().__init__()

    self.layer1 = nn.Sequential(
        nn.Conv2d(1,64,kernel_size=7,stride=2,padding=3,bias=False),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
    )

    self.layer2 = nn.Sequential(
        inception_module(64,128,96,128,16,32,32),
        inception_module(320,128,128,192,32,96,64),
        nn.MaxPool2d(3,2,1),
        nn.AdaptiveAvgPool2d(output_size=(1,1))
    )

    self.layer3 = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(480,10),
    )

  def forward(self,x):
    out = self.layer1(x)
    out = self.layer2(out)
    out = out.squeeze()
    out = self.layer3(out)
    return out

model_inception = Cnn_Inception().to(device)
optimizer = torch.optim.Adam(model_inception.parameters(),lr=0.01)
criterion = nn.CrossEntropyLoss()

batch_length = len(train_loader)
loss_inception = []
for epoch in range(10):
  
  avg_loss = 0
  
  for data,targets in train_loader:
    
    data = data.to(device)
    targets = targets.to(device)
    hypothesis = model_inception(data)
    cost = criterion(hypothesis,targets)
    
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    avg_loss += cost / batch_length
  loss_inception.append(avg_loss)
  print("Epoch : {} loss 값 : {}".format(epoch+1,avg_loss))

def train():
  correct = 0
  model.eval()
  with torch.no_grad():
    for data,targets in train_loader:
      data = data.to(device)
      targets = targets.to(device)
      outputs = model_inception(data)
      _,predicted = torch.max(outputs,1)
      correct += predicted.eq(targets.data.view_as(predicted)).sum()
  length = len(train_loader.dataset)
  print("train데이터에서 정확도 : {}/{} {:.2f}%".format(correct,length,correct / length * 100))

def test():
  correct = 0
  model.eval()
  with torch.no_grad():
    for data,targets in test_loader:
      data = data.to(device)
      targets = targets.to(device)
      outputs = model_inception(data)
      _,predicted = torch.max(outputs,1)
      correct += predicted.eq(targets.data.view_as(predicted)).sum()
  length = len(test_loader.dataset)
  print("test데이터에서 정확도 : {}/{} {:.2f}%".format(correct,length,correct / length * 100))
```

# 전이학습 및 파인튜닝   

```python
model_resnet18 = models.resnet18(pretrained=True)
optimizer = torch.optim.Adam(model_resnet18.parameters(),lr=0.01)
criterion = nn.CrossEntropyLoss()

model_resnet18.fc = nn.Linear(in_features=512,out_features=10,bias=True)
model_resnet18.conv1 = nn.Conv2d(1,64,kernel_size=(7,7),stride=(2,2),padding=(3,3),bias=False)
model_resnet18 = model_resnet18.to(device)

batch_length = len(train_loader)
loss_resnet18 = []
for epoch in range(10):
  
  avg_loss = 0
  
  for data,targets in train_loader:
    
    data = data.to(device)
    targets = targets.to(device)
    hypothesis = model_resnet18(data)
    cost = criterion(hypothesis,targets)
    
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    avg_loss += cost / batch_length
  loss_resnet18.append(avg_loss)
  print("Epoch : {} loss 값 : {}".format(epoch+1,avg_loss))

def train():
  correct = 0
  model.eval()
  with torch.no_grad():
    for data,targets in train_loader:
      data = data.to(device)
      targets = targets.to(device)
      outputs = model_resnet18(data)
      _,predicted = torch.max(outputs,1)
      correct += predicted.eq(targets.data.view_as(predicted)).sum()
  length = len(train_loader.dataset)
  print("train데이터에서 정확도 : {}/{} {:.2f}%".format(correct,length,correct / length * 100))

def test():
  correct = 0
  model.eval()
  with torch.no_grad():
    for data,targets in test_loader:
      data = data.to(device)
      targets = targets.to(device)
      outputs = model_resnet18(data)
      _,predicted = torch.max(outputs,1)
      correct += predicted.eq(targets.data.view_as(predicted)).sum()
  length = len(test_loader.dataset)
  print("test데이터에서 정확도 : {}/{} {:.2f}%".format(correct,length,correct / length * 100))

train()
test()
```