---
title:  "[딥러닝 스터디] Auto Encoder에 대해서 알아보자"
excerpt: "Auto encoder를 통해 데이터를 압축표현하고 다시 복구해서 시각화 해 보자!!"

categories:
  - DL
tags:
  - 

toc: true
toc_sticky: true
---

```python
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms, datasets
import torch
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import torch.nn.init
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
```

```python
transform = transforms.Compose([transforms.ToTensor()])
trainset = datasets.FashionMNIST('~/.pytorch/F_MNIST_data/', download=True, train=True, transform=transform)
testset = datasets.FashionMNIST('~/.pytorch/F_MNIST_data/', download=True, train=False, transform=transform)
```

![image](https://user-images.githubusercontent.com/104637982/166158903-10748a04-7ef8-4e19-834f-8bf03c82d4ee.png)

```python
train_loader = DataLoader(trainset,batch_size=128,shuffle=True,drop_last=True)
test_loader = DataLoader(testset,batch_size=128,shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(156)
if device =="cuda":
  torch.cuda_manula_seed(156)
```   

# 모델 구현   

```python
class Autoencoder(nn.Module):
  def __init__(self):
    super().__init__()

    self.encoder = nn.Sequential(
        nn.Linear(28*28,128),
        nn.ReLU(),
        nn.Linear(128,64),
        nn.ReLU(),
        nn.Linear(64,12),
        nn.ReLU(),
        nn.Linear(12,3),
    )

    self.decoder = nn.Sequential(
        nn.Linear(3,12),
        nn.ReLU(),
        nn.Linear(12,64),
        nn.ReLU(),
        nn.Linear(64,128),
        nn.ReLU(),
        nn.Linear(128,28*28),
        nn.Sigmoid(),
    )
  def forward(self,x):
    encoded = self.encoder(x)
    decoded = self.decoder(encoded)

    return encoded,decoded

model = Autoencoder().to(device)
optimizer = torch.optim.Adam(model.parameters(),lr=0.01)
criterion = nn.MSELoss()

view_data = x[:5].view(-1,28*28)

for epoch in range(10):
  model.train()
  avg_loss = 0
  for i,(x,target) in enumerate(train_loader):

    x = x.view(-1,28*28).to(device)
    y = x.view(-1,28*28).to(device)
    target = target.to(device)

    encoded,decoded = model(x)
    loss = criterion(decoded,y)
    optimizer.zero_grad()

    loss.backward()
    optimizer.step()
    avg_loss += loss / len(train_loader)
  print("epoch : {}일때 loss : {}".format(epoch+1,avg_loss))
```

# 시각화   

```python
fig,ax = plt.subplots(2,5,figsize=(10,5))

for i in range(5):
  img = np.reshape(view_data.numpy()[i],(28,28))
  ax[0][i].imshow(img,cmap="gray")
  ax[0][i].set_xticks(())
  ax[0][i].set_yticks(())

for i in range(5):
  img = np.reshape(decoded_data.detach().numpy()[i],(28,28))
  ax[1][i].imshow(img,cmap="gray")
  ax[1][i].set_xticks(())
  ax[1][i].set_yticks(())
```   

![image](https://user-images.githubusercontent.com/104637982/166158959-b108ea9c-da25-4672-9956-9170cfdfd908.png)   

# Denoise Auto Encoder

![image](https://user-images.githubusercontent.com/104637982/166158983-fb1d42eb-0429-4c38-8cee-7784bed787ec.png)   

```python
def add_noise(img):
  noise = torch.randn(img.size()) * 0.3
  noisy_img = img + noise
  return noisy_img

for epoch in range(10):
  model.train()
  avg_loss = 0
  for i,(x,target) in enumerate(train_loader):
    noisy_x = add_noise(x)
    noisy_x = noisy_x.view(-1,28*28).to(device)
    y = x.view(-1,28*28).to(device)
    target = target.to(device)

    encoded,decoded = model(noisy_x)
    loss = criterion(decoded,y)
    optimizer.zero_grad()

    loss.backward()
    optimizer.step()
    avg_loss += loss / len(train_loader)
  print("epoch : {}일때 loss : {}".format(epoch+1,avg_loss))

sample_data = x[:5].view(-1,28*28)
test_x = sample_data
_,decoded_data = model(test_x)

original_img = sample_data
noisy_img = add_noise(original_img)
_,recovered_img = model(noisy_img)
```

# 시각화

```python
fig,ax = plt.subplots(3,5,figsize=(10,5))

for i in range(5):
  img = np.reshape(original_img.numpy()[i],(28,28))
  ax[0][i].imshow(img,cmap="gray")
  ax[0][i].set_xticks(())
  ax[0][i].set_yticks(())

for i in range(5):
  img = np.reshape(noisy_img.numpy()[i],(28,28))
  ax[1][i].imshow(img,cmap="gray")
  ax[1][i].set_xticks(())
  ax[1][i].set_yticks(())

for i in range(5):
  img = np.reshape(recovered_img.detach().numpy()[i],(28,28))
  ax[2][i].imshow(img,cmap="gray")
  ax[2][i].set_xticks(())
  ax[2][i].set_yticks(())
```   

![image](https://user-images.githubusercontent.com/104637982/166159024-5dbd3dc6-7060-4aee-a1e3-8cc7611344e3.png)   

```python
sample_data = x[0].view(-1,28*28)
original_x = sample_data[0]
noisy_x = add_noise(original_x)
_,recovered_x = model(noisy_x)

fig,ax = plt.subplots(1,3,figsize=(15,15))

original_img = original_x.view(28,28)
noisy_img = noisy_x.view(28,28)
recovered_img = recovered_x.view(28,28).detach().numpy()
ax[0].set_title("Original Img")
ax[0].imshow(original_img,cmap="gray")
ax[1].set_title("Noisy Img")
ax[1].imshow(noisy_img,cmap="gray")
ax[2].set_title("Recovered Img")
ax[2].imshow(recovered_img,cmap="gray")
```   
![image](https://user-images.githubusercontent.com/104637982/166159049-e38f35d0-2c3f-4726-bbfb-c5f64dc5d99f.png)
