---
title:  "[논문 리뷰] 트랜스포머 코드 구현 및 분류 적용"
excerpt: "트랜스포머 논문을 직접 구현해보자!!"

categories:
  - NLP
tags:
  - [DL]

toc: true
toc_sticky: true
---

# 트랜스포머를 활용한 분류 문제

```python
vocab_size = 20000  
max_len = 200

(X_train, y_train), (X_test, y_test) = tf.keras.datasets.imdb.load_data(num_words=vocab_size)
print('훈련용 리뷰 개수 : {}'.format(len(X_train)))
print('테스트용 리뷰 개수 : {}'.format(len(X_test)))

X_train = tf.keras.preprocessing.sequence.pad_sequences(X_train, maxlen=max_len)
X_test = tf.keras.preprocessing.sequence.pad_sequences(X_test, maxlen=max_len)

X_train = torch.LongTensor(X_train)
X_test = torch.LongTensor(X_test)
y_train = torch.LongTensor(y_train)
y_test = torch.LongTensor(y_test)

train_tensor = TensorDataset(X_train,y_train)
train_loader = DataLoader(train_tensor,batch_size=32,shuffle=True,drop_last=True)

test_tensor = TensorDataset(X_test,y_test)
test_loader = DataLoader(test_tensor,batch_size=32,shuffle=False)
```

```python
hid_dim = 32
n_heads = 2
pf_dim = 32
input_dim = vocab_size
n_layers=1

encoder = Encoder(input_dim,hid_dim,n_layers,n_heads,pf_dim,dropout)
model = Seq2Seq(encoder,decoder,src_pad_idx,trg_pad_idx).to(device)

class Encoder_Decoder(nn.Module):
  def __init__(self,model):
    super().__init__()
    self.model = model
    self.dropout = nn.Dropout(0.1)
    self.fc_layer = nn.Linear(32,20)
    self.relu_layer = nn.ReLU()
    self.linear_layer = nn.Linear(20,2)
  
  def forward(self,X_train):
    src_mask = self.model.make_src_mask(X_train)
    x = self.model.encoder(X_train,src_mask)
    x = torch.mean(x,dim=1)
    x = self.dropout(x)
    x = self.fc_layer(x)
    x = self.relu_layer(x)
    x = self.dropout(x)
    x = self.linear_layer(x)

    return x


encoder_decoder = Encoder_Decoder(model).to(device)
optimizer = optim.Adam(encoder_decoder.parameters(),lr=0.001)
criterion = nn.CrossEntropyLoss()

for epoch in range(2):
  avg_loss = 0
  for X,target in train_loader:

    output = encoder_decoder(X)
    loss = criterion(output,target)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    avg_loss += loss / len(train_loader)

  print("epoch : {} 일때 loss : {}".format(epoch+1,avg_loss))

  def test():
  model.eval()
  corrected = 0
  with torch.no_grad():

    for X,target in test_loader:
      output = encoder_decoder(X)
      corrected += (output.argmax(dim=1) == target).sum()
  
  print("테스트 정확도 : {}".format(corrected / len(test_loader.dataset) * 100))

  test()
```
분류 문제를 풀 때는 인코더만을 사용하여 문제를 수행한다.