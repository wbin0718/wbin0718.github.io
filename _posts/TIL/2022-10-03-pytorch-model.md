---
title: "[TIL PyTorch] 파이토치 학습 모델 저장 및 로드"
excerpt: ""
categories:
  - TIL
tags:
  - 

toc: true
toc_sticky: true
--- 

# 모델 불러오기

> 학습 결과를 공유하고 싶을 때 사용.
>> 모델을 저장할 필요가 있다.

## model.save()

- 학습의 결과를 저장하기 위한 함수.
- 모델 형태(architecture)와 파라메터를 저장.
- 모델 학습 중간 과정의 저장을 통해 최선의 결과모델을 선택.
- 만들어진 모델을 외부 연구자와 공유하여 학습 재연성 향상.
- state_dict 모델의 파라메터를 표시.

```python
모델의 파라미터를 저장
torch.save(model.state_dict(),path)

모델의 형태에서 파라메터만 load
new_model = TheModelClass()
new_model.load_state_dict(torch.load(path),"model.pt")

모델의 architecture 저장
torch.save(model,path)

모델의 architecture와 함께 load
model = torch.load(path)
```

## checkpoints

- 학습의 중간 결과를 저장하여 최선의 결과를 선택.
- earlystopping 기법 사용시 이전 학습의 결과물을 저장.
- loss와 metric 값을 지속적으로 확인 저장.
- 일반적으로 epoch, loss, metric을 함께 저장하여 확인.
- colab에서 지속적인 학습을 위해 필요.

```python
torch.save({
    "epoch":e,
    "model_state_dict":model.state_dict(),
    "optimizer_state_dict":optimizer.state_dict(),
    "loss":epoch_loss,
},path)

checkpoint = torch.load(path)
model.load_state_dict(checkpoint["model_state_dict"])
optimizer.state_dict(checkpoint["optimizer_state_dict"])
epoch = checkpoint["epoch"]
loss = checkpoint["loss"]
```

## Transfer learning

- 다른 데이터셋으로 만든 모델을 현재 데이터에 적용.
- 일반적으로 대용량 데이터셋으로 만들어진 모델의 성능 향상.
- 현재의 DL에서는 가장 일반적인 학습 기법.
- backbone architecture가 잘 학습된 모델에서 일부분만 변경하여 학습을 수행함.

## Freezing

- pretrained model을 활용시 모델의 일부분을 frozen 시킴.

```python

사전 학습된 모델 불러오기

class MyNewNet(nn.Module):
    def __init__(self):
        super(MyNewNet, self).__init__()
        self.vgg19 = models.vgg19(pretrained=True)
        
        모델에 마지막 Linear Layer 추가

        self.linear_layer = nn.Linear(1000,1)
    
    def forward(self,x):
        x = self.vgg19(x)
        return self.linear_layer(x)

마지막 레이어를 제외하고 frozen

for param in my_model.parameters():
    param_requires_grad = False

for param in my_model.linear_layer.parameters():
    param.requires_grad = True
```





