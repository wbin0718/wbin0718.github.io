---
title: "[TIL PyTorch] 파이토치 데이터셋 및 데이터로더 생성"
excerpt: ""
categories:
  - TIL
tags:
  - 

toc: true
toc_sticky: true
--- 

# PyTorch Dataset

## Dataset 클래스

- 데이터 입력 형태를 정의하는 클래스.
- 데이터를 입력하는 방식의 표준화.
- Image, Text, Audio 등에 따른 다른 입력 정의.

```python
class CustomDataset(Dataset):
    def __init__(self,text,labels):
        self.data = text
        self.labels = labels
    def __len__(self):
        return len(self.labels)
    def __getitem(self,idx):
        label = self.labels[idx]
        text = self.data[idx]
        smaple = {"Text":text, "Class":label}
        return sample
```
- __init__은 데이터 생성 및 초기화.
- __len__은 데이터 길이 반환.
- __getitem__은 인덱스 따라 데이터 반환.
- transform 있을 시 getitem으로 적용.
- 초기화 할때 모든 데이터 적용 x (많은 시간 소요)

## Dataset 클래스 생성시 유의점

- 데이터 형태에 따라 각 함수를 다르게 정의함.
- 모든 것을 데이터 생성 시점에 처리할 필요는 없음.
    - image의 Tensor 변화는 학습에 필요한 시점에 변환.
- 데이터 셋에 대한 표준화된 처리방법 제공 필요.
    - 후속 연구자 또는 동료에게는 빛과 같은 존재.
- 최근에는 HuggingFace등 표준화된 라이브러리 사용.

## DataLoader 클래스

- Data의 Batch를 생성해주는 클래스.
- 학습직전(GPU feed전) 데이터의 변환을 책임.
- Tensor로 변환 + Batch 처리가 메인 업무.
- 병렬적인 데이터 전처리 코드의 고민 필요.
- collate_fn은 튜플 형태의 데이터를 train과 label끼리 묶어주는 역할.
