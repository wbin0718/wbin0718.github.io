---
title: "[TIL PyTorch] 파이토치 자동미분, 옵티마이저"
excerpt: ""
categories:
  - TIL
tags:
  - 

toc: true
toc_sticky: true
--- 

# PyTorch Autograd, Optimizer

## torch.nn.Module 구성

- 딥러닝을 구성하는 Layer의 base class.
- Input, Ouput, Forward, Backward 정의.
- 학습의 대상이 되는 Parameter(tensor) 정의.

## nn.Parameter

- Tensor 객체의 상속 객체.
- nn.Module 내에 attritute가 될 때는 required_grad=True 로 지정되어 학습 대상이 되는 Tensor.
- 우리가 직접 지정할 일은 잘 없음.
    - 대부분의 layer에는 weights 값들이 지정되어 있음

## Backward

- Layer에 있는 Parameter들의 미분을 수행.
- Forward의 결과값 (model의 outout= 예측치)과 실제값 간의 차이(loss)에 대해 미분을 수행.
- 해당 값으로 Parameter 업데이트.

## Backward from the scratch

- 실제 backward는 Module 단계에서 직접 지정가능.
- Module에서 backward 와 optimizer 오버라이딩.
- 사용자가 직접 미분 수식을 써야하는 부담.
    - 쓸일은 없으나 순서는 이해할 필요는 있음.




