---
title: "[TIL 딥러닝] Neural Networks"
excerpt: ""
categories:
  - TIL
tags:
  - 

toc: true
toc_sticky: true
---

# Neural Networks

> 인간을 모방하다.

## Linear Neural Networks

- 1차원 데이터.
- weight를 곱하고 bias를 더함.
- MSE의 loss를 구함.
- W,b를 loss로 편미분.
- 학습률은 너무 커도 안되고, 너무 작아도 안 됨.
- 여러층을 쌓을 때는 비선형 함수가 필요함.(더 많은 표현력을 얻음)
- ReLU, Sigmoid, Tangent 등 비선형 함수가 있음.

## Multi-Layer Perceptron

- 여러층을 쌓음.
- regression task는 MSE, classification task는 CE, probabillistic task는 MLE 사용.