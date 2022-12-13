---
title: "[TIL PyTorch] 파이토치 basic 문법"
excerpt: ""
categories:
  - TIL
tags:
  - 

toc: true
toc_sticky: true
--- 

# 파이토치

> 파이토치는 numpy와 AutoGrad를 합친 것.

## Array to Tensor   


* Tensor 생성은 list나 ndarray 사용 가능.


```python
1. torch.tensor()
2. torch.from_numpy()
```

## Numpy like operations


* pytorch는 numpy 사용법이 그대로 적용 됨.
* pytorch의 tensor는 GPU에 올려서 사용가능.

## Tensor handling

* view : reshape과 동일하게 tensor의 shape를 반환
* squeeze : 차원의 개수가 1인 차원 삭제
* unsqueeze : 차원의 개수가 1인 차원 추가

## Tensor operations

* numpy의 연산이 가능. (덧셈, 뺄셈)
* 행렬곱셈 연산은 dot이 아닌 mm사용.(matmul도 존재)

    * mm은 행렬 계산만 가능한 반면 matmul은 broadcasting이 일어나 행렬과 상수간 계산 가능.

## AutoGrad

* requires_grad = True










