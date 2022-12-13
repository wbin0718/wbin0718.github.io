---
title: "[TIL] Transformer"
excerpt: ""
categories:
  - TIL
tags:
  - 

toc: true
toc_sticky: true
---

# Sequential Model

- 순서가 있는 데이터는 다루기가 어려움.
- 생략되거나 순서가 바뀌거나 잘린 데이터를 다뤄야 함.

## Transformer

- sequential 한 데이터를 다루고 임베딩을 함.
- NLP만 사용되지는 않고 vision도 사용됨.
- 분류, 번역등 task 사용.
- 여러층의 encoder-decoder로 구성되어 있음. 파라미터는 모두 다름.
- 재귀적으로 학습하지 않고 모두 같이 처리.

### Self-Attention
- 각 단어들이 임베딩된 차원으로 입력값이 됨.
- n개의 단어들의 정보를 모두 활용함.
- Q,K,V 벡터들을 만듦.
- score 벡터들을 만듦. q벡터와 k벡터를 내적. 유사도를 구하는 것임. 행렬로 계산이 되면 정방행렬이 만들어짐.
- score 벡터를 정규화를 해줌. 특정 range내 속하도록 함.
- Q,K는 차원이 같아야하고 V는 달라도 됨.
- 주변 단어가 달라지면 출력이 달라져서 다양한 표현을 학습함.
- 메모리를 많이 소모하며 출력 길이 제한이 있음.

### Multi-Head Attention

- 인코더의 입력과 출력 크기를 맞추기 위해 차원을 맞춤.
- Self-Attention의 출력 값을 8개를 만들어 concatenate함.

### Positional Encoding

- 입력의 순서를 알지 못함.
- 입력 순서는 중요하므로 순서 정보를 sin과 cos을 활용해 추가함.

### Decoder

- Self-Attention을 사용.
- generative한 방법으로 출력 값 생성.
- encoder와 decoder가 상호작용을 함.
- encoder의 K,V decoder의 Q벡터를 사용.

## Vision Transformer

- 이미지로 적용한 transformer 모델임.