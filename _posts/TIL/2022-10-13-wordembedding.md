---
title: "[TIL] Word Embedding"
excerpt: ""
categories:
  - TIL
tags:
  - 

toc: true
toc_sticky: true
---

# Word Embedding

- 단어를 벡터로 표현.
- 비슷한 의미를 가지는 단어는 좌표상 비슷한 위치로 표현 됨.
- "cat"과 "kitty"는 비슷한 위치 "hamburger"는 다른 위치로 표현 됨.

## Word2Vec

- 인접한 단어들끼리는 의미가 비슷할 것임.

- 중심 단어를 주고 주변 단어의 확률 분포를 예측.

- Sentence : "I study math."

- Vocabulary : {"I","study","math"}

- window의 크기가 3이고 중심 단어가 I 라면 (I, study) 입력과 출력 쌍을 만듦.

- 중심 단어가 study라면 (study, I), (study, math) 입력 및 출력 쌍을 만듦.

- (study, math)가 입력 및 출력이 되었을 때 첫번 째 가중치 크기는 2X3 두번째 가중치 크기는 3X2임. 이때 입력 값이 study의 원핫벡터 이므로 study의 1과 곱해지는 column이 study의 단어 표현이 됨.

- 이때 행렬 곱 연산이 이루어지지 않고 단지 원핫벡터의 1인 부분의 값만 뽑으므로 임베딩 layer라고 함.

- 가중치의 크기는 은닉층의 크기를 M이라고 하면 V X M임.

- V는 단어 사전의 크기.

- 가중치 둘중 어떤 것을 word embedding으로 사용할 것이냐?

- 주로 첫번 째 가중치를 word embedding으로 사용함.

- 단어들 간 같은 관계를 가지면 같은 벡터를 가짐.

### Word Intrusion Detection

- 각 단어들 중 가장 상이한 단어를 고름.

- 단어들 간 유클리디안 거리를 다 구해서 평균을 구함.

## Glove

- 입력과 출력의 동시 등장 행렬을 구함.
- 학습이 Word2Vec보다 빠르고 데이터가 적을 때도 잘 동작함.




