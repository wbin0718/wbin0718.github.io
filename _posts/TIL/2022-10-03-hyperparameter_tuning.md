---
title: "[TIL PyTorch] 하이퍼 파라미터 튜닝"
excerpt: ""
categories:
  - TIL
tags:
  - 

toc: true
toc_sticky: true
--- 

# 하이터파라미터 튜닝

- 모델 스스로 학습하지 않는 값은 사람이 지정.(learning rate, 모델의 크기, optimizer 등)
- 하이퍼 파라미터를 따라 값이 크게 좌우 될 때도 있음. (요즘은 그렇지 않음)
- 마지막 0.01을 쥐어짜야 할 때 도전해볼만 함.

## grid vs random

- 가장 기본적인 방법 - grid vs random
- 최근에는 베이지안 기반 기법들이 주도

## Ray

- multi-node multi processing 지원 모듈
- ML/DL의 병렬 처리를 위해 개발된 모듈
- 기본적으로 현재의 분산병렬 ML/DL 모듈의 표준
- Hyperparameter Search를 위한 다양한 모듈 제공

> 하이퍼 파라미터 튜닝보다는 데이터가 성능 영향을 미침. 데이터 전처리를 더욱 잘하고 쥐어짤 때 하이퍼파라미터 튜닝을 하자.
