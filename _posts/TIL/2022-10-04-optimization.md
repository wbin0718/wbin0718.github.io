---
title: "[TIL PyTorch] 파이토치 Optimization"
excerpt: ""
categories:
  - TIL
tags:
  - 

toc: true
toc_sticky: true
---

# Optimization

## Generalization

- train error와 test error 사이 차이.

## Underfitting vs Overfitting

- train 데이터만 잘 맞추는 것.

## Cross-validation

- train과 test를 나누어 학습.
- 어떻게 나누면 좋을까?
- k개로 나누어 k-1개로 학습 나머지 한개로 validation 진행.
- 최적의 하이퍼파라미터를 찾고 모든 데이터를 사용해 다시 학습.

## Bias and Variance

- Variance는 입력을 넣었을때 출력이 일관적이면 낮음.
- bias는 평균적으로 타겟과 가까우면 낮음.
- Bias and Variance Tradeoff 둘다 줄이는 것은 얻기 어려움.

## Bootstrapping

- 학습 데이터를 몇개만 사용해 여러 모델을 만듦.

## Bagging vs Boosting   

### Bagging
    - 학습 데이터 일부를 사용해 여러 모델을 만들어 출력 값을 평균을 냄.

### Boosting

    - 100개 중 80개는 예측을 잘하고 20개는 예측을 못 하면 두번째 모델을 만들어 20개 데이터를 잘 예측하는 모델을 만듦.

# Gradient Descent Methods

- Stochastic gradient descent : 하나의 샘플을 통해 기울기를 구함.
- Mini-batch gradient descent : 배치 데이터 샘플을 통해 기울기를  구함.
- Batch gradient descent : 전체 데이터 샘플을 통해 기울기를 구함.

## Batch-size Matters

- 배치 사이즈를 줄이면 Flat Minimum 도달하고, 늘리면 Sharp Minimum을 도달함.
- Flat Minimum을 도달해야 일반화 성능이 좋음.

## Gradient Descent

1. Gradient Descent : learning rate를 설정하기 어려움.
2. Momentum : 흘러간 gradient를 잘 유지함.
3. Nesterov Accelerated Gradient : 현재 정보 방향으로 가고 그 방향의 기울기를 계산해서 acumulation 함.
4. Adagrad : 파라미터들의 변화를 봄. 많이 변화한 파라미터는 적게 적게 변화한 파라미터는 많이 변화시킴.
5. Adagelta : learning rate가 없음.
6. RMSprop : gradient squares를 그냥 구하지 않음.
7. Adam : past gradients와 squared gradients를 같이 사용함.

## Regularization
1. Early Stopping : loss가 커지는 시점 멈춤.
2. Parameter Norm Penalty : 파라미터의 크기가 커지지 않도록 함.
3. Data Augmentation : 데이터를 증가시킴.
4. Noise Robustness : 입력값으로 noise를 추가함.
5. Label Smoothing : 학습 데이터 두개를 섞음.
6. Dropout : 뉴런의 가중치 일부를 0으로 바꿈.
7. Batch Normalization : 적용하려는 layer를 정규화를 시킴.





