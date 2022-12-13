---
title: "[TIL] Generative Model"
excerpt: ""
categories:
  - TIL
tags:
  - 

toc: true
toc_sticky: true
---

# Generative Model

## Basic Discrete Distributions

- 베르누이 분포

## Independence

- 독립성으로 인해 파라미터를 줄임.

## Conditional Independence

- chain rule
- Bayes's rule
- Conditional independence

- chain rule을 사용하면 $2^n$-1개의 파라미터 사용.
- chain rule을 Markov assumption을 사용해 간단히 표현하면 2n-1의 파라미터를 가짐.

## Autoregressive Models

- 연속적으로 작동하는 모델.
- 28 x 28 binary pixels.
- chain rule로 확률분포 분해를 함.
- sampling이 쉬움.
- 확률 계산이 쉬움.