---
title:  "[논문 리뷰] Distill BERT"
excerpt: "경량화를 목표로한 Distill BERT 논문 요약입니다."

categories:
  - NLP
tags:
  - [DL]

toc: true
toc_sticky: true
---

# Distill BERT

* BERT보다 모델의 크기는 40% 줄이고 성능은 97% 가까이 되며 속도는 60% 정도 빠르다.

# Introduction

* Large Scale의 모델들은 좋은 성능을 내지만 많은 파라미터를 가지고 있다. 그리고 현재 연구들은 모델의 크기가 커질수록 downstream task의 성능이 좋아진다고 주장한다.

* 더 큰 크기의 모델로 향한 추세는 환경 비용과 증가하는 계산량과 메모리는 다양한 분야로 적용을 어렵게 한다.

* 따라서 더 작은 모델로 여러 downstream task의 성능이 비슷한 수준으로 도달하고 추론을 할 때 빠르고 가벼운 모델을 보이려고 한다.

* 적은 계산 비용을 요구로 한다.

# Knowledge Distillation

* knowledge distillation은 더 큰 모델의 행동을 따라하거나 앙상블 모델이다.

* student model은 teacher model의 soft 확률을 target으로 학습이 된다.

* softmax-temperature를 사용하며 student, teacher 모두 같은 T가 적용된다. 추론할 때는 T는 1로 적용된다.

* training loss는 supervised learning loss와 mlm loss와 cosine embedding loss를 결합하였다.

* cosine embedding은 학생과 선생의 방향을 정렬해준다.

# DistillBERT : a distilled version of BERT

* 학생인 Distill BERT는 BERT의 구조와 같다.

* 층의 수를 2의 인수로 줄이는 동안 token-type embeddings와 pooler 층을 제거했다.

* transformer의 구조는 선형대수학과 매우 잘 최적화 되었으며 층의 수와 같은 인자 변경보다 마지막 차원 변경이 더욱 적은 계산 효율적으로 영향을 미친다.

* 따라서 층의 수를 줄이는 것으로 초점을 맞췄다.

* student model의 차원은 teacher model과 같도록 했으며 student model의 가중치를 teacher model의 가중치로 초기화를 했다.

* Distill BERT는 동적 마스킹을 사용하고 NSP taks를 제거했다.