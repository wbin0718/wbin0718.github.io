---
title: "[TIL PyTorch] 파이토치 RNN"
excerpt: ""
categories:
  - TIL
tags:
  - 

toc: true
toc_sticky: true
---
# RNN

## Sequential Model

- 입력이 들어왔을 때 다음 입력을 예측.

- p($x_t$|x<sub>t-1</sub>x<sub>t-2</sub>...)

## Recurrent Neural Network

- 현재 정보와 이전 정보를 사용해 예측.

- Short-term dependencies : 입력이 길어지면 이전 정보가 흐릿해짐.
- sigmoid 활성화 함수를 사용하면 기울기 소실이 일어남.
- relu 활성화 함수를 사용하면 기울기 폭발이 일어남.

## Long Short Term Memory

- rnn의 문제를 해결하기 위해 사용.

- cell state 존재.

- Forget gate, Input gate, Output gate가 있음.

- cell state : 어떤 정보가 중요하고 안 중요한지 전달.

- Forget gate : cell state 중 어떤 정보를 버릴지 결정.

- Input gate : 새로 만들어진 정보 중 어떤 정보만 cell state로 추가할지 결정.

- Update cell : cell state 중 버릴 정보는 버리고 추가할 정보는 추가.

- Output Gate : 어떤 값을 받을 지 결정해 output 출력.

## GRU

- reset gate, update gate가 있음.
- cell state가 없음.
