---
title: "[TIL PyTorch] 파이토치 메모리 문제"
excerpt: ""
categories:
  - TIL
tags:
  - 

toc: true
toc_sticky: true
--- 

# PyTorch Troubleshooting

## OOM (out of memory)

- 왜 발생했는지 알기 어려움.
- 어디서 발생했는지 알기 어려움.
- Error backtracking이 이상한데로 감.
- 메모리의 이전상황의 파악이 어려움.
> 배치사이즈를 줄이고 GPU clean으로 해결.

## GPUUtill 사용하기

- nvidia-smi 처럼 GPU의 상태를 보여주는 모듈.
- Colab은 환경에서 GPU 상태 보여주기 편함.
- iter마다 메모리가 늘어나는지 확인.
```python
!pip install GPUUtil
import GPUUtil
GPUUtil.showUtilization()
```

## torch.cuda.empty_cache()

- 사용되지 않은 GPU상 cache를 정리.
- 가용 메모리를 확보.
- del 과는 구분이 필요.
- reset 대신 쓰기 좋은 함수.

## training loop tensor로 축적 되는 변수는 확인할 것

- tensor로 처리된 변수는 GPU상 메모리 사용.
- 해당 변수 loop 안 연산이 있을 때 GPU computational graph 생성 (메모리 잠식)
- 1-d tensor의 경우 python 기본 객체로 변환하여 처리할 것. (float 및 item 사용해서 기본 객체로 변환)

## del 명령어를 적절히 사용하기

- 필요가 없어진 변수는 적절한 삭제가 필요함.
- python의 메모리 배치 특성상 loop이 끝나도 메모리를 차지함.

## 가능 batch 사이즈 실험해보기
- 학습시 OOM이 발생했다면 batch 사이즈를 1로 해서 실험해보기.

## torch.no_grad() 사용하기

- Inference 시점에서는 torch.no_grad() 구문을 사용.
- backward pass 으로 인해 쌓이는 메모리에서 자유로움.

## 예상치 못한 에러 메세지

- OOM 말고도 유사한 에러들이 발생.
- CUDNN_STATUS_NOT_INIT 이나 device-side-assert 등
- 해당 에러도 cuda와 관련하여 OOM의 일종으로 생각될 수 있으며, 적절한 코드 처리의 필요함.

## 그외...

- colab은 너무 큰 사이즈는 실행하지 말 것. (linear, CNN, LSTM)
- CNN의 대부분의 에러는 크기가 안 맞아서 생기는 경우. (torchsummary 등으로 사이즈를 맞출 것)
- tensor의 float precision을 16bit로 줄일 수도 있음.



