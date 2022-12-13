---
title: "[TIL PyTorch] 파이토치 학습 모니터링 도구"
excerpt: ""
categories:
  - TIL
tags:
  - 

toc: true
toc_sticky: true
--- 

# Monitoring tools for PyTorch

## Tensorboard

- TensorFlow의 프로젝트로 만들어진 시각화 도구.
- 학습 그래프, metric, 학습 결과의 시각화 지원.
- Pytorch도 연결 가능 -> DL 시각화 핵심 도구.

- scaler : metric 등 상수 값의 연속(epoch)을 표시.
- graph : 모델의 computational graph 표시.
- histogram : weight 등 값의 분포를 표현.
- image : 예측 값과 실제 값을 비교 표시.
- mesh : 3d 형태의 데이터를 표현하는 도구.

```python
import os
logs_base_dir = "logs"
os.makedirs(log_base_dir, exist_ok=True)

from torch.utils.tensorboard import SummaryWriter
import numpy as np

writer = SummaryWriter(logs_base_dir)
for n_iter in range(100):
    writer.add_scaler("Loss/train", np.random.random(),n_iter)
    writer.add_scaler("Loss/test", np.random.random(),n_iter)
    writer.add_scaler("Accuracy/train", np.random.random(),n_iter)
    writer.add_scaler("Accuracy/test", np.random.random(),n_iter)
writer.flush()

%load_ext tensorboard
%tensorboard--logdir{logs_base_dir}
```

## weight & biases

- 머신러닝 실험을 원활히 지원하기 위한 상용도구.
- 협업, code versioning, 실험 결과 기록 등 제공.
- MLOps의 대표적인 툴로 저변 확대 중.
