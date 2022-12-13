---
title: "[프로그래머스][Python] x만큼 간격이 있는 n개의 숫자"
excerpt: ""
categories:
  - Algorithm
tags:
  - 

toc: true
toc_sticky: true
---   

```python
def solution(x,n):
    return set(range(x,x*n+x,x))
```