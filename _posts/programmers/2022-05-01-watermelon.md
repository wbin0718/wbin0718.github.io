---
title: "[프로그래머스][Python] 수박수박수박수박수박수?"
excerpt: ""
categories:
  - Algorithm
tags:
  - 

toc: true
toc_sticky: true
---   

```python
def solution(n):
    sum=[]
    for i in range(n):
        if i%2==0:
            sum.append("수")
        if i%2==1:
            sum.append("박")
    return "".join(sum)
```