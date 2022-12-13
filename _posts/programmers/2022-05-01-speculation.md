---
title: "[프로그래머스][Python] 콜라츠 추측"
excerpt: ""
categories:
  - Algorithm
tags:
  - 

toc: true
toc_sticky: true
---   

```python
def solution(num):
    x=0
    while num!=1 :
        x+=1
        if num%2==0:
            num = num//2
        else :
            num = num*3+1
        
        if x>=500:
            return -1
        
    return x
```