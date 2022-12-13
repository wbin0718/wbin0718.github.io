---
title: "[프로그래머스][Python] 포켓몬"
excerpt: ""
categories:
  - Algorithm
tags:
  - 

toc: true
toc_sticky: true
---   

```python
def solution(nums):
    num=[]
    for i in nums:
        if i not in num:
            num.append(i)
    maxi= len(nums)//2
    if len(num)< maxi:
        return len(num)
    elif len(num)==maxi:
        return maxi
    else:
        return maxi
```