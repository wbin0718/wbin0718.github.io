---
title: "[백준][Python] 10162번 전자레인지"
excerpt: ""
categories:
  - Boj
tags:
  - 

toc: true
toc_sticky: true
---

```python
n = int(input())
time = [300,60,10]
A = 0
B = 0
C = 0
for i in time:
    if i==300:
        A += n//i
        n = n % i
    elif i == 60:
        B +=n//i
        n = n % i
    else :
        C += n//i
        n = n % i
if n != 0:
    print(-1)
else :
    print(A,B,C)
```