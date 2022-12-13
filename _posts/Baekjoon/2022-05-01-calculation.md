---
title: "[백준][Python] 2644번 촌수계산"
excerpt: ""
categories:
  - Boj
tags:
  - 

toc: true
toc_sticky: true
---   

DFS 풀이   

```python
import sys
sys.setrecursionlimit(1000000)

n = int(input())
num1,num2 = map(int,input().split())
m = int(input())
graph = [[] for _ in range(n+1)]
for i in range(1,m+1):
    node1,node2 = map(int,input().split())
    graph[node1].append(node2)
    graph[node2].append(node1)

visited = [0] * (n+1)
result = [0] * (n+1)

def dfs(v):
    visited[v] = 1
    
    for i in graph[v]:
        if visited[i] == 0 :
            result[i] = result[v] + 1
            dfs(i)

dfs(num1)            
if result[num2] > 0:
    print(result[num2])
else :
    print(-1)
```   

BFS 풀이   

```python
from collections import deque

n = int(input())
num1,num2 = map(int,input().split())
m = int(input())
graph = [[] for _ in range(n+1)]
for i in range(1,m+1):
    node1,node2 = map(int,input().split())
    graph[node1].append(node2)
    graph[node2].append(node1)

visited = [0] * (n+1)
result = [0] * (n+1)

def bfs(v):
    
    queue = deque()
    queue.append(v)
    visited[v] = 1
    
    while queue:
        
        v = queue.popleft()
        
        for i in graph[v]:
            
            if visited[i] == 0:
                queue.append(i)
                visited[i] = 1
                result[i] = result[v] + 1
                
bfs(num1)

if result[num2] > 0:
    print(result[num2])
else :
    print(-1)
```