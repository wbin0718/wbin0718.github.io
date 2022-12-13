---
title: "[백준][Python] 1967번 트리의 지름"
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

def dfs(v,weight):
    visited[v] = 1

    for i in graph[v]:
        
        x,y = i
        
        if visited[x] == 0 :
            visited[x] = 1
            if length[x] == -1:
                    
                length[x] = weight + y
                dfs(x,weight+y)

n = int(input())
graph = [[] for _ in range(n+1)]
for _ in range(n-1):
    node1,node2,w = map(int,input().split())
    graph[node1].append([node2,w])
    graph[node2].append([node1,w])
    
visited = [0 for _ in range(n+1)]
length = [-1] * (n+1)
length[1] = 0
dfs(1,0)
visited = [0 for _ in range(n+1)]
node = length.index(max(length))
length = [-1] * (n+1)
length[node] = 0
dfs(node,0)
print(max(length))
```   

BFS 풀이

```python
from collections import deque

def bfs(v,weight):
    queue = deque()
    queue.append((v,weight))
    visited[v] = 1
    length[v] = 0
    while queue :
        
        v,weight = queue.popleft()
        
        for i in graph[v]:
            x,y = i
            
            if visited[x] == 0 :
                if length[x] == -1 :
                    visited[x] = 1
                    length[x] = weight + y
                    queue.append((x,length[x]))

n = int(input())
graph = [[] for _ in range(n+1)]
for _ in range(n-1):
    node1,node2,w = map(int,input().split())
    graph[node1].append([node2,w])
    graph[node2].append([node1,w])

visited = [0 for _ in range(n+1)]
length = [-1 for _ in range(n+1)]
bfs(1,0)                    
node = length.index(max(length))

visited = [0 for _ in range(n+1)]
length = [-1 for _ in range(n+1)]
bfs(node,0)    
print(max(length))
```