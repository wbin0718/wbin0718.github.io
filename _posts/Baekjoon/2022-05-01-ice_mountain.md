---
title: "[백준][Python] 2573번 빙산"
excerpt: ""
categories:
  - Boj
tags:
  - 

toc: true
toc_sticky: true
---   

```python
from collections import deque

n,m = map(int,input().split())

graph = []
for _ in range(n):
    graph.append(list(map(int,input().split())))
    
dx = [-1,1,0,0]
dy = [0,0,-1,1]

def bfs(x,y):
    
    global reduce
    queue = deque()
    queue.append((x,y))
    visited[x][y] = 1
    result = 1
    while queue:
        
        x,y = queue.popleft()
        num = 0
        for i in range(len(dx)):
            
            nx = x + dx[i]
            ny = y + dy[i]
            
            if 0 <= nx < n and 0 <= ny < m:
                
                if graph[nx][ny] == 0 :
                    num += 1
                
                if visited[nx][ny] == 0 and graph[nx][ny] != 0:
                    queue.append((nx,ny))
                    visited[nx][ny] = 1
                    result += 1
        if num != 0:
            sealist.append((x,y,num))
    
    for x,y,sea in sealist:
        graph[x][y] = max(0,graph[x][y]-sea)

    return result

result = 0

ice = []
for i in range(n):
    for j in range(m):
        if graph[i][j]:
            ice.append((i,j))

while True :
        
    visited = [[0] * m for _ in range(n)]
    cnt = []
    sealist = []

    for i,j in ice:
        
        if visited[i][j] == 0 and graph[i][j] != 0 :
            cnt.append(bfs(i,j))
        
    if len(cnt) >= 2 :
        break
    else :
        result += 1
    
    if sum([sum(row) for row in graph]) == 0 :
        result = 0
        break
print(result)
```