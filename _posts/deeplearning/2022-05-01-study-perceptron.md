---
title:  "[딥러닝 스터디] Perceptron의 등장"
excerpt: "Perceptron으로 시작된 딥러닝"

categories:
  - DL
tags:
  - [DL]

toc: true
toc_sticky: true
---

# 퍼셉트론의 등장

![image](https://user-images.githubusercontent.com/104637982/166157183-f74ff051-0a5c-4821-998a-c0fd710e6b5e.png)   

가지돌기를 통해 신호를 받아들이고 이 신호가 일정 임계치 이상을 넘으면 축삭돌기를 통해 신호가 전달된다.   
   
![image](https://user-images.githubusercontent.com/104637982/166157198-5ec13a16-8ec7-4809-a11b-8cb98db14274.png)   

퍼셉트론 역시 입력이라는 신호를 받고 어떤 가중치와 곱해져 임계치를 넘으면 1을 출력 (이때 가중치는 입력값의 중요도에 따라 값이 커지거나 작아짐.)   

![image](https://user-images.githubusercontent.com/104637982/166157235-fbaf0bf0-0f58-4c53-9efe-0ba77f107fd8.png)
   
0과 1로 출력하도록 사용되는 계단함수.
이를 활성화 함수라고 함.

# 단층 퍼셉트론

![image](https://user-images.githubusercontent.com/104637982/166157249-5178125c-f05f-4651-8a85-e908a2922de6.png)
   

입력층과 출력층만 있는 것을 단층 퍼셉트론이라고 함.   
단층 퍼셉트론은 직선식으로만 게이트 문제를 해결할 수 있음.   

![image](https://user-images.githubusercontent.com/104637982/166157354-cd88b754-b676-48ba-ba0b-eeccbd12d906.png)   
   
AND, NAND, OR 게이트 문제를 해결가능.   
XOR 게이트 문제는 해결하지 못함.

# 다층 퍼셉트론(MLP)

![image](https://user-images.githubusercontent.com/104637982/166157344-d3a9e83c-b19e-47bb-8499-c5f76a29b9b2.png)

실제로 XOR문제를 해결하기 위해서는 AND, OR, NAND게이트의 조합으로 해결이 가능.

![image](https://user-images.githubusercontent.com/104637982/166157384-513071da-559f-452b-ac35-95765b1ddb59.png)
   
입력층, 출력층 이외에 은닉층이 추가되면 이를 다층 퍼셉트론이라고 함.   

단층 퍼셉트론의 단점인 평면에서 해결하려는 것을 은닉층을 추가하면서 평면을 휘어주는 방법을 생각함.

![image](https://user-images.githubusercontent.com/104637982/166157414-23fcd471-1108-4c98-8b89-a7377672d10b.png)

하나의 퍼셉트론으로 XOR문제를 해결하지 못 하기 떄문에 추가의 퍼셉트론이 필요함.
   
-> 그렇다면 XOR문제를 해결 가능

# 심층신경망

![image](https://user-images.githubusercontent.com/104637982/166157460-b7e528ae-59d8-451e-9b1f-b6b53e7f11d5.png)

위에서 은닉층 1개이상이 추가되면 다층 퍼셉트론이라고 함.
이때 2개 이상의 은닉층이 있다면 이를 심층신경망이라고 하고 이를 자동적으로 옵티마이저와 손실함수를 통해 최적화 하는 것을 딥러닝이라고 함.












