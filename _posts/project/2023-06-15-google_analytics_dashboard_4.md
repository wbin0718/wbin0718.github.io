---
title: "[이벤트] 구글 애널리틱스 보고서"
excerpt: 구글 데모 데이터인 merchandise store의 데이터를 빅쿼리를 가지고 데이터를 추출하고 태블로를 활용 해 구글 애널리틱스 보고서를 만드는 프로젝트
categories:
  - project
tags:
  - 
toc: true
toc_sticky: true
---

구글 애널리틱스4의 참여도 지표중 마지막인 이벤트 보고서를 빅쿼리로 데이터를 추출하려고 합니다! 사실 이벤트는 전환이랑 차이점이 없습니다. 단지 전환은 전환 이벤트로 설정된 데이터만 본 것이라면 이벤트는 전체 이벤트를 다 살펴보는 것이라고 할 수 있습니다. 간단하니 바로 살펴보겠습니다.

## 이벤트수

![image](https://github.com/wbin0718/google_analytics_dashboard/assets/104637982/66d94f17-f86b-45b1-b692-0380bb9f94a0)

이벤트수도 전환과 마찬가지로 page_view가 가장 많은 것을 볼 수가 있습니다.   
두 번째로 user_engagement 이벤트가 가장 많은 것을 볼 수가 있는데요, 기본적으로 구글 애널리틱스4는 사용자가 접속을 했을 때 page_view 이벤트가 발생하고 10초이상을 머무르게 되면 참여라는 user_engagement 이벤트가 발생하게 됩니다. 이로 인해서 page_view, user_engagement 이벤트가 가장 많이 발생한 이벤트인 것 같습니다.   

## 총사용자

![image](https://github.com/wbin0718/google_analytics_dashboard/assets/104637982/78b01278-836e-4c05-8087-566d624fc428)

총 사용자 역시 page_view가 많습니다.

## 사용자당 이벤트수

![image](https://github.com/wbin0718/google_analytics_dashboard/assets/104637982/85e8ffa3-f7d3-406a-b740-834ca53eca2e)

view_item이라는 이벤트는 평균적으로 사용자 한명이 6번정도 이벤트를 발생시키는 것으로 보입니다.

## 총수익

![image](https://github.com/wbin0718/google_analytics_dashboard/assets/104637982/b1041afd-4a9a-4107-aded-fcae022fbea0)

전환과 마찬가지로 구매 이벤트만 약 2억 정도의 수익이 있는 것으로 확인됩니다.

이렇게 참여도로 속하는 페이지 화면 및 클래스, 방문 페이지, 전환, 이벤트 4가지를 빅쿼리를 통해 데이터를 추출하고 태블로로 시각화까지 해봤는데요, 모니터링 하는 지표들을 정확하게 이해하고 있어야만 적절하게 데이터를 활용하여 향상된 서비스 제공으로 이어질 수 있다고 생각합니다. 참여도 보고서를 직접 만들면서 구글 애널리틱스 4가 제공하는 지표들을 정확하게 이해하는 데 많음 도움이 되었습니다.




