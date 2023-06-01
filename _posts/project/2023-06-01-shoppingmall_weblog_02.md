---
title: "[데이터 분석] 쇼핑몰 웹 로그 분석 프로젝트-2"
excerpt: 쇼핑몰 웹 로그 데이터를 활용해서 DAU를 구해보자!
categories:
  - project
tags:
  - 

toc: true
toc_sticky: true
---

## DAU 구하기

먼저 퍼널 분석을 하기 전 데이터의 행이 너무 많은 관계로 DAU가 가장 높은 날을 기준으로 진행을 하려고 합니다.   
1편 마지막 SELECT문 출력을 보면 event_time 컬럼의 값들이 UTC 시간대로 나타나는 것을 볼 수가 있습니다. 사실 UTC로 진행을 해도 무관하지만, 한국 표준 시간대로 바꾸는 것도 익힐겸 한국 시간으로 바꿔서 분석을 진행했습니다.   
빅쿼리는 `DATETIME` 함수를 사용하면 시간대 변경이 가능합니다.

![DAU1](https://github.com/wbin0718/shoppingmall_weblog/assets/104637982/fcd6bef9-3439-48ca-92a3-57e3cad5067c)

위 쿼리문을 작성할 때 DAU를 구하기 위해서는 event_time 컬럼을 활용해서 DAY를 추출해야 합니다.   
빅쿼리는 `EXTRACT` 함수를 사용하면 원하는 YEAR, MONTH, DAY를 추출할 수 있습니다. DAY를 추출하고 DAY로 그룹화를 하여 user_id를 통해 DAU를 구했습니다.

![Untitled](https://github.com/wbin0718/shoppingmall_weblog/assets/104637982/463570b3-cae1-45a3-92e4-73ffed3fb7e1)

쿼리 추출 결과를 DAU 내림차순으로 정렬을 하면 17일이 가장 높은 DAU인 것을 알 수가 있습니다.   
따라서 17일 기준으로 분석을 해 보겠습니다. 아래 쿼리는 17일 데이터만 추출하는 쿼리입니다.      

![Untitled](https://github.com/wbin0718/shoppingmall_weblog/assets/104637982/965527e5-527c-40b2-915c-8824c11e18a1)

WHERE절은 한국 시간대로 바꾸고 SELECT는 UTC로 가져오는 실수를 했는데, 쿼리 순서를 생각해서 SELECT문도 한국 시간대로 변경해서 추출해야 합니다!   
아래 쿼리를 보면 UTC가 T로 변경되고, 17일 데이터만 추출해온 것을 볼 수가 있습니다.

![Untitled](https://github.com/wbin0718/shoppingmall_weblog/assets/104637982/53f03a78-a0ce-41b2-bdff-12947b878b73)