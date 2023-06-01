---
title: "[데이터 분석] 쇼핑몰 웹 로그 분석 프로젝트-4"
excerpt: 쇼핑몰 웹 로그 데이터를 활용해서 퍼널 분석을 진행 해 보자!
categories:
  - project
tags:
  - 

toc: true
toc_sticky: true
---

### 퍼널 분석

빅쿼리를 통해 간단한 데이터 탐색을 진행했었는데요, 이번에는 퍼널 분석을 진행하려고 합니다. 파이썬을 사용하면 훨씬 더 편리한 분석 진행을 할 수가 있어 빅쿼리를 파이썬과 연동해서 사용하겠습니다.   

![Untitled](https://github.com/wbin0718/shoppingmall_weblog/assets/104637982/c55c872b-43fe-465e-bdda-e8143f470bea)

파이썬 코드를 보시면 json 파일을 사용하게 됩니다. json 파일은 구글 클라우드 플랫폼으로 가서 키를 발급 받으면 주어지게 됩니다. 주어진 키를 다운로드하고, 경로를 알맞게 설정하시면 됩니다. 아래 참고한 링크를 올려두겠습니다.   
[파이썬, 빅쿼리 연동](https://wooiljeong.github.io/python/python-bigquery/)

![Untitled](https://github.com/wbin0718/shoppingmall_weblog/assets/104637982/28c3fcb5-7f9b-4c6d-991e-b26b52e4ecec)

파이썬과 연동을 하게 되면 파이썬으로 쿼리를 짜고, 원하는 데이터를 추출할 수가 있습니다.   
위와 같은 코드를 작성하면 바로 실행이 될줄 알았는데..!! 코드를 짤 때 뭐든 그냥 넘어가는 일은 없는 것 같아요ㅋㅋㅋ   
db-types라는 패키지를 설치하라는 오류가 나왔고, 설치를 했는데도 같은 오류가 떠서 검색을 했더니.. 스택오버 플로우를 잘 읽어보니 해결이 되었습니다.(커널을 재시작하라고 말씀하시네요ㅋㅋ)
[빅쿼리 실행 오류](https://stackoverflow.com/questions/72511979/valueerror-install-dbtypes-to-use-this-function)   

다시 위 쿼리를 실행하니   

![Untitled](https://github.com/wbin0718/shoppingmall_weblog/assets/104637982/c8d38b6b-0f12-4bb3-ae51-8abaa828a4a1)   

원하는 결과가 나온 것을 볼 수가 있습니다!! 위 쿼리는 사용자 마다 보기, 장바구니, 구매 3개의 행동 이벤트 기록을 추출한 데이터 입니다.   
퍼널 분석을 위해 피벗 형태의 데이터로 만들어주었습니다.   

![Untitled](https://github.com/wbin0718/shoppingmall_weblog/assets/104637982/70908f34-ac26-4b6e-b0a4-94443012fa77)   

위 피벗 테이블은 특정 고객이 view, cart, purchase 페이지로 넘어간 시간을 보여주고 있습니다. 이를 통해 각 단계로 넘어가는 전환율을 분석 해 보려고 합니다.   
파이썬으로 피벗 테이블을 만들 때, pivot 함수와 pivot_table 함수가 있습니다. 두 함수의 용도는 같지만 pivot_table이 집계 연산을 수행하는 역할도 한다고 합니다!   

![Untitled](https://github.com/wbin0718/shoppingmall_weblog/assets/104637982/b5b12904-8f31-4490-9a8a-4390e5171de3)   

파이썬 plotly 라이브러리를 사용하면 자동으로 퍼널 모양의 차트를 그려줍니다. 하지만 퍼센트가 아닌 숫자로 표시가 되어 전환율이 얼마나 되는지 파악하기가 어려워 seaborn 라이브러리를 통해 barplot으로 전환율 차트를 그려보았습니다.   

![Untitled](https://github.com/wbin0718/shoppingmall_weblog/assets/104637982/91637ee7-8b12-471a-af9d-5615730a0144)   
![Untitled](https://github.com/wbin0718/shoppingmall_weblog/assets/104637982/de35569d-a8d6-4cec-a93e-4e1d725f2160)   

그래프를 보면 view -> cart의 전환율은 23.35% 이며, cart -> purchase 전환율은 50.68%를 보이고 있습니다.   
장바구니로 상품을 담은 고객의 50%는 구매로 이어지는 것을 볼 수가 있습니다. 하지만 상품을 장바구니로 담는 고객은 그리 많지 않은데요. 고객들이 장바구니로 담기 위한 마케팅을 기획하거나, 어떤 불편함이 있어 장바구니로 담지 않는지를 분석해 UI/UX를 개선하는 방안을 마련할 수 있습니다.   

이처럼 3개의 파트로 나누어 웹 로그 분석을 진행 해 보았습니다.   
퍼널 분석은 많은 웹 혹은 앱을 제공하는 기업들은 많이 사용하고 있으며, 이 부분에서 문제점을 찾아 그로스 해킹을 이루려는 시도도 이루어지고 있습니다.   
여기서 끝이 아닌 실제로 실시간으로 쌓이는 현업 데이터를 보며 문제점 분석 및 개선 방안을 마련하고 A/B test를 진행 해 개선된 서비스를 제공으로 이어질 수 있습니다.