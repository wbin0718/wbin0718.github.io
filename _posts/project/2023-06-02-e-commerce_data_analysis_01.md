---
title: "[데이터 분석] e-commerce 데이터 분석-1"
excerpt: 브라질 기업 olist 데이터를 활용해 데이터 분석을 진행 해 보자!
categories:
  - project
tags:
  - 

toc: true
toc_sticky: true
---

## 이커머스 데이터를 활용한 데이터 분석

캐글 데이터인 브라질 olist 기업 데이터를 활용한 데이터 분석을 진행 해 보려고 합니다. 캐글 노트북을 참고하여 작성하였으며, 분석하고 싶은 부분만을 작성했습니다:)   
   
[참고 캐글 노트북](https://www.kaggle.com/code/seungbumlim/exploratory-e-commerce-data-analysis-using-sqlite)   

제가 작성한 빅쿼리 코드는 [링크](https://github.com/wbin0718/e-commerce_data_analysis/blob/master/e_commerce_sql.ipynb)를 참고해 주세요!

## 재구매율 분석

해당 데이터는 총 9개의 테이블을 가지고 있습니다. order 라는 핵심 테이블 하나와 다른 8개 테이블이 각자의 기본키, 참조키 형태로 연결되어 있습니다.   
먼저 사람들이 가장 많은 주문을 한 제품 카테고리가 무엇인지 확인 해 보겠습니다.

<br>
![image](https://github.com/wbin0718/shoppingmall_weblog/assets/104637982/acd5f017-2523-49dd-867d-04df1ef9b333)   

사람들이 가장 많이 주문한 카테고리 3개는 bed_bath_table, health_beauty, computers_accessories 입니다.

따라서 이 3개 제품의 재구매율은 어떻게 되는 지 살펴보겠습니다!   
<br>
![image](https://github.com/wbin0718/shoppingmall_weblog/assets/104637982/470e305a-a29e-45b4-9915-d5f4055ee343)

먼저 bed_bath_table을 주문한 고객만을 쿼리로 추출했습니다. 그 다음 고객의 첫 번째 구매 날짜, 마지막 구매 날짜, count_order라는 고객 제품 구매 횟수, 재구매 여부 repurchase, interval_pur라는 마지막 구매 날짜 - 첫 번째 구매 날짜, 구매 주기인 interval_pur / (count_order - 1) 를 나타내는 cycle_pur 컬럼을 만들었습니다.

<br>

![image](https://github.com/wbin0718/shoppingmall_weblog/assets/104637982/f2cba375-02b5-4236-9ad8-4b5d6eb7c896)

가장 많이 주문한 상품인 bed_bath_table의 경우 재주문율은 2.4433%로 매우 낮은 것을 볼 수가 있습니다.   
위와 같은 방식으로 WHERE 절을 bed_bath_table을 다른 2개 상품으로 바꾸어 적용하면 health_beauty 재주문율은 1.49%, computer_accessories 재주문율은 1.71%로 낮은 것을 볼 수 있습니다.   