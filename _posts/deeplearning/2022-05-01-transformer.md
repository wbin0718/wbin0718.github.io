---
title:  "[논문 리뷰] 트랜스포머"
excerpt: "트랜스포머 논문을 읽어보자!!"

categories:
  - NLP
tags:
  - [DL]

toc: true
toc_sticky: true
---

# Attention is all you need

# Abstract

* 주로 transduction 모델들은 인코더 디코더를 포함한 순환신경망이나 합성곱 신경망 등이 대부분 차지하고 있다. 그중 가장 좋은 성능을 가진 모델은 attention 메커니즘을 사용한 인코더 디코더 구조였는데 이들은 모두 순환신경망, 합성곱 연산이 포함되어있다. 따라서 논문에서는 오로지 어텐션 메커니즘을 사용하고 순환신경망과 합성곱은 사용하지 않는 트랜스포머라는 새로운 네트워크를 제안한다.

* 논문에서 제안한 트랜스포머는 두가지 task에서 좋은 성능을 보였는데 영어-독일어를 번역하는 task에서 기존 sota모델보다 2BLEU가 높았으며 영어-프랑스어 번역에서도 좋은 성능을 보였다.

* 트랜스포머는 적은 데이터나 큰 데이터 어디에서도 English constituency parsing에서 일반화가 잘 됨을 보였다.

# Introduction

![image](https://user-images.githubusercontent.com/104637982/166153243-d3ecc963-ffb6-42c7-9d2a-ff73db0a52d0.png)

*  주로 RNN, LSTM, GRU등이 언어모델링, transduction 문제에서 SOTA모델로 자리매김하고 있었는데 많은 사람들이 인코더-디코더 구조와 recurrent 문제의 경계를 넘나드는 노력을 해왔다.

* recurrent 모델들은 input과 output의 순서의 위치에 따라 분해하여 계산을 한다. 즉 계산 시점에서 매 스텝마다 위치들을 정렬하면서 은닉상태 h_t를 이전 시점 h_t-1과 input인 t로 생성한다.
이렇게 본질적으로 순서가 있는 특성은 훈련할 때 병렬화를 할 수 없는데 메모리 제약 문제는 배치화를 할 수 없기 때문에 병렬화는 문장의 길이가 더 길 때는 더 중요한 문제가 된다.
위처럼 병렬화를 할 수 없기 때문에 recurrent 모델은 계산 효율성이 떨어진다고 볼 수 있는데, 최근 연구들은 factorization tricks와 conditional computation등을 통하여 많은 계산 효율의 향상을 이뤄냈고 후자의 경우 모델 성능의 향상 또한 있었다.
하지만 근본적으로 순서대로 계산하는 제약은 남아있었다.

* 어텐션 메커니즘이 sequence modeling 그리고 transduction 모델이 다양한 task에서 가장 중요한 부분이 되었는데, input과 output sequence가 거리에 상관없는 의존성 있는 모델링을 할 수 있었다.
하지만 대부분으 그러한 어텐션 매커니즘은 recurrent 네트워크와 함께 사용되었다.

* 따라서 Transformer를 제안하는데, 이는 recurrece한 모델의 사용을 없애고 대신에 input과  output의 global한 dependencies를 이끌 수 있는 오로지 어텐션 매커니즘에 의존하는 구조이다.
Transformer는 병렬화를 할 수 있으며 번역 task에서 12시간 훈련을 시킨 모델로 SOTA를 달성할 수 있었다.

# Background

* 잇달아 일어나는 계산을 줄이려는 목표는 Extended Neural GPU, ByteNet, ConvS2S의 기반을 형성했다. 이 모델들은 합성곱 뉴럴 네트워크를 building block으로 사용했는데, input과 output을 병행하여 숨겨진 표현을 계산한다. 이러한 모델들은 두개의 임의의 input과 output으로 부터 신호를 관련시키기 위해 필요한 계산 수는 위치의 거리에 따라 증가하는데, ConvS2s는 선형적으로, ByteNet은 logarithmically하게 증가한다.
따라서 이는 멀리 떨어진 위치들 간에는 의존성을 학습하기를 어렵게 한다.
Transformer에서는 계산 수를 일정 상수로 줄였는데, 이는 어텐션 가중합을 평균내기 때문에 감소된 효율적인 해상도의 비용이 있지만, 이후에 나올 멀티 헤드 어텐션으로 이를 상쇄시킬 수 있다.

* intra-attention이라 불리는 self-attention은 sequence의 표현을 계산하기 위한 sequence의 다른 위치들을 관련시키는 어텐션 메커니즘이다. self-attention은 다양한 task에서 성공적으로 사용되었다.

* End-to-end memory 네트워크는 순서대로 정렬된 recurrence 대신에 recurrent attention 매커니즘에 기반을 두었는데, 단순한 언어 question answering 과 언어 모델링 task에서 좋은 성능을 보였다.

* 하지만 우리가 아는 한에 Transformer는 순서대로 정렬된 RNN, 합성곱 모델의 사용없이 input과 output의 표현을 계산하기 위해서 오로지 self-attention에 의존하는 첫번째 transduction 모델이다.

# Model Architecture

![image](https://user-images.githubusercontent.com/104637982/166153264-d97704a3-63e8-489e-a660-5dd6de81fc4b.png)


* 가장 경쟁력있는 neural sequence transduction 모델은 인코더-디코더 구조이며 인코더는 input sequence (x1, ..., xn)를 연속된 표현 z = (z1, ..., zn)로 매핑시킨다. 디코더는 z라는 표현이 주어지면 output seqeunce인 (y1, ..., ym)를 한 시점에 하나씩 만들어낸다. 각 단계에서 모델은 다음 단어를 예측할 때 추가적인 input과 이전에 만들어진 단어표현을 사용하는 auto-regressive이다.  

* Transformer는 이 전체적인 구조를 인코더와 디코더에서 self-attention과 point-wise fully connected layers를 쌓아올리면서 사용하였다.

## Encoder and Decoder Stacks

* Encoder : 인코더는 동일한 6개의 층으로 구성되어 있다. 인코더는 두개의 서브 층을 가지고 있는데, 첫번째는 multi-head-attention 매커니즘이고, 두번째는 단순한 position-wise fully conntected feed-forward network이다. 각 sub-layer층에서 잔차연결을 사용했으며 그 후에 잔차연결을 층 정규화를 해주는 구조이다. 즉 각 sub-layer의 output은 **LayerNorm(x + Sublayer(x))**이며 **Sublayer(x)**는 sub-layer 그자체로 사용된 함수이다. 이 잔차연결을 용이하게 하기 위해서 embedding층 뿐만 아니라 model내의 모든 sub-layers의 output의 차원을 **dmodel = 512**로 하였다.

* Decoder : 디코더 또한 동일한 6개의 층으로 구성되어 있다. 두개의 sub-layers이외의 디코더는 세번째 sub-layer를 사용하였는데, 이는 인코더의 출력물과 multi-head-attention을 수행한다.
디코더도 인코더와 동일하게 잔차연결을 사용한 후 층 정규화를 적용했다. 디코더에서 self-attention sub-layer를 각 위치가 뒤에나오는 위치를 참고할 수 없게 하기 위해서 수정을 했다.
이 마스킹 기법은 i번째 단어를 예측할 때 i보다 작은 단어들의 outputs에만 의존할 수 있도록 보장한다.

## Attention

* 어텐션 함수는 query와 key, value의 쌍을 한 output으로 매핑하는 역할을 한다고 설명할 수 있다.
여기서 query, key, value, output은 모두 벡터이다. output은 values들을 가중합하여 계산이 되며 각 value에 할당되는 가중치들은 query와 그에 대응하는 key와의 계산에 의해서 구해진다.    
![image](https://user-images.githubusercontent.com/104637982/166153309-118cabff-4e87-4fc8-b5b9-c4c86f3beb19.png)

![image](https://user-images.githubusercontent.com/104637982/166153325-70b42a4a-2a98-4955-bc75-fc90128856a8.png)

### Scaled Dot-Product Attention

* 특정 어텐션을 Scaled Dot-Production Attention이라고 부른다. input은 d_k차원의 query, key와 d_v의 values로 이루어져있다. 각 query와 모든 key간에 내적을 하고 √d_k로 나누어 계산하고, values에 대하여 가중치를 얻기 위해서 softmax 함수를 적용해준다. 이때 계산할 때는 행렬 Q를 만들어 동시에 계산하며 keys와 values는 행렬 K와 V를 통하여 행렬 계산을 하여 한번에 계산을 한다.

![image](https://user-images.githubusercontent.com/104637982/166153354-17918286-fcd5-4679-a7eb-84d43c2933db.png)

* 두개의 주로 사용되는 어텐선 함수는 additive attention과 dot-product attention이다. Scaled Dot-Product Attention은 dot-product attention에서 √1/dk로 scaling 한것 이외에는 동일한 알고리즘이다. additive attention은 single hidden layer와 feed-forward network를 사용하여 계산한다.
두개가 비슷한 복잡성을 갖지만 dot-product이 매우 최적화된 행렬 곱으로 구현되기 때문에 attention이 더 빠르고 실제로 공간 효율이 좋다.
d_k 값이 작을때는 두 매커니즘 모두 비슷하게 성능을 보이지만 큰 d_k의 값에서는 d_k로 scaling 하지 않을때 additive attention의 성능이 dot product attention을 능가했다. d_k 값이 커지면 dot products의 벡터의 크기도 커지기 때문에 softmax함수를 거치면 0에 가까운 값들이 많다. 그러면 기울기가 작아지므로 1/√d_k로 scale을 한다.   

![image](https://user-images.githubusercontent.com/104637982/166153371-76e5be5c-f99d-4e49-8b46-19b938d83b70.png)

![image](https://user-images.githubusercontent.com/104637982/166153388-57188122-da83-4f18-9a54-6682761e8133.png)

### Multi-Head Attention

* d_model 차원의 keys, values, queries와 하나의 어텐션을 수행하는 것 대신에 다양하게 학습된 d_k, d_k, d_v h번 선형적으로 project하는 것에 대한 장점을 발견했다. 이 queries, keys, values의 projected versions에서 d_v차원의 output values를 만들면서 어텐션 함수를 병렬적으로 수행한다.
이 값들은 concatenated 되고 다시 projected 되면서 마지막 values를 만들어낸다.
Multi-head attention은 다양한 위치에 있는 다양한 표현들로 부터 정보를 학습할 수 있도록 한다.

* 단일 attention head에서는 averaging이 이것을 억제한다.

* h=8개로 8개의 병렬 attention layers를 사용했다.
d_k = d_v = d_model/h=64를 사용했다. 각 head에서 축소된 차원때문에 전체 계산비용은 완전히 full dimensionality를 사용한 하나의 어텐션과 동일하다.

### Applications of Attention in our Model

* Transformer는 다른 세가지 방식으로 multi-head attention을 사용한다.

* 인코더-디코더 어텐션 층에서는 이전 디코더 층에서 출력된 queries와 인코더의 출력으로 부터 온 memory keys와 values를 사용한다.
이는 디코더의 매 위치에 있는 단어들이 input sequence의 모든 위치를 참고할 수 있도록 한다. 이는 시퀀스투시퀀스 모델에서의 인코더-디코더 어텐션 매커니즘을 모방한다.

* 인코더는 self-attentionn layers를 포함하고 있다. self-attention에서는 모든 keys, values, queries가 같은 장소에서 만들어지며 인코더의 이전 층의 output으로부터 만들어진다.

* 비슷하게 디코더의 self-attention layers는 디코더의 각 위치에 있는 단어전까지의 디코더의 모든 단어들과 정보를 교환할 수 있도록 한다. auto-regressive 특성을 보존하기 위해서 디코더의 정보 흐름을 막을 필요가 있다. 이것을 scaled dot-product attention 내부에서 각 위치에 있는 단어의 이후 단어들을 softmax 하기 전에 -inf 값을 주면서 masking을 구현했다.

## Position-wise Feed-Forward Networks

* 어텐션 sub-layers 이외에도 인코더-디코더의 각 층은 fully connected feed-forward network를 포함하며 각 위치에 해당하는 token에 개별적으로 동일하게 적용된다. 이는 두개의 선형변환으로 구성되어 있으며 그 사이에 ReLU함수가 포함되어 있다.

![image](https://user-images.githubusercontent.com/104637982/166153431-f9aaa647-b520-4ec9-b144-305be73a2101.png)

* 선형변환은 다양한 위치마다 같지만 층마다 다른 파라미터를 사용한다. 이것을 묘사하는 다른 방법은 kernel size가 1인 합성곱 신경망으로써 할 수 있다.
input과 output의 차원은 d_model = 512이며 inner-layer의 차원은 d_ff = 2048을 갖는다.

## Embeddings and Softmax

* 다른 sequence transduction 모델들과 비슷하게 d_model의 차원을 갖는 벡터로 input tokens와 output tokens를 사용하기 위해서 학습된 embeddings 벡터를 사용했다. 또한 다음 token 확률을 계산하기 위하여 학습된 선형변환 층과 softmax 함수를 사용했다. 모델에서 두 임베딩 층과 softmax를 사용하기 전의 선형변환 층에서는 가중치 행렬을 공유했다. 임베딩 층에서는 가중치에 √d_model을 곱해주었다.

## Positional Encoding

* Transformer는 recurrence와 convolution이 없기 때문에 sequence의 순서를 모델이 이해하도록 하기 위해서 tokens의 상대적, 절대적 위치에 관한 정보를 추가했다. 요약하자면 **positional encodings**를 인코더-디코더 맨 아래층에 있는 input embeddings에 추가했다. positional encodings는 임베딩 벡터와 positional encoding 벡터와 더하기 위해서 d_model 과 같은 임베딩 차원을 가진다. 학습가능하고 고정된 positional encodings의 선택지는 많이 있다.

* 트랜스포머에서는 sine, cosine 함수를 사용한 positional encodings을 사용했다.

![image](https://user-images.githubusercontent.com/104637982/166153461-e4fb8acc-afa9-4ccf-9fe3-d7be6af17d5d.png)

* pos는 위치를 나타내고 i는 벡터의 차원 인덱스를 나타낸다. 즉 positional encoding의 각 차원은 sinusoid와 상응한다. 2π to 10000 · 2π 부터 wavelengths는 기하학적 progression을 형성한다. 트랜스포머는 이 함수를 사용했는데 그 이유는 모델이 더 쉽게 학습할 수 있도록 가설을 세웠기 때문이다. 고정된 크기의 offset k 그리고 P Epos+k가 PEpos 선형변환으로서 표현될 수 있다.

* 학습하는 positional embeddings를 사용하여 실험을 했는데, 두가지의 동일한 근처의 결론을 얻었다. 모델이 훈련하는 동안 더 긴 sequence에 해당하는 sequence를 extrapolate를 할 수 있어 트랜스포머는 sinusoid version을 사용하여 positional encoding을 진행했다.

# Why Self-Attention

* self-attention layers와 가변길이의 sequence의 표현들을 동일한 길이의 다른 sequence 표현으로 mapping 하는 recurrent와 convolution layers를 비교할 것이다. self-attention의 사용을 동기부여하면서 세가지 관점에서 비교할 것이다.

* 첫번 째는 층마다 전체 계산 복잡성이고, 두번 째는 병렬화 될 수 있는 계산량 (sequential operations의 최소한의 수)이다.

* 세번 째는 네트워크에서 long-range dependencies사이의 path length이다. long-range dependencies는 다양한 sequence transduction task들에서 중요한 문제이다. 그러한 dependencies를 학습하는데 영향을 미치는 중요한 요인은 네트워크에서 forward와 backward 신호들이 횡단할 필요가 있는 paths의 길이이다. input과 output sequences에서 positions의 조합 사이에서의 paths가 짧으면 짧을수록 long-range dependencies를 학습하기가 더 쉽다. 따라서 다양한 layer로 구성된 네트워크에서 input, output positions 사이의 최대 path 길이를 비교했다.

* self-attention layer는 순차적으로 실행된 operations의 일정한 상수와 함께 모든 positions를 연결시킨다. 반면에 recurrent layer는 O(n)의 sequential operations를 필요로 한다. 계산 복잡성의 관점에서 self-attention layers는 word-piece와 byte-pair과 같은 기계 번역에서 SOTA로 사용된 문장 표현에서 n이 표현 차원 d보다 작을때 self-attention은 recurrent layers보다 더 빠르다. 매우 긴 sequences한 tasks들을 포함하여 계산하는 성능을 향상시키기 위해서 self-attention은 각각의 output position 주변에 중심화된 input sequence에서 r 크기의 이웃만을 고려하도록 제한될 수 있다. 최대 path length를 O(n/r) 까지 증가시킬 수 있다. 향후 논문에서 이 접근을 더욱 연구할 계획에 있다.

* n보다 작은 k의 kernel width를 갖는 하나의 합성곱 층은 모든 input과 output 위치의 쌍들을 연결할 수 없다. 그렇게 하는 것은 네트워크에서 두 위치간에 가장 긴 paths를 증가시키면서 contiguous kernels에 대비하여 O(n/k)를 요구하고 dilated convolutions에 대비하여 O(logk(n))를 요구한다. convolutional layers는 일반적으로 k 요인에 의해서 recurrent layers보다 더 비싼 연산을 한다. 하지만 separable convolutions는 O(k·n·d + n·d^2)까지 계산 복잡성을 상당히 감소시킨다. 하지만 k=n에서는 separable 복잡성은 transformer에서 사용한 접근법인 self-attention의 조합과 point-wise feed-forward layer와 동일하다. 부수적인 혜택으로 self-attention은 해석 가능한 모델을 산출한다. transformer로부터 어텐션 분포를 확인하고 보여주고 부록의 예시를 토론한다. 각각의 어텐션 헤드들은 다양한 tasks들을 수행하기 위하여 학습할 뿐만 아니라 문장의 syntactic 하고 semantic 구조와 관련된 행동들을 억제하는 것처럼 보인다. 

# Training

## Training Data and Batching

* 4.5 million의 쌍을 갖는 WMT 2014 영어-독일어 데이터를 학습시켰다. 문장들은 byte-pair 인코딩을 사용하여 인코딩 되었고 37000개의 vocabulary를 가지고 있다. 영어-프랑스 데이터에서는 36M개의 문장을 사용했고 32000개의 word-piece vocabulary로 나눴다. 문장 쌍은 sequence length로 배치했다. 각각의 훈련 배치들은 대략 25000개의 tokens와 25000개의 target tokens로 구성되었다. 

## Hardware and Schedule

* 모델을 8개의 NVDIA P100 GPU를 사용하여 훈련시켰다. 논문에서 설명한 하이퍼파라미터를 사용한 base models는 각각 훈련 step에서 0.4초가 걸렸다. base models를 100000 steps를 훈련시켰으며 12시간을 훈련시켰다. big models에서는 훈련 step에서 1초가 걸렸다. big models는 300000 steps를 훈련시켰으며 3.5일이 걸렸다.

## Optimizer

![image](https://user-images.githubusercontent.com/104637982/166153514-a9efaf92-240a-47e4-b462-0f23741c4697.png)

## Regularization

* 학습하는 동안 3개의 규제를 사용했다.
* 첫번 째는 각 sub-layer의 출력 값을 dropout(0.1)을 적용했다.
* 임베딩 벡터와 positional encoding의 합 벡터를 dropout을 적용했다.
* label smoothing을 적용했다.

# Results

## Machine Translation

* 영어-독일어 task에서 이전에 달성한 SOTA 모델보다 BLEU 점수를 2.0이나 앞지르면서 좋은 성능을 보였다. GPU 8개로 3.5일이 학습하는데 걸렸다. transformer와 경쟁력을 갖추는 모델들보다 학습 비용 부분에서 앞질렀다.

* 영어-프랑스 task에서 이전에 달성한 SOTA 모델보다 학습 비용을 1/4을 낮추면서 성능을 능가했다. 영어-프랑스 task에서는 dropout을 0.3으로 설정했다.

* base model에서는 10분 간격으로 마지막 5개를 평균을 냈으며 big model에서는 마지막 20개를 평균냈다.

* beam search를 beam size를 4로 하였고 length penalty α = 0.6으로 했다. 이 파라미터는 실험을 거친 후에 정해졌다. 추론을 할때는 output length의 길이를 input length 길이의 +50으로 설정했으며 가능할 때 일찍 종료했다.

* 아래에서 다른 모델과 transformer를 번역 품질과 학습 비용에 관해서 비교하였다. transformer에 사용된 부동소수점의 연산을 훈련시간, 사용된 GPU의 수, 각 GPU의 단정도 수용능력을 곱하면서 추정했다.

## Model Variations

* transformer의 다른 구성요소의 중요성을 평가하기 위해서 영어-독일어 번역 task에서 성능의 변화를 확인하면서 base model을 다양한 방식으로 다르게 했다.
* attention heads, attention key, attention value의 차원을 다르게 했으며 computation 상수는 유지했다. single-head attention은 best 환경보다 0.9 BLEU가 낮았으며 많은 heads로 transformer를 구성하면 번역 품질이 낮아졌다.

* attention key 크기인 d_k를 줄이면 모델 성능이 좋지 않았으며 예상한대로 모델이 클 수록 성능이 더 좋았고, dropout은 과적합을 피하는데 도움이 되었다.
positional encoding에서는 sinusoidal positional encoding을 학습되는 positional embeddings를 사용했는데 base model에서 성능은 동일했다.

## English Constituency Parsing

* transformer가 다른 tasks에서도 일반화를 잘 하는지 평가하기 위해서 english constituency parsing 실험을 수행했다. 이 tasks는 어려움을 보이는데, output이 구조적 제약이 있고, input보다 상당히 길다. 게다가 rnn sequence-to-sequence 모델들은 작은 데이터에서는 SOTA를 달성해올 수 없었다.

* 40K의 훈련 문장을 가지고 d_model = 1024, 4개의 층을 갖는 transformer를 학습했다. 또한 대략 17M의 문장과 함께 larger high-confidence와 BerkleyParser corpora를 사용하면서 semi-supervised setting에서 훈련을 했다. WSJ 환경에서는 16K의 단어집합과 semi-supervised setting에서는 32K의 단어집합을 사용했다.

* 실험을 수행할 때 dropout, attention, residual, learning rates, beam size만을 설정했고 나머지 파라미터들은 영어-독일어 번역모델과 동일하게 설정했다. 추론을 하는 동안 output의 길이를 input 길이의 +300을 설정했다. WSJ, semi-supervised setting에서 모두에서 beam size를 21로 설정했고, α = 0.3으로 설정했다.

* task에 맞는 구체적인 tuning 없이도 Recurrent Neural Network Grammer를 예외하고 이전에 기록된 모델들 보다 transformer는 좋은 성능을 냈다.

* RNN sequence-to-sequence 모델과 대조하요 transformer는 WSJ의 40K 문장으로 학습을 하는 경우에도 Berkeley-Parser에서 성능이 좋았다.

* 논문에서 multi-headed self-attention과 함께 인코더-디코더 구조에서 주로 사용된 recurrent layers를 대신하면서 오로지 attention을 사용한 첫번 째 sequence transduction model인 transformer를 제시했다.

* 번역 tasks에서 transformer는 recurrent나 convolutional layers를 사용한 구조보다 훈련 속도가 빨랐다. 영어-독일어와 영어-프랑스 번역 tasks에서 SOTA를 달성했다. 영어-독일어 task에서는 transformer는 이전에 기록된 앙상블 모델보다 좋은 성능을 보였다.

* 어텐션에 기반을 둔 모델들의 미래에 대해서 흥미로우며, 다른 tasks들에도 transformer를 적용하려고 계획할 것이다. text 말고도 input과 output modalities를 포함한 문제와 investigate local, restricted attention mechanisms에 image, audio, video와 같은 더 큰 input과 output을 효율적으로 다루기 위해서 transformer를 연장하는 것을 계획할 것이다. sequential이 덜하는 generation을 만드는 것은 다음 연구의 목표이다.


