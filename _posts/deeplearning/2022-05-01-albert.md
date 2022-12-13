---
title:  "[논문 리뷰] ALBERT"
excerpt: "ALBERT 논문을 읽어보자!!"

categories:
  - NLP
tags:
  - [DL]

toc: true
toc_sticky: true
---

# Abstract

* natural language representations를 pretraining 할때 model size를 늘리는 것은 downstream tasks의 성능을 향상시킨다. 그러나 계속해서 model의 size를 증가시키는 것은 GPU/TPU 메모리 제한과 훈련 시간으로 인해 어렵다.

* 이러한 문제점을 다루기 위해 메모리 소비를 줄이고 BERT의 훈련속도를 증가시키기 위한 2가지 파라미터 감소 기법을 제시한다.

* 기존 BERT의 scale한 모델이다.

# Introduction

* 사전 학습 시키는 모델의 크기가 클수록 성능은 향상된다고 알려져있고 큰 모델을 사전 학습하고 실제 task에서는 모델의 크기를 줄이는 것이 일반적인 관행이다. 그렇다면 모델의 크기를 크게 하면 NLP 모델의 성능이 좋을까?

* 현재 SOTA 모델들은 많은 파라미터들을 가지고 있고, 이는 메모리 부족 문제에 마주친다. 그래서 우리는 모델의 크기를 scale 하고자 한다. communication overhead도 파라미터의 개수와 비례하기 때문에 학습속도도 느려질 수 있다.

* 앞에서 언급한 문제들의 해결책으로 model parallelization, clever momory management가 있지만 이러한 해결책은 메모리 제한 문제만 해결할 뿐 communication overhead의 해결책은 아니라고한다. 따라서 두 문제를 해결하기 위해 BERT보다 파라미터 수를 줄인 ALBERT를 설계했다.

* ALBERT는 두 가지의 파라미터 감소 기법을 사용한다. 첫번 째는 **factorized embedding parameterization** 이다. input 값을 임베딩 벡터로 바꾸려고 할 때 V x H 만큼의 파라미터가 필요하다. **ALBERT는 이를 (V x E + V x H) 로 분리한다. 두번 째는 **cross-layer parameter sharing** 이다. 이는 network의 깊이가 깊어져도 파라미터의 수가 증가하지 않는다.

* ALBERT의 성능을 향상시키기 위해 SOP를 사용했다.

# Related Work

## Scaling Up Representation Learning For Natural Language

* 지난 2년동안 가장 큰 변화 중 하나는 사전 학습된 임베딩 벡터의 사용이 full-network를 pre-training 하고나서 task-specific인 fine-tuning을 수행하는 것이다. 이러한 모델들은 모델의 크기가 클수록 성능이 향상하는 것을 보여주었다. 예를들어 BERT는 larger hidden size, more hidden layers, more attention heads를 사용하면 성능을 향상시키는 것을 보여주었다. 그러나 BERT는 hidden size를 1024에서 멈췄으며, 아마도 model size와 computation cost 문제 때문일 것이다.

* large models는 computational constraints 때문에 실험을 하는 것이 어렵다. SOTA 모델들은 많은 파라미터들을 가지고 있어서 메모리를 쉽게 hit 할 수 있다. 이를 다루기 위해 Chen et al. (2016)은 gradient checkpointing을 제안하고 Gomez et al. (2017)은 reconstruct each layer's activations를 제안했다. 두 방법 모두 메모리 consumption을 감소시킨다. Raffel et al. (2019)은 giant model을 학습시킬 때 model parallelization을 사용하는 것을 제안했다. 대조적으로 우리의 파라미터 감소 기법들은 메모리 consumption을 감소시키고 학습 속도를 증가시킨다.

## Cross-layer Parameter Sharing

* 가중치를 층간 공유하는 것은 Transformer 구조로 계속 탐구되어왔다. 하지만 이전 방법들은 pretraining/finetuning이 아닌 encoder-decoder를 tasks로 학습을 하는데 초점을 맞췄다.

* Bai et al. (2019)는 DQE를 제안했으며 특정 층의 input embedding과 output embedding이 같은 층에 있을 때 equilibrium point를 도달할 수 있는 것을 보여주었다.

* Hao et al. (2019)는 파라미터를 공유하는 transformer와 기본 transformer를 연결했고, 이는 기본 transformer의 파라미터 수를 증가시킨다.

## Sentence Ordering Objectives

* ALBERT는 두 연속적인 segments의 순서를 예측하는 pretrainig loss를 사용한다.

* 여러 연구자들은 discourse coherence와 관련된 pretraining objectives를 실험해왔다.

* BERT는 두번 째 segment가 다른 문서의 segment와 한 쌍을 이루는지 아닌지를 예측하는 loss를 사용한다.

* 우리의 실험은 이 loss를 비교했으며, 문장 순서를 맞추는 것은 더욱 어려운 pretraining task이며 특정 downstream tasks에서 더욱 유용한 것을 알아냈다.

* 우리의 work와 비슷하게 Wang et al. (2019)은 두 연속적인 문장의 순서를 예측하는 것을 시도했다. 하지만 NSP와 SOP를 결합하여 사용했다. 

# The Elements of ALBERT

* ALBERT의 design decisions를 보여주고, BERT 구조의 configurations와 상응하는 비교를 제공한다.

## Model Architecture Choices

* ALBERT의 backbone은 transformer encoder와 GELU 함수를 사용하는 BERT와 비슷하다.

* BERT notation conventions를 따르고, the vocabulary embedding sizse를 E, the number of encoder layers를 L, the hidden size를 H로 정의한다.

* BERT처럼 the feed-forward/filter size를 4H, the number of attention heads를 H/64로 설정한다.

### Factorized embedding parameterization

* BERT뿐만 아니라 XLNet, RoBERTa는 WordPiece embedding size E가 hidden layer size H와 연결되어 있다. 다시말해서 E=H이다.

* 모델링 관점에서 WordPiece embeddings는 context-independent representations를 학습하도록 의도되는 반면, hidden-layer embeddings는 context-dependent representations를 학습하도록 의도된다.

* BERT와 같은 representations의 강력함은 그러한 context-dependent representations를 학습하는 signal을 제공하는 context의 사용으로부터 온다. WordPiece embedding size E를 hidden layer size H로부터 풀어주는 것은 전체 model 파라미터들의 효율적인 사용을 하도록 해 준다. H > E를 의미한다.

* 실용적인 관점으로 보면 natural language processing은 the vocabulary size인 V가 large 한 것을 요구한다.

* E=H 일 때 H가 증가하면 V x E를 가지는 the size of the embedding matrix를 증가시킨다.

### Cross-layer parameter sharing

* ALBERT는 cross-layer parameter sharing을 parameter 효율성을 향상시키기 위한 방법으로 제안한다.

* parameters를 공유하는 방법은 여러가지가 있다. feed-forward network의 파라미터들만 공유하는 방법과 attention 파라미터들만 공유하는 방법이 있다. ALBERT는 기본 값으로 모든 층의 파라미터들을 공유한다.

### Inter-sentence coherence loss

* BERT는 MLM과 NSP를 사용했다. 후속 연구는 NSP의 영향은 믿을만 하지 못하고, NSP 없이도 여러 tasks의 성능이 향상되었기 때문에 NSP를 제거하기로 결정했다.

* 우리는 NSP의 비효율성의 주된 이유를 MLM과 비교했을 때 학습하기가 쉽다는 것으로 어림짐작 했다.

* 우리는 inter-sentence modeling은 language understanding의 중요한 측면이라는 점을 유지하고, coherence로 기본으로 하는 loss를 제안한다. 즉 ALBERT는 topic prediction을 피하고 inter-sentence coherence로 집중하는 SOP loss를 사용한다. SOP loss는 BERT와 같은 기법으로 positive examples를 사용하고 negative examples는 같은 연속적인 segments를 그들의 순서를 바꾸어서 사용한다. 이는 모델이 discourse-level coherence properties의 세밀한 차이를 학습하도록 해준다.

* NSP는 SOP tasks를 전혀 해결하지 못하지만, SOP는 아마도 misaligned coherence cues를 분석하는데 집중하는 정도로 NSP tasks를 해결할 수 있다.

## Model Setup

* ALBERT는 BERT보다 훨씬 적은 파라미터의 개수를 갖는다. ALBERT-large는 BERT-large보다 18x 적은 18M versus 334M의 파라미터를 갖는다.

* ALBERT-xlarge는 H = 2048를 갖고 60M의 파라미터 수를 가지며 ALBERT-xxlarge는 H = 4096을 갖고 233M의 파라미터 수를 갖는다. BERT-large의 70%이다. ALBERT-xxlarge는 24-layer network가 12-layer network와 비슷한 결과를 얻지만 더욱 비싼 연산이므로 12-layer network를 주로 보고한다.

* 파라미터 효율성이라는 향상은 ALBERT의 설계 선택의 가장 중요한 장점이다.

* 이러한 장점을 정량화 하기 전에, 더 많은 experimental setup를 소개할 필요가 있다.

## Experimental Results

## Experimental Setup

* 가능한 의미있는 비교를 하기 위해 BookCorpus와 English Wikipedia를 사용하여 pretraining baseline models를 하는 BERT setup을 따랐다.

* 이 두 corpora는 16GB의 압축되지 않은 text로 이루어져있다.

* 30000 단어사전을 사용하고, XLNet과 같이 SentencePiece를 사용하여 tokenized를 했다.

* 모든 모델의 updates는 4096의 batch size를 사용했고 learning rate 0.00176으로 LAMB optimizer를 사용했다. 각 무작위로 선택된 n-gram mask와 함께 n-gram masking을 사용하면서 MLM targets을 위한 masked inputs를 만들었다. 우리는 n-gram의 길이가 최대 3이 되도록 설정했다. 

* 우리는 구체화 하지 않았다면 125000 stepes를 모든 모델들을 학습시켰다. 훈련을 할 때 사용된 TPU의 수는 64에서 512까지이다.

## Evaluation Benchmarks

### Intrinsic Evaluation

* 훈련과정을 관찰하기 위해 SQuAD와 RACE로부터 development sets로 근거하는 development set을 만들었다.

* MLM과 sentence classification tasks의 정확성을 보고했다.

* 우리는 이 set을 model selection을 통하여 어떠한 downstream evaluation의 성능으로 영향을 미칠 수 있는 방식으로 사용하지 않았고, model이 어떻게 수렴하는지를 확인하려고 사용했다.

## Downstream Evaluation

* 우리는 모델을 세가지의 인기있는 benchmarks인 The General Language UnderStanding Evaluation benchmark와 two versions of the Stanford Question Answering Dataset과 the ReAding Comprehension from Examinations dataset을 사용했다.

* 우리는 early stopping을 development sets에서 수행했고, task leaderboards를 기초로 하는 final comparisons를 제외하고는 모든 비교를 보고했고, 또한 test set results를 보고했다.

* GLUE datasets은 dev set에서 large variances를 갖기 때문에 우리는 5runs를 넘는 median을 보고했다. 

### Overall Comparison Between BERT And ALBERT

* 파라미터 효율성의 향상은 ALBERT's design choices의 가장 중요한 장점이라고 보여준다. BERT-large의 70% 파라미터와 함께 ALBERT-xxlarge는 여러 representative downstream tasks를 위해 development set scores의 차이로 인해 측정된 BERT-large보다 성능 향상을 성취했다. SQuADv1.1(+1.9%), SQuADv2.0(+3.1%), MNLI(+1.4%), SST-2(+2.2%), RACE(+8.4%)

* 다른 흥미있는 관찰은 같은 training configuration 아래의 훈련시간에서 data throughput의 속도이다. 적은 communication과 적은 computations로 인하여 ALBERT 모델은 BERT 모델과 비교하여 더 높은 data throughput을 갖는다.

* BERT-large를 baseline으로 사용한다면 ALBERT-large는 data를 1.7배 빠르게 iterating 하고 ALBERT-xxlarge는 큰 구조 때문에 3배 느리다.

## Factorized Embedding Parameterization

* non-shared condition은(BERT-style) larger embedding size가 더 좋은 성능을 보였지만 많이는 아니었다.

* all-shaered condition은(ALBERT-style) embedding size 128이 가장 좋은 성능을 보였다.

## Cross-Layer Parameter Sharing

* 우리는 all-shared strategy (ALBERT-style), not-shared strategy (BERT-style), attention parameters만 공유하거나 FFN parameters만 공유하는 intermediate strategies를 비교한다.

* all-shared strategy는 (E = 768, E = 128) 일때 모두 성능이 좋지 않았고, E = 128인 경우가 E = 768인 경우보다 덜 성능이 좋지 않았다.

* 대부분 성능의 하락은 FFN-layer parameters를 공유하는 것으로 부터 나타났고, attention parameters를 공유하는 것은 하락이 없었다. E = 128인 경우는 평균적으로 +0.1이었으며, E = 768일 경우는 평균적으로 -0.7의 감소를 보였다. 층간 parameters를 공유하는 다른 전략이 있다. 예를들면, L layers를 N groups of size M으로 나누고 each size-M group은 parameters를 공유한다. 우리의 실험은 전체적으로 group size M이 작을 수록 성능이 더 좋았다. 하지만 group size M을 줄이는 것은 전체적인 parameters의 수를 증가시켰다. 우리는 all-shared strategy를 기본 선택 값으로 골랐다.

## Sentence Order Prediction (SOP)

* intrinsic tasks의 결과들은 NSP는 SOP task의 discriminative한 power를 가져오지 못하는 것을 밝혀냈다.

* 이는 NSP는 오직 topinc shift만 모델링하는 것으로 결론 맺도록 해 준다.

* 대조적으로 SOP loss는 NSP task를 비교적 잘 해결하며 SOP task도 잘 해결한다.

* SOP loss는 일관적으로 multi-sentence encoding tasks와 같은 downstream task의 성능을 평균적으로 +1% 향상시키는 것으로 나타난다. (+1% for SQuAD1.1, +2% for SQuAD2.0, +1.7% for RACE)

## What If We Train For The Same Amount Of Time

* 긴 훈련 시간이 더 좋은 성능으로 이끌기 때문에, data throughput을 controlling 하는 것 대신 actual training time을 control하는 비교를 수행했다.

* ALBERT-xxlarge model을 125k training steps를 훈련할 때 필요한 시간과 동일한 400k training steps를 훈련한 BERT-large 모델의 성능을 비교했다.

* 같은 시간을 훈련한 후 ALBERT-xxlarge는 평균적으로 BERT-large보다 +1.5% 향상을 보였고, RACE는 +5.2% 만큼의 차이가 있었다.

## Additional Training Data And Dropout Effects

* 지금까지 수행된 실험들은 Wikipedia와 BookCourpus datasets를 사용한다. 이번 섹션은 XLNet과 RoBERTa에서 사용된 추가적인 data의 영향을 측정한다.

* 1M steps를 훈련한 후 가장 큰 모델은 training data로 과적합 되지 않았고, 우리는 model capacity를 증가시키기 위해 dropout을 제거하기로 결정했다.

* ALBERT-xxlarge의 1M training steps 후 intermediate evaluation은 dropout을 제거하는 것이 downstream tasks를 돕는다는 것을 확인했다. 우리가 아는한에 large Transformer-based models에서 dropout이 성능을 좋지않게 할 수 있는 것을 보여주는 첫번째이다.

* 하지만 ALBERT를 기초로 하는 network 구조는 transformer의 특별한 구조이고, 더 나아가 다른 모델들도 이러한 현상이 일어나는지 확인하기 위해 실험이 필요하다.

## Current State-Of-The-Art On NLU Tasks

* 이번 섹션에서 우리가 보고한 결론은 Liu et al. (2019)와 Yang et al. (2019) 뿐만 아니라 Devlin et al. (2019)에서 사용한 training data를 사용했다. fine-tuning을 위한 single-model,ensembles인 두 개의 settings 아래의 SOTA 결과들을 보고한다. 모든 settings로 우리는 single-task-fine-tuning을 했다. Liu et al. (2019)을 따르면서 development set에서 우리는 five runs넘게 median result를 보고한다.

* GLUE와 RACE benchmarks를 위해 우리는 12-layer와 24-layer 구조를 사용하여 다양한 training steps로부터 fine-tuned된 후보들을 평균을내어 예측을 진행하였다. SQuAD를 위해서는 우리는 multiple probabilities를 가지는 spans를 위한 prediction scores를 평균을 냈다. 우리는 또한 unanswerable decision의 scores를 평균을 냈다.

* single-model과 ensemble 결과들은 ALBERT가 GEUE score는 89.4, SQuAD 2.0 test F1 score는 92.2, RACE test accuracy는 89.4를 성취하면서 모든 세 개의 benchmarks에서 상당한 SOTA로 향상시키는 것을 보여주었다. 후자는 상당히 강한 향상을 보였으며, BERT보다는 +17.4%를 뛰어넘었고, XLNet보다는 +7.6%를 뛰어넘었고, RoBERTa보다는 +6.2%를 뛰어넘었고, reading comprehension tasks를 위해 설계된 multiple models의 ensemble인 DCMI+를 5.3%를 뛰어넘었다.

* 우리의 single model은 ensemble model의 SOTA를 2.4%의 성능을 향상시키면서 정확성 86.5%를 성취했다.

# Discussion

* ALBERT-xxlarge는 BERT-large보다 더 적은 파라미터들을 가지고 있고 상당히 성능도 좋지만, larger 구조 때문에 계산적으로 더욱 비싼 연산이다. 중요한 다음 step은 sparse attention과 block attention과 같은 방법들을 통해 ALBERT의 학습과 추론 속도를 향상시키는 것이다. 추가적으로 우리는 sentence order prediction이 더 나은 language representations을 이끄는 유용한 learning task라는 것을 설명해왔고, 우리는 resulting representations를 위한 추가적인 representation power를 생성할 수 있는 현재 self-supervised training losses에 의해 포착되지 않은 더 많은 차원들이 있다고 가정한다.
