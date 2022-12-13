---
title:  "[논문 리뷰] Bert"
excerpt: "Bert 논문을 읽어보자!!"

categories:
  - NLP
tags:
  - [DL]

toc: true
toc_sticky: true
---

# Abstract

* 트랜스포머로부터 양방향 인코더 표현을 특징으로하는 새로운 언어모델인 BERT를 소개한다.

* BERT는 다른 언어표현 모델과는 다르게 양쪽 문맥을 참고하여 사전학습하도록 설계되었으며 그 결과로 많은 task들에서 fine-tuning을 통해서 SOTA를 달성할 수 있었다.

* BERT는 개념적으로 그리고 경험적으로도 강력했는데, 11개의 natural language processing tasks에서 기존의 score를 넘으면서 SOTA를 달성했다.

# Introduction

* 사전학습된 언어모델은 여러 NLP task에서 효과적이다.

* 사전학습된 언어 표현을 down-strean task에 적용하는 방법은 feature-based와 fine-tuning으로 나뉜다. feature-based 방법은 사전학습된 표현을 task-specific 구조에 추가해주는 방법으로 ELMo가 있고 fine-tuning은 최소한의 task-specific 파라미터를 사용하고 모든 파라미터가 downstream task를 학습하면서 훈련이 된다. 두 방법 모두 같은 objective function을 가지지만 사전학습을 진행할 때 단방향 모델을 통하여 언어 표현을 학습한다.

* 단방향 모델을 사용하여 사전학습을 진행하면 모델 구조의 선택을 제한하게 된다. 예를들어 GPT는 왼쪽-오른쪽 구조를 사용하는데 이는 매 단어들이 오직 이전 단어들만 참고할 수 있다. 이는 sentence-level tasks에서는 차선책이며 token-level tasks에서는 양쪽 문맥이 모두 중요하기 떄문에 좋지 않다.

* 이러한 단방향 모델들의 제약을 BERT에서는 MLM을 사용하여 완화시켰다. MLM은 input token의 일부를 mask해서 원래 단어를 문맥을 반영하여 예측하는 것이 목적이다.

* MLM은 깊은 양방향 문맥을 학습할 수 있고 MLM이외에도 NSP인 다음 문장 예측이라는 task를 적용하였다.

# Related Work

* 언어표현을 사전학습하는 것은 긴 역사가 있다. 주로 많이 사용되는 접근법에 대해서 간략하게 설명을 하겠다.

## Unsupervised Feature-based Approaches

* 사전학습을 통한 단어 임베딩은 NLP 시스템에서 처음부터 학습하는 임베딩을 넘어서 상당한 향상을 제공하므로 중요한 부분이다.

* ELMo는 left-right, right-left 모델로 부터 단어의 특징을 추출했고, 각 token의 문맥적 표현은 left-right, right-left 모델로 부터 추출된 표현을 concatenation을 한다. 이 문맥 표현이 담긴 단어 임베딩을 존재하는 task-specific 구조와 통합해서 ELMo는 여러 주요한 NLP benchmarks에서 SOTA를 달성할 수 있었다.

## Unsupervised Fine-tuning Approaches

* 최근에 문맥적 token 표현을 생성하는 sentence or document encoders는 unlabeled text로부터 사전학습이 되었고, downstream task에 fine-tuning 되었디. 이러한 접근법의 장점은 처음부터 학습되는 파라미터들이 적어진다. 이러한 장점때문에 GPT는 GLUE 벤치마크로부터 많은 sentence-level tasks에서 SOTA를 달성했다.

## Transfer Learning from Supervised Data

* large datasets를 가지고 있는 tasks들로부터 효율적인 transfer-learning을 보여주는 연구도 있었다.

* computer vision 연구는 ImageNet으로 사전학습된 모델을 fine-tune하는 효율적인 recipe인 거대한 사전학습된 모델로부터 transfer-learning의 중요성을 증명했다.

# BERT

* BERT는 pre-training과 fine-tuning으로 2단계가 있다. pre-training 하는 동안은 모델은 다양한 pre-training tasks로부터 unlabeled된 data로 학습을 하고, fine-tuning을 하는 동안은 BERT 모델은 사전 학습된 파라미터들로 초기화 되고 모든 파라미터들은 downstream tasks로 부터 labeled data를 사용하여 fine-tuned된다.

* 사전 학습된 구조와 downstream 구조간에는 약간의 차이만 있다.

## Model Architecture

* BERT는 양방향 Transformer의 인코더를 여러개 쌓은 구조이다.

* L을 the number of layers, H를 the hidden size, A를 the number of self-attention heads로 정의한다.

* model size에 따라 BERT_BASE, BERT_LARGE로 나누어진다. BERT_BASE는 GPT와 비교를 목적으로 같은 model size로 선택되었다. BERT는 양방향 self-attention을 사용하는 반면 GPT는 every token이 왼쪽 문맥만 참고할 수 있는 제한된 self-attention을 사용한다.

## Input/Ouput Representations

* BERT를 다양한 down-stream tasks에서 다루기 위해서 한 token sequence에서 단일 문장과 문장 쌍을 분명하게 표한한다.

* WordPiece embeddings를 사용하여 30000개의 token vocabulary를 가지고 있으며 every sequence의 첫번 째 token은 항상 [CLS] token을 갖는다. 이 token과 대응하는 final hidden state는 분류 tasks들의 aggregate sequence representation으로 사용이 된다.

* 문장 쌍은 single sequence로 같이 들어가는데 문장을 두 가지 방법으로 구분했다. 첫번 째는 [SEP] token으로 구분했고, 두번 째는 sentence A에 속하는지 sentence B에 속하는지 알려주면서 every token과 학습되는 embedding을 추가했다.

* E를 input embedding, C를 [CLS] token의 final hidden vector, T_i를 i번째 input token의 final hidden vector로 정의한다.

* token이 주어졌을 때 input 표현은 token, segment, position embeddings를 더하면서 구성한다.

## Pre-training BERT

### Task #1: Masked LM

* 표준 언어 모델은 양방향으로 조절하는 것이 각 단어들이 모든 단어들을 참고할 수 있도록 하기 때문에 오직 left-to-right, right-to-left로 학습이 된다.

* 깊은 양방향 표현을 학습하기 위해서 무작위로 input tokens의 일부 percentage를 mask하고 masked된 tokens를 예측한다. 이러한 경우에 mask tokens와 대응하는 final hidden vectors는 output softamx로 들어가게 된다. each sequence에서 무작위로 모든 WordPiece tokens의 15%를 mask한다.
denoising auto-encoders와 대조되게 전체 input을 재구성하는 것이 아니라 masked된 단어들을 예측한다.

* 이는 양방향 사전 학습된 모델을 얻을 수 있지만 fine-tuning 동안 [MASK] token이 나타나지 않기 때문에 pre-training과 fine-tuning간에 mismatch가 일어나게 되는 단점이 있다. 이를 완화하기 위해서 [MASK] token을 항상 "masked"로 대체하지 않는다. 예측을 위해서 무작위로 token positions의 15%를 선택한다. i번째 token이 선택되면 80%를 [MASK]로 대체하고 10%를 다른 token으로 바꾸고, 나머지 10%를 바꾸지 않고 그대로 둔다.

### Task #2: Next Sentence Prediction (NSP)

* Question Answering, Natural Language Inference와 같은 downstream tasks들은 언어 모델로 포착되지 않는 두 문장 사이의 관계를 이해하는 데 기초가 된다. 문장 관계를 이해하는 모델을 학습하기 위해서는 binarized next sentence prediction task를 사전학습한다. 각각 pre-trainig example을 위해 문장 A와 문장 B를 고를 때 50%는 sentences B는 sentences A가 나온 다음에 이어지고, 50%는 corpus에서 이어지지 않는 무작위의 문장이다. C는 next sentence prediction을 위해서 사용되고, 단순함에도 불구하고 이 task를 사전학습하는 것은 QA와 NLI에서 좋은 성능을 보여주는 것을 증명했다.

* NSP task는 Jernite et al. (2017), Logeswaran and Lee (2018)에서 사용된 representation learning objectives와 관련성이 있다. 하지만 이전 논문들은 sentence embeddings를 down-stream tasks로 전이학습하지만 BERT는 end-task model parameters를 초기화 하는데 모든 parameters를 전이학습한다.

### Pre-training data

* pre-training을 하는데 BooksCorpus와 English Wikipedia를 사용한다. Wikipedia에서는 오직 text passages, lists, tables, headers를 추출한다.

## Fine-tuning BERT

* fine-tuning은 Transformer의 self-attention이 적절한 inputs과 outputs를 바꾸면서 single text 혹은 text pairs를 포함하는지 안 하는지 많은 downstream tasks를 모델링 하도록 하기 때문에 단순하다.

* 문장 쌍을 포함하는 applications에서는 공통적인 패턴은  Parikh et al. (2016)와  Seo et al. (2017).와 같은 bidirectional cross attention을 적용하기 전에 독립적으로 text pairs를 encode한다.  BERT는 대신에 두 문장 사이에서 bidirectional cross attention을 효율적으로 포함하는 self-attention과 함께 concatenated text pair를 인코딩 하면서 두 단계를 통합하는 self-attention 매커니즘을 사용한다.

* 각 task를 수행할 때 단순히 task-specific inputs과 outputs를 BERT에 입력하고 처음부터 끝까지 모델의 모든 파라미터들을 finetune한다.

* pre-training과 비교하여 fine-tuning은 상대적으로 inexpensive하다. TPU로 1시간, GPU로는 몇 시간안으로 정확히 같은 사전 학습된 모델을 사용하면 paper의 모든 결론을 동일하게 얻을 수 있다.

# Experiments

## GLUE

* fine-tuning 동안 도입된 새로운 parameters는 classification layer 가중치인 W이며 크기는 K * H 가지며 K는 label의 수이다. C와 W로 표준 classification loss를 계산한다. 모든 GLUE tasks들을 위해 batch size는 32로 하고 3epochs를 사용한다. 가장 좋은 fine-tuning 학습률을 5e-5, 4e-5, 3e-5, 2e-5중에서 선택했다.

* BERT_BASE, BERT_LARGE는 이전 SOTA 모델들보다 각각 평균적으로 4.5%에서 7.0% 정확도 향상을 얻으면서 상당한 margin에 의해 모든 tasks들보다 성능이 앞섰다.

* BERT_LARGE는 적은 데이터의 모든 tasks들을 넘어서 BERT_BASE보다 성능이 좋았다.

## SQuAD v1.1

* 표준 Question Answering Dataset은 100k의 crowd-sourced된 question/answer pairs의 집합이다.

* best performing system은 ensembling에서 top leaderboard system보다 1.5 F1이 증가했고, single system은 1.3 F1이 증가했다.

* SQuAD를 fine-tuning 하기 전에 TriviaQA를 먼저 fine-tuning하면서 최신의 data augmentation을 사용한다. single BERT 모델은 F1 score 관점에서 top ensemble system보다 성능이 좋았다. TriviaQA fine-tuning data없이는 오직 0.1-0.4 F1이 낮았지만 여전히 넓은 margin으로 모든 존재하는 systems보다 성능을 앞질렀다.

## SQuAD v2.0

* SQuAD 2.0 task는 problem을 더욱 현실적으로 만들면서 제공된 paragraph를 더 짤은 answer이 없는 가능성을 허용하면서 SQuAD 1.1 problem을 확장한다.

* 이 task를 위해 SQuAD v1.1 BERT model을 확장하기 위해서 단순한 접근법을 사용한다.

* 2 epochs, learning rate 5e-5, batch size 48을 적용하여 fine-tuned 했다. 이전 best system을 +5.1 F1만큼 성능이 좋았다.

## SWAG

* 3 epochs, learning rate 2e-5, batch size 16으로 fine-tune 했다.

* BERT_LARGE 모델은 저자의 baseline인 ESIM+ELMO system보다 27.1% 성능이 좋았고, GPT보다는 8.3% 성능이 좋았다.

# Ablation Studies

* BERT의 많은 측면들의 상대적인 중요성을 더욱 잘 이해하기 위해서 ablation experiments를 수행했다.

## Effect of Pre-training Tasks

* BERT_BASE의 정확히 같은 pre-training data, fine-tuning scheme, hyperparameters 사용하는 두개의 pre-training objectives를 평가하면서 깊은 양방향 BERT의 중요성을 증명한다.

### No NSP

* 양방향 모델이 NSP task없이 MLM을 사용하면서 학습된다.

### LTR & No NSP

* left-context only model은 MLM 대신에 표준 Left-to-Right를 사용하여 학습이 된다. 왼쪽만 참고하는 제약은 pre-train과 fine-tune의 mismatch는 downstream의 성능 저하를 도입하기 때문에 fine-tuning에도 적용이 된다,

* 추가적으로 이 모델은 NSP task 없이 pre-trained 되었다. GPT와 비교하여 larget training dataset, input representation, fine-tuning scheme를 사용하였다.

* NSP를 제거하는 것은 QNLI, MNLI, SQuAD 1.1 task의 성능을 좋지 않게 한다. 다음은 NO NSP와 LTR & NO NSP를 비교하면서 양방향 표현의 형향을 평가했다. LTR 모델은 모든 tasks에서 성능이 MLM보다 좋지 않았으며, MRPC, SQuAD에서는 더 큰 폭으로 좋지 않았다.

* SQuAD task에서 LTR 모델은 token-level의 오른쪽 hidden states가 없으므로 token predictions에서 성능이 좋지 않은 것은 직관적으로 명백하다. LTR system을 개선하기 위해서 랜덤하게 초기화된 BiLSTM을 top에 추가했다. 이는 SQuAD에서 상당한 향상을 보였지만 양방향 모델로 사전학습된 모델들보다 성능은 좋지 않았다. BiLSTM은 GLUE tasks에서 좋은 성능을 내지 못했다.

*  ELMo처럼 LTR 모델과 RTL 모델을 학습시키고 두 모델을 concatenation해서 각각 token을 표한할 수 있다고 인식했다. 하지만 single bidirectional model보다 비용이 2배 들었고, RTL 모델은 질문에 맞는 대답을 조절할 수 없기 때문에 QA와 같은 tasks에는 직관적이지 않다. 이러한 모델은 every layer에서 left, right 문맥을 둘다 사용하기 때문에 deep bidirectional model보다 강력하지 않다.

## Effect of Model Size

* model size가 fine-tuning task 정확도에 미치는 영향을 탐구한다. 같은 hyperparameters와 이전에 설명된 동일한 훈련 절차를 사용하면서 layers, hidden units, attention heads의 수를 다양하게 하면서 많은 BERT 모델을 훈련시켰다.

* 더 큰 모델은 MRPC는 3600개의 labeled training examples만 있었고 pre-training tasks가 달랐지만 4개의 datasets 전역에서 정확도 향상이 있었다.

* Table 6에서 보여준 LM perplexity of held-out training data로 증명된 machine translation, language modeling과 같은 large-scale tasks들에서 model size가 커지면 성능 향상이 있다는 것은 잘 알려진 사실이다. 하지만 충분히 사전학습이 잘 된 모델이 주어지면 very small scale tasks에서도 성능 향상을 이끌 수 있다고 믿는 첫번 째 작업이라고 믿는다.  Peters et al. (2018b)는 2층부터 4층까지 사전학습된 bi-LM의 size를 증가시키는 것이 downstream으로 미치는 영향에 대한 mixed 결론을 제시했고  Melamud et al. (2016)은 200부터 600까지 hidden dimenstion size를 증가시키는 것은 도움이 되었지만 1000가지는 더욱 향상을 가져오지는 않는다고 언급했다. 이들 모두는 feature-based approach를 사용했다. 모델이 downstream tasks로 직접적으로 fine-tuned되고 오직 랜덤하게 초기화된 추가된 parameters의 적은 수만 사용할 때 task-specific models는 downstream task data가 매우 적을 때도 더 크고, 더욱 효율적인 사전학습된 표현들로 성능 향상을 얻을 수가 있다. 

## Feature-based Approach with BERT

* 사전학습된 모델로부터 추출된 fixed features를 사용하는 feature-based approach는 장점을 갖는다. 첫번 째 모든 tasks들은 Transformer의 인코더 구조로 쉽게 표현될 수 없다 그러므로 추가되는 task-specific model 구조가 요구된다. 두번 째는 expensive representation을 pre-compute 하기 위한 컴퓨팅 연산의 장점과 top에 대한 더욱 비용 연산이 저렴한 모델들과 많은 실험을 할 수 있다는 장점이 있다.

* 두 접근법을 NER task를 적용하여 비교했다. BERT의 어떠한 parameters들의 fine-tuning 없이 하나 이상의 층으로부터 activations를 추출하면서 feature-based approach를 적용했다. 이 contextual embeddings는 classification layer전 랜덤하게 초기화된 2층의 768차원을 갖는 BiLSTM의 입력으로 사용되었다.

* 사전학습된 Transformer의 맨 위의 4개의 hidden layers로 부터 얻어진 token representations를 concatenates하는 것이 가장 좋은 성능을 보였던 방법이며, 전체 모델을 fine-tuning 했을 때 보다 오직 0.3 F1만 뒤쳐졌다. 이는 BERT가 fine-tuning, feature-based approaches 모두 효율적이라는 것을 증명한다.

# Conclusion

* 풍부하고 비지도 사전학습이 많은 language understanding systems에서 중요한 부분이며 이는 low-resorce tasks에서도 양방향 구조의 장점을 얻을 수 있다.

* 주요한 기여는 사전학습된 모델이 넓은 NLP tasks들에서 일반화 했다는 점이다.