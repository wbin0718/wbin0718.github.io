---
title:  "[논문 리뷰] RoBERTa"
excerpt: "RoBERTa 논문을 읽어보자!!"

categories:
  - NLP
tags:
  - [DL]

toc: true
toc_sticky: true
---

# Abstract

# Introduction

* 우리는 hyperparameter tuning과 training set size의 효과의 평가를 포함하는 BERT pretraining의 연구를 제시한다.

* BERT는 과소훈련이 되었고, BERT를 학습시키기 위해 향상된 레시피인 RoBERTa를 제안한다.

* RoBERTa는 첫번 째로 많은 데이터는 더 큰 batch로 학습을 오래 시키고 두번 째는 NSP를 제거하고, 세번 째는 더 긴 sequences를 학습시키고, 역동적으로 training data로 적용되는 masking pattern을 변화시킨다. 또한 더 큰 dataset을 수집하고, training set size effects를 더 성능이 좋게 통제한다.

* 요약하면 첫번 째 우리는 중요한 BERT design choices와 training strategies를 제시하고, downstream task의 성능을 향상시킬 수 있는 대안책을 소개한다. 두번 째 우리는 새로운 dataset인 CC-NEWS를 사용하고 pretraining을 할 때 더 많은 데이터를 사용하는 것은 downstream tasks의 성능을 향상시킬 수 있는 것을 확인했다. 세번 째는 우리의 training improvements는 올바른 모델 설계를 했을 때 masked language model pretraining이 최근에 게시된 방법들과 경쟁력을 갖추는 것을 보여주었다.

# Background

* 실험적으로 조사할 BERT의 pretraining approach와 training choices의 일부를 간략하게 설명한다.

## Setup

* BERT는 x1, . . . , xN, y1, . . . , yM 두개의 segments를 연결한 input을 받아들인다. M과 N은 M + N < T라는 제약이 있고 T는 학습하는 동안 최대 sequence length를 통제하는 파라미터이다.

* 모델은 많은 양의 unlabeled text corpus로 먼저 사전학습이 되고, end-task labeled data를 사용하여 finetuned 된다.

## Architecture

* 우리는 L layers의 transformer 구조를 사용한다. 각 블록은 A self-attention heads와 hidden dimension H를 사용한다.

## Training Objectives

* 학습하는 동안 BERT는 masked language modeling과 next sentence prediction을 사용한다.

### Masked Language Model (MLM)

* MLM objective는 masekd tokens를 예측할 때 cross-entropy loss를 사용한다. BERT possible replacement를 위해 input tokens의 15%를 무작위로 선택한다. 선택된 tokens 중 80%는 [MASK]로 대체되고, 10%는 변화하지 않고 그대로 유지되며 10%는 무작위로 선택된 token으로 대체된다.

### Next Sentence Prediction (NSP)

* NSP는 두개의 segments가 원래 같은 text로 포함되어 있었는지 예측하는 binary classification loss이다.

* Positive examples는 text corpus로부터 연속적인 sentences를 가져오면서 생성된다. Negative examples는 다른 문서로부터 segments의 쌍을 이루면서 생성된다. Positive와 negative examples는 동일한 확률로 추출된다.

* NSP objective는 문장 쌍의 관계를 추론하는 것을 요구하는 Natural Language Inference와 같은 downstream tasks의 성능을 향상시키기 위해 설계되었다.

## Optimization

* BERT는 B_1 = 0.9, B_2 = 0.999, ǫ = 1e-6 , L_2 weight decay 0.01인 Adam optimizer를 사용했다.

* learning rate는 10000 steps까지 warmed up되고 1e-4까지 상승했다가 선형적으로 감소된다.

* BERT는 all layers, attention weights, GELU activation function를 dropout 0.1로 훈련시켰다.

* 모델들은 S = 1000000 updates를 사전훈련했고, T = 512 tokens의 sequences를 B = 256의 개수를 포함하는 mini batches를 사용한다.

## Data

* BERT는 전체의 16GB의 압축되지 않은 text인 BOOKCORPUS와 English WIKIPEDIA를 사용하여 훈련했다.

# Experimental Setup

* 우리는 replication study of BERT의 experimental setup을 소개한다.

## Implementation

* 우리는 FAIRSEQ로 BERT를 재실행했다. 우리는 각 setting을 위해 분리하여 tuned된 peak learning rate와 number of warmup steps를 제외하고 동일한 BERT의 optimization hyperparameters를 따랐다. 우리는 추가적으로 Adam epsilon term으로 인해 학습이 예민해질 수 있는 것을 발견했고, 그것을 tuning 하면서 안정성을 향상시켰다. 또한 큰 batch sizes를 사용하여 훈련할 때 setting B_2 = 0.98이 안정성을 향상시키는 것을 발견했다.

* 우리는 무작위로 short sequences를 주입하지 않았고, 첫번 째 90%의 updates는 감소된 sequence length로 훈련을 하지 않았다. 오로지 full-length sequences를 훈련했다.

## Data

* BERT-style pretraining은 큰 quantities of text를 의존한다. Baevski et al. (2019)은 data size가 증가할수록 향상된 end-task performance를 얻을 수 있는 것을 보여주었다.

* 모든 추가적인 datasets가 공식적으로 공개가 되지 않는다. 우리의 연구를 위해 우리는 각각 비교를 위한 적절하면서 전체적으로 품질과 양을 일치시키는 것을 허용하면서 실험을 위한 가능한 많은 데이터를 모으는데 집중했다.

* 우리는 5개의 전체적으로 160GB의 압축되지 않은 text를 sizes와 domains를 다양하게 하면서 English-language corpora를 고려했다.

* BOOKCORPUS, English WIKIPEDIA를 사용했다.

* CommonCrawl News dataset의 English 부분으로 부터 수집된 CC-NEWS를 사용했다. CC-NEWS는 September 2016 부터 February 2019 기간으로 부터 crawled 된 63 million English news articles를 포함한다.

* Radford et al. (2019)가 설명한 WebText corpora의 open-source recreation인 OPENWEBTEXT을 사용했다. text는 적어도 3 upvotes인 Reddit을 공유한 URLs로 부터 추출된 web contenct이다.

* Trinh and Le (2018)이 소개하고 Winograd schemas의 story-like style을 일치하도록 filtered된 CommonCrawl data의 subset을 포함하는 dataset인 STORIES를 사용했다.

## Evaluation

* 이전 작품을 따르면서, 우리는 3개의 benchmarks를 사용하여 downstream tasks로 pretrained models를 평가한다.

### GLUE

* The General Language Understanding Evaluation (GLUE) benchmark는 natural language understanding systems를 평가하는 9개의 datasets의 집합이다. Tasks들은 single-sentence classification 이나 sentence-pair classification tasks로 구성된다.

* GLUE organizers는 private held-out test data의 systems를 참여자들이 평가하고 비교할 수 있도록 하면서 submission server와 leaderboard 뿐만 아니라 training and development data splits를 제공한다.

* Section 4에서 replication study를 위해 우리는 single task training data와 상응하는 pretrained models를 finetuning하고 development sets의 결과를 보고한다.

* Section 5에서 publi leaderboard 로부터 얻어진 test set의 결과를 우리는 추가적으로 보고한다.

* 이러한 결과들은 Section 5.1에서 설명한 다양한 task-specific modifications를 의존한다.

## SQuAD

* The Stanford Question Answering Dataset (SQuAD)는 context의 paragraph와 question을 제공한다. task는 context로부터 관련있는 부분을 추출하여 질문을 대답하는 것이다.

* 우리는 SQuAD의 V1.1과 V2.0 두 가지 versions를 평가했다. V1.1은 context가 항상 대답을 포함하고 있는 반면 V2.0은 task를 더욱 어렵게 만들면서 일부 질문이 제공된 context로 대답을 할 수 없다.

* SQuAD V1.1을 위해 우리는 BERT와 같은 span prediction method를 채택했다. SQuAD V2.0을 위해  질문이 대답이 가능한지 예측을 하고 classification과 span loss terms를 더하면서 학습을 하는 binary classifier을 추가했다. 평가하는 동안 우리는 대답가능하다고 분류된 쌍들의 span indices를 예측한다.

## RACE

* ReAding Comprehension from Examinations (RACE) task는 28000개 이상의 passages와 거의 100000 질문을 가지고 있는 large-scale reading comprehension이다.

* dataset은 중국의 중학교, 고등학교 학생들을 위해 설계된 영어 시험으로 부터 수집되었다.

* 각각 passage는 multiple questions와 관련되어있다. 매 질문을 위해 task는 4개의 선택지로부터 correct answer를 하나 고른다. RACE는 상당히 다른 인기있는 reading comprehension datasets 보다 긴 context와 추론을 많이 요구하는 질문들의 비율을 가지고 있다.

## Training Procedure Analysis

* 이번 섹션은 BERT를 성공적으로 pretraining 하기 위해 선택들이 중요하다는 것을 탐구하고 정량화한다.

* 우리는 BERT models를 BERT_BASE (L = 12, H = 768, A = 12110M params)와 같은 configuration으로 BERT models를 훈련시키면서 시작한다.

### Static vs. Dynamic Masking

* original BERT implementation은 single static mask를 하면서 data preprocessing을 하는 동안 masking을 한번 수행했다. 매 epoch마다 instance를 training을 위해 같은 mask를 사용하는 것을 피하기 위해 training data는 10번 복제되었고, 이로 인해 각 sequence는 training의 40 epochs 중 10번이 다양한 방식으로 masked된다. 그래서 각각 training sequence는 훈련하는 동안 4번정도가 같은 mask로 보여진다.

* 우리는 모델로 sequence를 줄 때 매번 masking pattern을 바꾸는 dynamic masking 전략으로 비교한다. 이는 더 많은 steps와 더 큰 datasets으로 pretraining 할 때 중요하다.

### Results

* 우리는 static masking으로 reimplementation은 original BERT model과 빗스한 성능을 보이는 것을 알아냈고, dynamic masking은 비교할 만 하거나 근소하게 static masking보다 성능이 뛰어나디.

* 이러한 결과들과 추가적인 dynamic masking의 효율성이라는 장점이 주어졌을 때 우리는 dynamic masking을 실험의 remainder로 사용한다.

## Model Input Format and Next Sentence Prediction

* original BERT pretraining 절차는 model이 같은 documents나 다른 documents로 부터 sampled된 두 개의 연결된 document segments를 준수한다. masked language modeling objective 말고도 모델은 auxiliary Next Sentence Prediction (NSP) loss를 통하여 같거나 다른 documents로 부터 관측된 document segments인지 예측하면서 학습이 된다.

* NSP loss는 original BERT model을 학습할 때 중요한 요소라고 여겨진다. Devlin et al. (2019)는 NSP를 제거하는 것은 QNLI, MNLI, SQuAD 1.1의 상당한 성능 저하와 함께 성능을 떨어뜨린다고 관측했다. 하지만 몇몇 최근 연구는 NSP loss의 필요성의 대한 질문을 하고있다.

* 이러한 불일치를 더 이해하기 위해 우리는 몇가지의 대안의 training formats를 비교한다.

* SEGMENT-PAIR+NSP: 이는 NSP loss와 함께 BERT가 사용한 original input format을 따른다. 각각의 input은 multiple natural sentences를 포함할 수 있지만 전체 결합된 길이는 512 tokens보다 작아야 하는 한 쌍의 segments를 가지고 있다.

* SENTENCE-PAIR+NSP: 각각의 input은 하나의 document의 일부로 부터 sampled 되거나 별도의 documents로부터 sampled된 한쌍의 natural sentences를 포함한다. 이 inputs는 상당히 512 tokens보다 작아서 우리는 tokens의 전체 수를 SEGMENT-PAIR+NSP와 비슷한 상태로 만들기 위해 batch size를 증가시켰다. 우리는 NSP loss를 재학습했다.

* FULL-SENTENCES: 각각의 input은 하나 이상의 documents로 부터 sampled된 full sentences로 꾸려지고 total length는 많아야 512 tokens이다. Inputs는 document 경계를 넘나든다. 하나의 document의 끝으로 도달할 때 다음 document로부터 문장들을 sampling 하고 documents사이는 추가적인 separator token을 추가한다. 우리는 NSP loss를 제거했다.

* DOC-SENTENCES: Inputs는 document 경계를 넘나들 수 없다는 것을 제외하고는 FULL-SENTENCES와 비슷하게 구성되었다. 한 document의 마지막 부분으로부터 sampled된 Inputs는 512 tokens보다 짧아서 우리는 동적으로 FULL-SENTENCES와 total tokens와 비슷한 수를 만들기 위해 이러한 경우는 batch sizse를 증가시켰다. 우리는 NSP loss를 제거했다.

### Results

* 우리는 먼저 BERT의 SEGMENT-PAIR input format을 SENTENCE-PAIR format과 비교했다. 둘다 NSP loss를 포함하지만 후자는 single sentences를 사용한다. 우리는 individual한 sentences를 사용하는 것은 모델이 long-range dependencies를 학습할 수 없으므로 downstream tasks의 성능을 좋지 않게 한다는 것을 알아냈다.

* 우리는 NSP loss 없이 훈련하는 것과 single document (DOC-SENTENCES)로 부터 text의 blocks로 훈련하는 것을 비교했다. 우리는 이런 setting이 BERT_BASE 결과들의 성능을 앞지른다는 것과 Devlin et al. (2019) 와는 대조적으로 NSP loss를 제거하는 것이 downstream task의 성능과 일치하거나 약간 향상시킨다는 것을 발견했다. original BERT implementation은 SEGMENT-PAIR input format을 보유하지만 loss term을 제거했을지도 모르는 것이 가능하다.

* 최종적으로 우리는 sequences가 single document로 부터 오는 것을 제한하는 것은 multiple documents (FULL-SENTENCES)로 부터 sequences를 꾸리는 것보다 성능이 조금 더 좋다는 것을 발견했다. 하지만 DOC-SENTENCES format은 가변적인 batch sizes를 사용하므로 우리는 관련된 work와 더욱 쉬운 비교를 위해 우리의 실험의 remainder로 FULL-SENTENCES를 사용한다.

## Training with large batches

* 최근 연구는 BERT도 큰 batch training을 받을 수 있음을 보여주었다. (You et al., 2019)

* Devlin et al. (2019)는 256 sequences의 batch size로 1M steps BERT_BASE를 훈련시켰다. 이는 2K sequences의 batch size로 125K steps 혹은 8K의 batchsize로 31K steps를 훈련할 때 computational cost, gradient accumulation가 동일하다.

* 우리는 큰 batches로 training 하는 것은 end-task accuracy 뿐만 아니라, perplexity의 성능을 향상시키는 것을 관측했다. Large batches는 분산된 data paraller training을 통하여 병렬화 하기가 쉽고,나중 실험으로 우리는 8k sequences의 batches로 훈련시켰다.

* You et al. (2019)는 32K sequences까지 batch sizes를 키워서 BERT를 훈련시켰다. 우리는 나중 작품으로 큰 batch training의 제한의 탐구를 남길 것이다.

## Text Encoding

* Radford et al. (2019)는 base subword units으로 unicode chracters를 사용하는 것 대신 bytes를 사용하는 clever한 BPE implementation을 소개했다. bytes를 사용하는 것은 어떠한 "unknown" tokens를 도입하는 것 없이 어떠한 input을 encode할 수 있는 modest size (50K units)의 subword vocabulary를 학습하도록 해 준다.

* original BERT implementation은 heuristic tokenization rules로 input을 전처리하고 학습이 되는 size 30K의  character-level BPE vocabulary를 사용한다. Radford et al. (2019)를 따르면 우리는 대신 추가적인 전처리와 input의 tokenization 없이 50K sub word units를 포함하는 더 큰 byte-level BPE vocabulary로 BERT를 훈련하는 것을 고려한다. 이는 각각 BERT_BASE, BERT_LARGE보다 대략 15M, 20M의 추가적인 파라미터들이 추가된다.

* 초기 실험들은 Radford et al. (2019)의 BPE가 some tasks의 end-task의 성능이 약간 더 좋지 않으면서 이러한 encodings간 근소한 차이만 드러났다. 그럼에도 불구하고 우리는 universal encoding scheme의 장점이 약간의 성능하락보다 더 중요하다고 믿으며 우리의 실험의 remainder로 이 encoding을 사용한다.

# RoBERTa

* RoBERTa는 Robustly optimized BERT approach라고 부른다.

* 구제적으로 RoBERTa는 dynamic masking, FULL-SENTENCES without NSP loss, large mini-batches, larger bytle-level BPE로 학습되었다.

* 추가적으로 우리는 이전 work가 과소 강조한 두 가지의 다른 중요한 요소를 조사했다. 첫번 째는 pretraining을 위해 사용된 data이고 두번 째는 data를 통한 training passes의 수이다. 예를들면 최근에 제안된 XLNet 구조는 original BERT보다 10배 많은 data를 사용하여 pretrained 되었다. 그것은 많은 optimization steps를 반으로 8배 많은 batch size를 사용하고 BERT와 pretraining을 비교했을 때 4배나 많은 sequences를 보이면서 훈련을 했다.

* 다른 modeling 선택으로부터 이러한 요소들의 중요성을 푸는 것을 돕기 위해 BERT_LARGE (L = 24, H = 1024, A = 16, 355M parameters) 구조를 따르는 RoBERTa를 훈련하면서 시작했다. 우리는 Devlin et al. (2019)가 사용했었던 비교가능한 BOOKCORPUS와 WIKIPEDIA datasets를 사용하여 100K steps를 pretrain을 했다.

## Results

* 우리는 RoBERTa가 섹션 4에서 탐구했던 design choices의 중요성을 확고히 하면서 BERT_LARGE 모델의 성능보다 향상을 제공하는 것을 관측했다.

* 우리는 섹션 3.2에서 설명했던 3개의 추가적인 datasets을 위의 data와 결합을 했다. 우리는 RoBERTa를 이전의 training steps과 같은 100K번을 학습시켰다. 전체적으로 160GB의 text로 pretrain을 했다. 우리는 pretraining 할 때 data size와 diversity의 중요성을 평가하면서 모든 downstream tasks의 성능 향상을 관측했다.

* 마침내 우리는 RoBERTa를 the number of pretraining steps를 100K부터 300K까지 더욱이 500K까지 상당히 오랫동안 pretrain을 시켰다. 우리는 downstream tasks의 성능 향상을 관찰할 수 있었고, 300K, 500K의 step models는 대부분 tasks를 XLNet_LARGE보다 성능이 향상되었다.

* 우리는 가장 오랫동안 훈련된 model은 data를 overfit 하지 않아보였고, 추가적인 training으로 부터 성능 향상을 얻을 수 있었다.

* paper의 나머지는 우리는 3개의 benchmarks인 GLUE, SQuaD, RACE를 best RoBERTa 모델로 평가한다. 우리는 섹션 3.2에서 소개한 5개의 datasets로 500K steps 학습된 RoBERTa를 고려한다.

## GLUE Results

* GLUE는 두개의 finetuning settings를 고려한다. 첫번 째 setting은 우리는 일치하는 task의 training data를 사용하여 각각의 GLUE tasks를 위한 RoBERTa를 finetune한다. 우리는 첫번 째 steps의 6% 동안은 linear warmup을 사용하고 linear decay를 0까지 제한된 hyperparameter sweep을 batch sizes는 {16,32}, learning rates {1e-5,2e-5,3e-5}으로 고려한다.

* 두번 째 setting은 우리는 GLUE leaderboard를 통하여 test set으로 다른 접근법들과 RoBERTa를 비교한다. 많은 GLUE leaderboard의 submissions이 multitask finetuning을 의존하지만, 우리의 submission은 오직 single-task finetuning을 의존한다. RTE, STS, MRPC가 우리는 baseline pretrained RoBERTa보다 MNLI single-task model로부터 finetune을 시작하는 것이 도움이된다는 것을 발견했다. 우리는 더 넓은 Appendix로 설명한   hyperparameter space를 탐구하고, 하나의 task를 5개 그리고 7개의 모델들을 ensemble한다.

### Task-specific modifications

* GLUE tasks의 2개는 경쟁적인 leaderboard results를 얻기 위해 task-specific finetuning approaches를 요구한다.

* QNLI : GLUE leaderboard의 최근 submissions는 training set으로부터 mined 되고, 다른 하나와 비교가 되는 QNLI task를 위한 pairwise ranking formulation을 채택하고, single pair은 positive로 분류된다. 이 formulation은 상당히 task를 단순화하지만 BERT와 직접적으로 비교가능하지는 않다. 최근 work를 따르면 우리는 test submission을 위해 ranking approach를 채택하지만 BERT와 직접적인 비교를 위해 우리는 pure classification approach를 기본으로하는 development set results를 보고한다.

* WNLI: 우리는 제공된 NLI-format data가 어려운 task라는 것을 발견했다. 대신 우리는 span of the query pronoun and referent을 보여주는 SuperGLUE로부터 reformatted된 WNLI을 사용한다. 우리는 Kocijan et al. (2019).로 부터 margin ranking loss를 사용하면서 RoBERTa를 finetune한다.

### Results

* 첫번 째 setting은 RoBERTa는 GLUE task development sets의 9개 모두 SOTA를 성취했다. 중요하게 RoBERTa는 같은 masked language modeling pretraining objective와 BERT_LARGE의 구조를 사용하지만 일관성있게 BERT_LARGE와 XLNet_LARGE의 성능이 우수했다. 이는 이 work에서 우리가 탐구한 dataset size와 training time과 같은 더욱 평범한 세부사항들과 비교하여 model architecture와 pretraining objective의 상대적인 중요성의 대한 의문을 남길 수 있다.

* 두번 째 setting은 우리는 RoBERTa를 GLUE leaderboard로 제출했고 9개의 tasks중 4개를 SOTA를 달성했고 지금까지 가장 높은 평균 점수를 성취했다. 이는 다른 top submissions와는 달리 multi-task finetuning을 의존하지않는 RoBERTa 때문에 특히 흥미롭다. 우리는 future work가 더 많은 섬세한 multi-task finetuning procedures를 포함하면서 이러한 결과들을 향상시킬 수 있다는 것을 기대한다.

## SQuAD Results

* 우리는 이전 work와 비교하여 SQuAD를 더욱 단순한 approach로 채택한다. BERT (Devlin et al., 2019)와 XLNet (Yang et al., 2019)은 추가적인 QA datasets로 training data를 증가시켰지만, 우리는 제공된 SQuAD training data를 사용하면서 RoBERTa를 finetune했다. 우리가 모든 층을 같은 learning rate를 사용하는 반면 Yang et al. (2019)은 또한 XLNet을 finetune하기 위해 custom layer-wise learning rate schedule을 채택했다.

* SQuAD v1.1은 우리는 Devlin et al. (2019).과 같은 finetuning procedures를 따랐다. SQuAD v2.0은 우리는 추가적으로 주어진 question이 대답가능한지를 분류한다. 우리는 classification과 span loss terms를 더하면서 span predictor과 함께 이 classifier를 학습한다.

### Results

* SQuAD v1.1 development set은 RoBERTa는 XLNet의 SOTA와 일치했다. SQuAD v2.0 development set은 RoBERTa는 XLNet보다 EM은 0.4 points, F1은 0.6 points를 향상시키면서 SOTA를 달성했다.

* 우리는 RoBERTa를 public SQuAD 2.0 leaderboard로 제출하고, 상대적으로 다른 systems와 성능을 평가했다. 대부분 top systems는 BERT 혹은 XLNet으로 만들어졌고, 두 모델은 추가적인 외부의 training data를 의존한다. 대조적으로 우리의 submission은 어떠한 추가적인 data를 사용하지 않는다.

* 우리의 single RoBERTa 모델은 완전히 single model submissions의 성능보다 향상되었고, data augmentation을 의존하지 않는 모델들 중 top scoring system이다.

## RACE Results

* RACE는 systems는 passage of text, associated question, four candidate answers를 제공받았다. Systems는 4개의 candidate answers가 정확한지 분류하도록 요구된다.

* 우리는 이 task를 위해 각각의 candidate answer를 일치하는 question과 pasage를 연결하면서 RoBERTa를 수정한다. 우리는 각각 이러한 4개의 sequences를 encode하고 resulting representations를 정확한 answer를 예측하도록 사용되는 fully-connected layer로 지나게 했다. 우리는 필요하다면 128 tokens가 넘어가는 question-answer pairs와 passage를 삭제하고 그렇게 함으로써 total length는 많아야 512 tokens이다.

* RoBERTa는 middle-school과 high-school settings를 SOTA를 달성했다.

# Related Work

* Pretraining methods는 language modeling  (Dai and Le, 2015; Peters et al., 2018; Howard and Ruder, 2018), machine translation (McCann et al., 2017), masked language modeling (Devlin et al., 2019;Lample and Conneau, 2019).를 포함하는 다양한 training objectives로 설계되어 왔다. 많은 최근 papers는 각각 end task (Howard and Ruder,
2018; Radford et al., 2018)를 위해 finetuning models의 기본 recipe와 masked language model objective로 약간의 진폭과 함께 pretraining을 사용해왔다. 하지만 더욱 최신의 methods는 entity embeddings (Sun et al., 2019), span prediction (Joshi et al., 2019), multiple variants of autoregressive pretraining (Song et al., 2019;
Chan et al., 2019; Yang et al., 2019)을 포함하면서 multi-task fine tuning (Dong et al., 2019)하면서 성능을 향상시켜왔다. 성능은 전형적으로 더 많은 data (Devlin et al.,
2019; Baevski et al., 2019; Yang et al., 2019;
Radford et al., 2019)로 bigger models를 training 시키면서 향상되었다. 우리의 목표는 이러한 모든 methods의 상대적인 성능을 더욱 이해하기 위해 reference point로써 BERT의 training을 복제하고, 단순화하고, 더욱 잘 tune하는 것이다.

# Conclusion

* 우리는 BERT models를 pretraining 할 때 많은 design decisions를 평가한다. 우리는 성능은 더 많은 data로 큰 batch size로 지속적으로 모델을 길게 학습하고, next sentence prediction objective를 제거하고, 더욱 긴 sequences를 학습하고, training data로 적용되는 masking pattern을 변화시키면서 향상된다는 것을 발견했다. RoBERTa라고 불리는 우리의 향상된 pretraining procedure는 GLUE을 위해 multi-task finetuning과 SQuAD를 위해 추가적인 data 없이 GLUE, RACE, SQuAD의 tasks를 SOTA를 달성했다. 이러한 결과들은 이전은 간과한 design decisions의 중요성을 설명하고, BERT의 pretraining objective는 최근 제안된 대안들과 경쟁적이라는 것을 제안한다. 우리는 추가적으로 새로운 dataset인 CC-NEWS를 사용하고 pretraining과 finetuning을 위한 모델과 code를 공개한다.