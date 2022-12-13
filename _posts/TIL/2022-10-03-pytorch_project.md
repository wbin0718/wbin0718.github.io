---
title: "[TIL PyTorch] 파이토치 프로젝트 구조"
excerpt: ""
categories:
  - TIL
tags:
  - 

toc: true
toc_sticky: true
--- 

# Pytorch 프로젝트 구조 이해하기

## Jupyter

* 초기 단계에서는 대화식 개발 과정이 유리

    * 학습과정과 디버깅 등 지속적인 확인

* 배포 및 공유 단계에서는 notebook 공유의 어려움

    * 쉬운 재현의 어려움, 실행순서 꼬임

* DL 코드도 하나의 프로그램

    * 개발 용이성 확보와 유지보수 향상 필요


> 코드를 레고블럭 처럼

## Module 구성

pytorch-template/   
│   
├── train.py - main script to start training   
├── test.py - evaluation of trained model   
│   
├── config.json - holds configuration for training   
├── parse_config.py - class to handle config file and cli options   
│   
├── new_project.py - initialize new project with template files   
│   
├── base/ - abstract base classes   
│   ├── base_data_loader.py   
│   ├── base_model.py   
│   └── base_trainer.py   
│   
├── data_loader/ - anything about data loading goes here   
│   └── data_loaders.py   
│   
├── data/ - default directory for storing input data   
│   
├── model/ - models, losses, and metrics   
│   ├── model.py   
│   ├── metric.py   
│   └── loss.py   
│   
├── saved/   
│   ├── models/ - trained models are saved here      
│   └── log/ - default logdir for tensorboard and logging    output   
│   
├── trainer/ - trainers   
│   └── trainer.py   
│   
├── logger/ - module for tensorboard visualization and logging   
│   ├── visualization.py   
│   ├── logger.py   
│   └── logger_config.json   
│   
└── utils/ - small utility functions   
    ├── util.py   
    └── ...   


