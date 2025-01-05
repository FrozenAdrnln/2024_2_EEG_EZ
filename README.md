# 2024_2 프로메테우스 EEG 프로젝트팀 

ElectroEncephaloGram을 사용한 Emotion Recognition model 개선 및 성능 비교 프로젝트

## Completion Timeline

Due ~ 25/02/08

## ACRNN
* Channel-wise attention + cnn + lstm + attention
* tensorflow 구버전과 gpu 사이 연동 이슈로 인해 대용량 데이터를 학습시키에 무리가 있다고 판단 후 보류

## SEED_DEEP
* torcheeg 사용
* CNN + LSTM with maxpool, dropout, and residual connections
* softmax, cross entropy

## seed_ccnn
* torcheeg에서 지원하는 Continuous Convolutional Neural Network(CCNN)을 사용
* EEG의 시간적 관계(시계열)와 공간적 관계(부위별 전극 사이)를 학습
  
## seed_ccnn 실행방법

1. Installation

pip install -r requirements.txt

2. Prepare dataset

    Preprocessed_EEG/

    ├── label.mat

    ├── readme.txt

    ├── 10_20131130.mat

    ├── ...

    └── 9_20140704.mat
  
3. 실행

python examples_seed_ccnn.py


## Members
이지은, 조현진, 윤지찬, 최윤서, 윤상민, 심수민
