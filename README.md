# 🔥2024_2 프로메테우스 EEG 프로젝트팀 

ElectroEncephaloGram을 사용한 Emotion Recognition model 개선 및 성능 비교 프로젝트

## Training Dataset
* SEED dataset 사용
* 보유하고 있는 뇌파 측정기의 channel이 1개인 이슈로 인해 62개의 channel을 사용해 측정한 SEED dataset에서 1개의 channel만 추출하고 전처리

## Model 1: CNN-bi-LSTM with Attention
* torcheeg를 통해 SEED dataset에서 1개의 channel만 불러옴과 동시에 전처리 진행
* CNN과 bi-LSTM 사용 (Pooling layer 제외)
* Attention 메커니즘 사용
* 최종 모델

## Model 2: CCNN
* torcheeg를 통해 SEED dataset에서 1개의 channel만 불러옴과 동시에 전처리 진행
* torcheeg에서 지원하는 Continuous Convolutional Neural Network(CCNN)을 사용
* EEG의 시간적 관계(시계열)와 공간적 관계(부위별 전극 사이)를 학습
  
## Model 3: CNN-bi-LSTM with Residual Connection
* torcheeg를 통해 SEED dataset에서 1개의 channel만 불러옴과 동시에 전처리 진행
* 3개의 합성곱 층을 쌓은 CNN과 bi-LSTM을 결합 + Residual Connection 이용

## Model 4: Resnet-bi-Lstm with Attention
* torcheeg를 통해 SEED dataset에서 1개의 channel만 불러옴과 동시에 전처리 진행
* ResNet-34로 공간적 특징을 추출 
* BiLSTM으로 시계열 패턴을 학습 
* Attention 메커니즘으로 중요정보를 강조하는 신경망 모델

## Members
이지은, 조현진, 윤지찬, 최윤서, 윤상민, 심수민
