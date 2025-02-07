# 🔥2024_2 프로메테우스 EEG 프로젝트팀 

ElectroEncephaloGram을 사용한 Emotion Recognition model 개선 및 성능 비교 프로젝트

## 🤔Overview
* 기존 시계열 데이터 학습에 많이 쓰이는 다양한 모델에 뇌파 데이터를 학습 시켜 현재 그 사람이 어떤 감정을 느끼고 있는 지 판단 -> 성능 비교
* 감정은 긍정, 중립, 부정으로 label을 나눔

## 📑Training Dataset
* SEED dataset 사용
* 보유하고 있는 뇌파 측정기의 channel이 1개인 이슈로 인해 62개의 channel을 사용해 측정한 SEED dataset에서 1개의 channel만 추출하고 전처리
* torcheeg의 SEEDDataset 함수를 통해 학습용 SEED data 전처리

## 🧠Model 1: CNN-bi-LSTM with Attention
* torcheeg를 통해 SEED dataset에서 1개의 channel만 불러옴과 동시에 전처리 진행
* CNN과 bi-LSTM 사용 (Pooling layer 제외)
* Attention 메커니즘 사용
* **최종 모델**

## 🧠Model 2: CCNN
* torcheeg를 통해 SEED dataset에서 1개의 channel만 불러옴과 동시에 전처리 진행
* torcheeg에서 지원하는 Continuous Convolutional Neural Network(CCNN)을 사용
* EEG의 시간적 관계(시계열)와 공간적 관계(부위별 전극 사이)를 학습
  
## 🧠Model 3: CNN-bi-LSTM with Residual Connection
* torcheeg를 통해 SEED dataset에서 1개의 channel만 불러옴과 동시에 전처리 진행
* 3개의 합성곱 층을 쌓은 CNN과 bi-LSTM을 결합 + Residual Connection 이용

## 🧠Model 4: Resnet-bi-Lstm with Attention
* torcheeg를 통해 SEED dataset에서 1개의 channel만 불러옴과 동시에 전처리 진행
* ResNet-34로 공간적 특징을 추출 
* BiLSTM으로 시계열 패턴을 학습 
* Attention 메커니즘으로 중요정보를 강조하는 신경망 모델

## ⭐Demo Day
* 2025/02/08 프로메테우스 데모 데이 부스 운영
* 화제의 넷플릭스 드라마 오징어게임2에서 활약한 캐릭터들의 장면을 보여주고, 체험자가 가장 긍정의 감정을 느낀 캐릭터와 관련된 상품을 증정
![Image](https://github.com/user-attachments/assets/4799129f-64c6-474a-be45-c97041b1e235)

## 😎Members
| 심수민 (개발)      | 윤상민 (개발)     | 윤지찬 (개발)     | 이지은 (팀장, 개발)  | 조현진 (개발) | 최윤서 (디자인) |
|:-----------------:|:----------------:|:-----------------:|:--------------------:|:------------:|:--------------------:|
| 2기      | 5기 | 6기 | 4기 | 6기 | 6기 |
| [use08174](https://github.com/use08174)        |  [vivamini7](https://github.com/vivamini7)  |  [BETONM](https://github.com/BETONM)    | [FrozenAdrnln](https://github.com/FrozenAdrnln)|  [hyun-jin891](https://github.com/hyun-jin891))| [yun31](https://github.com/yun31)|
