# 2026년 1학기 인공지능 레포트(전이학습 분석)

## 디렉토리 구조
- [main.ipynb](./main.ipynb): 메인 노트북
- [settings.py](./settings.py): 하이퍼파라미터 및 통합 설정
- [model_factory.py](./model_factory.py): 모델 생성 및 전이 학습 설정
- [trainer.py](./trainer.py): 실제 학습 및 평가
- [visualizer.py](./visualizer.py): 테스트 중 확인용 임시 시각화 기능

## 구현 계획
### 1. 기초 베이스라인
- 필수 모델(ResNet50, DenseNet121)과 필수 데이터셋(ani, CUB200)만 이용
- 규제 기법 적용 없이 순수 성능 측정

### 2. 규제 기법을 사용한 추가 테스트 및 분석
- 드롭아웃 등 여러 규제 기법 적용 후 성능 향상 정도 측정

### 3. 제 3의 모델을 사용한 추가 테스트 및 분석
- ResNet50, DenseNet121와 다른 모델 간 성능 비교

### 4. 외부 데이터셋을 사용한 추가 테스트 및 분석
- 외부 데이터셋을 활용한 모델의 범용성 확인

## 실행 방법
1. 필요한 라이브러리 설치(PyTorch, Torchvision 등)
2. [settings.py](./settings.py)에서 데이터 경로 및 하이퍼파라미터 확인
3. [main.ipynb](./main.ipynb)를 실행