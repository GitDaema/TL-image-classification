from torchvision import models

"""
파라미터 등 상수 및 환경 설정 저장용 파일
"""

# 데이터셋 경로 저장용 딕셔너리
DIR_DICT = { 
    'ani' : ('data/ani/train', 'data/ani/val'),         
    'CUB200' : ('data/CUB200/train', 'data/CUB200/val'), 
    'hymenoptera' : ('data/hymenoptera_data/train', 'data/hymenoptera_data/val'), 
}

# 경로 설정
TRAIN_DIR, VAL_DIR = DIR_DICT['ani']

# 테스트할 모델 리스트
MODEL_NAME_LIST = ['resnet50'] # , 'densenet121']
MODEL_INFO_DICT = {
    'resnet50': (models.resnet50, models.ResNet50_Weights, 'fc', ['layer3', 'layer4']),
    'densenet121': (models.densenet121, models.DenseNet121_Weights, 'classifier', ['features.denseblock4', 'features.norm5'])
}
""" 모델 정보 딕셔너리(모델 이름 : 모델 정보 튜플)

첫번째는 모델, 정확히는 모델 생성자로, 이것과 가중치 객체를 이용해 모델 객체 생성 가능
모델 및 사전 학습된 가중치 - 파이토치 공식 문서 
https://docs.pytorch.org/vision/main/models.html
두번째는 가중치, 정확히는 가중치 클래스로, 여기에는 여러가지 가중치가 저장되어 있음
세 번째는 모델의 마지막 레이어 이름
네 번째는 추가 재학습 필요 레이어 이름 리스트
"""

""" 추가 재학습 필요 레이어 선정 참고 자료
만약 멀리 있는 앞쪽 레이어까지 한꺼번에 열면 소규모 데이터 셋에서는 가중치 과다로 인해 과적합 발생
최종 출력층과 가까운 후반부 레이어만 해제해야 역전파 시 기울기 소실 없이 오차 정보를 온전히 전달받아 가중치를 갱신할 수 있음

추가 재학습 필요 레이어 선정은 모델 코드에서 레이어들의 이름을 찾은 뒤, Gemini Pro 3.1와 대화하며 얼마나 열지 시행착오를 진행

Resnet50 
JayPatwardhan / Resnet50-Pytorch / ResNet / ResNet.py 깃허브 코드
https://github.com/JayPatwardhan/ResNet-PyTorch/blob/master/ResNet/ResNet.py
conv1 ... layer1 ~ 4 ... fc(마지막 레이어)

Densenet121
pytorch / vision / torchvision / models / densenet.py 깃허브 코드
https://github.com/pytorch/vision/blob/main/torchvision/models/densenet.py

반복문에서 self.features.add_module("denseblock%d" % (i + 1), block)처럼 레이어 추가하는 방식
features.conv0 ... features.denseblock1 ~ 4, features.norm5 ... classifier (마지막 레이어)
"""

# 하이퍼파라미터
BATCH_SIZE = 32
EPOCHS = 70
LEARNING_RATE = 0.0005 # 학습률

# 모멘텀 = 가중치 이동 방향에 관성을 부여해 학습 성능을 높이는 기법
OPTIM_MOMENTUM = 0.9 # 옵티마이저에 적용할 모멘텀 수치

# 드롭 아웃 = 일정 비율의 가중치를 불능으로 만들고 학습하는 규제 기법
DROPOUT_RATE = 0.5 # 모델 레이어에 적용할 드롭아웃 비율

# 가중치 감쇠 = 과잉 적합으로 인해 가중치 값이 커지는 현상을 막는 기법
WEIGHT_DECAY = 0.05 # 옵티마이저에 적용할 가중치 감쇠 정도

# 레이블 스무딩 = 과적합을 막기 위해 데이터 정답 확신도 일부를 다른 클래스에 배분하는 기법
LABEL_SMOOTHING = 0.1 # 손실 함수에 적용할 레이블 스무딩 비율

# 조기 종료 = val loss의 최솟값 갱신을 n회 이상 못하면 의미 없는 학습이라 판단하고 중단하는 기능
EARLY_STOP_PATIENCE = 15

# 기본적인 레이어 가중치 동결 여부(동결하지 않으면 미세 조정 방식)
# 동결해도 마지막 레이어는 학습 가능한 상태로 열려 있음
CAN_FREEZE_LAYERS = True

# 미리 정해 놓은 추가 레이어 재학습 가능 적용 여부(얼리지 않고 더 많이 부분 미세 조정)
CAN_GRAD_MORE_LAYERS = True

CAN_DRAW_PLOT = False # 임시 그래프 그리기 기능 온오프