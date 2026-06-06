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
DATA_SET_NAME = 'ani'
TRAIN_DIR, VAL_DIR = DIR_DICT[DATA_SET_NAME]

SAVED_MODELS_FOLDER_NAME = "my_models"
LOADING_MODEL_NAME = ""

# 테스트할 모델 리스트
MODEL_NAME_LIST = ['resnet50']
MODEL_INFO_DICT = {
    'resnet50': (models.resnet50, models.ResNet50_Weights, 'fc', ['layer3', 'layer4']),
    'densenet121': (models.densenet121, models.DenseNet121_Weights, 'classifier', ['features.denseblock3', 'features.transition3', 'features.denseblock4', 'features.norm5']),
    'vgg16': (models.vgg16, models.VGG16_Weights, 'classifier', ['features.19', 'features.21', 'features.24', 'features.26', 'features.28']),
    'efficientnet_b4': (models.efficientnet_b4, models.EfficientNet_B4_Weights, 'classifier', ['features.5', 'features.6', 'features.7']),
}
{ """ 모델 정보 딕셔너리(모델 이름 : 모델 정보 튜플)

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

======

Densenet121
pytorch / vision / torchvision / models / densenet.py 깃허브 코드
https://github.com/pytorch/vision/blob/main/torchvision/models/densenet.py
features.conv0 ... features.denseblock3, features.transition3, features.denseblock4, features.norm5 ... classifier (마지막 레이어)

Densenet은 반복문에서 아래 코드처럼 transition과 denseblock을 번갈아가며 추가하는 방식
self.features.add_module("denseblock%d" % (i + 1), block)
num_features = num_features + num_layers * growth_rate
if i != len(block_config) - 1:
    trans = _Transition(num_input_features=num_features, num_output_features=num_features // 2)
    self.features.add_module("transition%d" % (i + 1), trans)
    num_features = num_features // 2

======

VGG16
pytorch / vision / torchvision / models / vgg.py 깃허브 코드
https://github.com/pytorch/vision/blob/main/torchvision/models/vgg.py
features.0(Conv2d) ... features.23(MaxPool2d), features.24(Conv2d), features.25(ReLU), features.26(Conv2d), 
features.27(ReLU), features.28(Conv2d), features.29(ReLU) ... classifier (마지막 레이어)

VGG는 다음과 같이 반복문을 돌면서 레이어 이름이 인덱스 번호로 자동 지정되도록 설정
Conv2d 레이어 사이에 ReLU 함수가 하나씩 끼어 있어서 실제 후반부 레이어는 2칸씩 떨어져 있음
단, 23번은 MaxPool2d라서 여기만 잠깐 또 건너뛰기(19, 21, 24, 26, 28)
def make_layers(cfg: list[Union[str, int]], batch_norm: bool = False) -> nn.Sequential:
    layers: list[nn.Module] = []
    in_channels = 3
    for v in cfg:
        if v == "M":
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            v = cast(int, v)
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)

======

EfficientNet_b4
pytorch / vision / torchvision / models / efficientnet.py 깃허브 코드
https://github.com/pytorch/vision/blob/main/torchvision/models/efficientnet.py
features.0(첫 Conv) ... features.4(MBConv), features.5(MBConv), features.6(MBConv), 
features.7(마지막 Conv2dNormActivation) ... classifier (마지막 레이어)

MBConv는 역전된 잔차 블록으로, 비교적 좁은 입력 계층을 더 넓은 내부 계층에 매핑하고, 그 내부 레이어가 좁은 출력 계층으로 매핑을 전환함
https://kr.linkedin.com/pulse/anatomy-high-performance-mbconv-block-andrew-lavin?tl=ko

반복문을 돌며 순서대로 MBConv 블록들을 self.features 리스트에 append 한 뒤 
마지막에 최종 Conv 층을 추가하고 묶어줌, 이 과정에서 0 ~ 7까지 통재로 인덱스가 부여
for cnf in inverted_residual_setting:
    stage: list[nn.Module] = []
    for _ in range(cnf.num_layers):
        # copy to avoid modifications. shallow copy is enough
        block_cnf = copy.copy(cnf)

        # overwrite info if not the first conv in the stage
        if stage:
            block_cnf.input_channels = block_cnf.out_channels
            block_cnf.stride = 1

        # adjust stochastic depth probability based on the depth of the stage block
        sd_prob = stochastic_depth_prob * float(stage_block_id) / total_stage_blocks

        stage.append(block_cnf.block(block_cnf, sd_prob, norm_layer))
        stage_block_id += 1

    layers.append(nn.Sequential(*stage))

# building last several layers
lastconv_input_channels = inverted_residual_setting[-1].out_channels
lastconv_output_channels = last_channel if last_channel is not None else 4 * lastconv_input_channels
layers.append(
    Conv2dNormActivation(
        lastconv_input_channels,
        lastconv_output_channels,
        kernel_size=1,
        norm_layer=norm_layer,
        activation_layer=nn.SiLU,
    )
)

self.features = nn.Sequential(*layers)

======

""" }

# 하이퍼파라미터
BATCH_SIZE = 32
EPOCHS = 50

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
EARLY_STOP_PATIENCE = 50 # val loss가 몇 번 안 줄어들면 중단할지 횟수

# 기본적인 레이어 가중치 동결 여부(동결하지 않으면 미세 조정 방식)
# 동결해도 마지막 레이어는 학습 가능한 상태로 열려 있음
CAN_FREEZE_LAYERS = True

# 미리 정해 놓은 추가 레이어 재학습 가능 적용 여부(얼리지 않고 더 많이 부분 미세 조정)
CAN_GRAD_MORE_LAYERS = True

# 학습이 정체되면 학습률을 자동으로 낮춰주는 학습률 스케줄러 적용 여부
CAN_USE_SCHEDULER = True 
SCHEDULER_FACTOR = 0.5 # 학습률 감소 비율(배수)
SCHEDULER_PATIENCE = 5 # val loss가 몇 에포크 동안 안 줄어들면 학습률 낮출지 횟수

# 주어진 데이터를 좌우 반전, 회전 등 인위적으로 늘리는 데이터 증대 적용 여부
CAN_USE_AUGMENTATION = True

# 데이터가 적고 클래스 데이터 수가 불균형할 때 이를 맞춰주는 오버샘플링 적용 여부
CAN_USE_OVERSAMPLING = True
SAMPLER_MULTIPLIER = 2 # 에포크 당 학습하는 중복 데이터를 몇 배로 늘릴지

# 마지막 레이어와 이전 초중반 레이어의 학습률에 차이를 두는 차등 학습률 적용 여부
CAN_USE_DIFFERENTIAL_LEARNING_RATE = False
BASE_LEARNING_RATE_MULTIPLIER = 0.1 # 초중반 레이어 학습률에 곱할 값

CAN_DRAW_PLOT = False # 임시 그래프 그리기 기능 온오프