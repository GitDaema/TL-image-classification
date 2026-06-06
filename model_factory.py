from torch import optim
from ast import mod
from torch.nn import modules
import torch.nn as nn
from torchvision import models, transforms

import settings

class ModelFactory:  
    def create_model_and_transforms(model_name, num_classes):
        """ 모델명과 클래스 수를 입력받아 모델 객체와 학습/검증용 전처리 객체를 반환하는 메서드
        """

        """ 모델 가중치 가져올 때 if문 말고 딕셔너리 쓴 이유
        처음에 모델 가중치 가져올 때는 죄다 이렇게 if문으로 구별하도록 구현
        if model_name == 'resnet50':
            # DEFAULT는 파이토치에서 제공하는 최고 성능 가중치 데이터
            weights = models.ResNet50_Weights.DEFAULT
            model = models.resnet50(weights=weights)
            target_layer_name = 'fc' # resnet 출력층 이름은 fc
            in_features = model.fc.in_features

        그런데 if문은 늘어나는데 안에서는 비슷한 문장만 나와서 좋지 않은 구조라고 판단
        그래서 서로 다른 부분만 딕셔너리에 저장하고 공통 부분만 따로 떼어 구현
        """
        
        model_constructor, weight_class, last_layer_name, more_grad_layer_name = settings.MODEL_INFO_DICT[model_name]

        weights = weight_class.DEFAULT # DEFALUT는 파이토치가 추천하는 최고 성능 가중치
        model = model_constructor(weights=weights)

        
        """ 이미 잘 학습된 모델의 가중치는 그대로 놔두어야 함 
        그러므로 우선 가중치 재학습 가능 여부 전부 False로 설정 
        단, 마지막 레이어는 고정하지 않아야 학습 가능하나 이후 새 레이어 생성 시 알아서 True로 풀림
        컴퓨터 비전을 위한 전이 학습 튜토리얼 - 파이토치 공식 튜토리얼
        https://docs.pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
        여기에서 말하길 새로 만들어 넣은 레이어는 재학습 가능 여부가 기본값인 True, 아래 3줄 참고 
        # Parameters of newly constructed modules have requires_grad=True by default
        num_ftrs = model_conv.fc.in_features
        model_conv.fc = nn.Linear(num_ftrs, 2)
        """
        if settings.CAN_FREEZE_LAYERS:
            for param in model.parameters():
                param.requires_grad = False

        """ 몇몇 추가 재학습 필요 레이어들을 더 선정해 가중치 재학습 가능 여부를 True로 풀어주기
        만약 멀리 있는 앞쪽 레이어까지 한꺼번에 열면 소규모 데이터 셋에서는 가중치 과다로 인해 과적합 발생
        최종 출력층과 가까운 후반부 레이어만 해제해야 역전파 시 기울기 소실 없이 오차 정보를 온전히 전달받아 가중치를 갱신할 수 있음

        마찬가지로 모델마다 후반부 추가 재학습 필요 레이어의 이름이 달라 정보 딕셔너리에 저장 후 사용
        """
        if settings.CAN_GRAD_MORE_LAYERS:
            for name, param in model.named_parameters():
                for target_layer in more_grad_layer_name:
                    if target_layer in name:
                        param.requires_grad = True

        # 우선 문자열로 저장된 모델의 마지막 레이어 이름으로 실제 그 변수에 접근해 값 가져오기
        old_layer = getattr(model, last_layer_name)

        # 최종 레이어가 받아들이는 입력 데이터 개수(in_features)를 알아내는 방식
        # ResNet, DenseNet은 마지막 레이어가 단일 nn.Linear문이라 in_features를 그대로 가져올 수 있음
        # 그러나 VGG, EfficientNet은 끝부분이 여러 레이어가 묶인 nn.Sequential 구조라 그대로 못 가져옴
        # 따라서 구조가 어떻게 되어 있든 내부에서 in_features를 가진 첫 번째 레이어 값을 추출하도록 구현
        in_features = None 

        for module in old_layer.modules(): # 모든 하위 레이어를 확인
            # 해당 레이어에 in_features 속성이 있으면 그 값을 저장하고 반복문 종료
            if hasattr(module, 'in_features'):
                in_features = module.in_features 
                break

        # 그 다음 입력 차원은 유지하고 출력 차원만 이번 학습 클래스 수에 맞춰 바꿔 새 레이어 생성 
        linear_layer = nn.Linear(in_features, num_classes)

        # 드롭아웃 레이어 추가 생성
        dropout_layer = nn.Dropout(p=settings.DROPOUT_RATE)

        # 여러 레이어를 순서대로 쌓아 주는 nn.Sequential로 합치기
        # 만약 맨 마지막 단계에 드롭아웃을 써버리면 몇몇 결과값을 0으로 만들어 버려 학습 실패
        # 따라서 레이어 쌓는 순서 조심
        new_layer = nn.Sequential(dropout_layer, linear_layer) 
        
        # getattr와 반대로 그 이름의 변수에 새로 만든 레이어 객체를 저장해 덮어씌우기
        setattr(model, last_layer_name, new_layer)

        # 아까 선택한 가중치가 학습될 때 쓴 모든 전처리 규칙, 즉 표준 규칙을 가져와 저장
        weights_transforms = weights.transforms() 

        # 온갖 데이터 증대를 하더라도 마지막에는 표준 양식으로 돌아와야 함
        # 따라서 ToTensor(0 ~ 255를 0.0 ~ 1.0으로), 그리고 정규화(펑균 0, 표준편차 1)를 거침

        # 정규화할 때 원래는 ImageNet 표준 정규화 기본값으로 다음과 같이 작성
        # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        # 이렇게 전처리 객체에서 정규화 정보를 바로 가져올 수도 있음
        normalize = transforms.Normalize(mean=weights_transforms.mean, std=weights_transforms.std)

        if settings.CAN_USE_AUGMENTATION:
            # transform.Compose는 여러 전처리 및 데이터 증강 기법을 하나의 파이프라인으로 묶어주는 객체
            # 안에 들어간 리스트에 적힌 순서대로 적용되니 주의, ToTensor, 정규화 맨 뒤로
            # 단, 이미지의 일부를 무작위로 지우는 것은 '최종 결정 이후의 변화'이니 다 끝나고 진행해야 함

            # 크기 조절하는 이유는 여러 이미지를 동일 행렬 모양으로 묶어 계산하면서, 최종 분류층의 고정 입력에 맞추기 위함
            # 224인 이유는 반으로 계속 나누어도 딱 떨어지다가 마지막에 홀수인 7이 남아서 사진의 정중앙을 찾을 수 있기 때문 
            train_transform = transforms.Compose([
                # scale(최소, 최대) 비율만큼의 이미지 영역을 무작위로 오려내서 224 크기로 다시 확대 
                transforms.RandomResizedCrop((224, 224), scale=(0.8, 1.0)), 
                # transforms.Resize((224, 224)),

                # 상하좌우로 이미지를 translate(최대, 최대)만큼 평행 이동
                transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),

                transforms.RandomHorizontalFlip(p=0.5), # 좌우 반전, p는 확률(0.0 ~ 1.0)
                transforms.RandomRotation(degrees=15), # 이미지 회전, -degrees ~ degrees 사이 회전각 랜덤

                # 밝기, 대비, 채도, 색조를 무작위로 변경
                # 최대 얼마나 변동(플러스 마이너스)할 것인지 비율
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),

                # 선명도 조절, sharpness_factor는 원본보다 몇 배 선명하게 만들지, p는 확률
                transforms.RandomAdjustSharpness(sharpness_factor=2.0, p=0.3),

                # 가우시안 블러(흐림 효과)를 적용해 저화질처럼 흐릿하게 보이게, p는 확률
                # 커널 크기는 흐림 필터 크기(중앙이 있어야 해서 반드시 홀수), sigma(최소, 최대) 범위 내 흐림 강도
                transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))], p=0.1),

                transforms.ToTensor(),
                normalize, 

                # p만큼의 확률로, 이미지 전체에서 scale(최소, 최대) 비율만큼의 영역을 무작위로 지움 
                # value 값을 random으로 하면 무작위 컬러 픽셀로 채우니 흑백/컬러 분류 상관 없이 0으로 통일 
                transforms.RandomErasing(p=0.2, scale=(0.02, 0.05), value=0)
            ])
            # val 때도 마찬가지로 Resize로 최종 분류층을 위한 크기 통일은 필수
            # 하지만 검증에 쓸 문제 이미지를 무작위로 변형하면 매번 정확도가 불안정해져서 성능을 제대로 측정할 수 없음
            # 따라서 좌우 반전이나 회전 같은 증대 기법을 val에서 사용해서는 안 됨  
            val_transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                normalize
            ])
        else: # 데이터 증대 안 할 때도 모델이 항상 동일한 크기와 정규화를 거치도록 고정
            normalized_transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                normalize
            ])
            train_transform = normalized_transform
            val_transform = normalized_transform

        return model, train_transform, val_transform
