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
        
        # 그 다음 입력 차원은 유지하고 출력 차원만 이번 학습 클래스 수에 맞춰 바꿔 새 레이어 생성 
        linear_layer = nn.Linear(old_layer.in_features, num_classes)

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

        train_transform = weights_transforms
        val_transform = weights_transforms

        return model, train_transform, val_transform