import torch
import settings
from torchvision.transforms import v2

def freeze_bn_buffers(module):
    """ 배치 정규화 레이어 동결 함수

    전이 학습할 때 보통 앞쪽 레이어의 가중치를 동결
    그러나 파이토치에서는 model.train()을 호출하면 가중치를 동결해도
    '배치 정규화 레이어'는 새로운 입력 데이터의 평균과 분산을 계산해 통계량을 자기가 업데이트해버림
    따라서 가중치가 동결된 배치 정규화 레이어는 아예 평가 모드로 강제 고정해야 함
    """
    # 현재 레이어가 배치 정규화 레이어인지 검사
    if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
        # 해당 레이어의 가중치가 존재하고, 재학습이 안 되는 동결 상태라면
        if module.weight is not None and module.weight.requires_grad == False:
            # 확실하게 평가 모드로 전환해서 통계량 고정
            module.eval()
            
class Trainer:
    """
    모델, 설정, 손실 함수, 옵티마이저를 받아와 모델 학습 및 평가를 진행하는 클래스
    """
    def __init__(self, model, device, criterion, optimizer, num_classes=None):
        # to(장치)로 모델을 설정에서 정한 최적의 장치로 보냄
        self.model = model.to(device)
        self.device = device
        self.criterion = criterion
        self.optimizer = optimizer

        # 믹스업은 두 이미지를 투명하게 겹치는 기법 
        # 컷 믹스는 한 이미지의 일부 영역을 다른 이미지에서 잘라와 대신해 붙이는 것
        # 이러면 레이블이 원-핫에서 소프트 레이블로 바뀜(정답 확신도가 a 70%, b 30%처럼 나뉨)
        # 06-image-classification/05-data-augmentation.md - 깃허브
        # https://github.com/jsonpassion/forge-tutorial-vision/blob/main/06-image-classification/05-data-augmentation.md
        self.mixup_cutmix = None
        if settings.CAN_USE_MIXUP_CUTMIX and num_classes is not None:
            mixup = v2.MixUp(num_classes=num_classes, alpha=settings.MIXUP_ALPHA)
            cutmix = v2.CutMix(num_classes=num_classes, alpha=settings.CUTMIX_ALPHA)
            self.mixup_cutmix = v2.RandomChoice([mixup, cutmix]) # 컷믹스와 믹스업 중 하나 랜덤 선택

    def train_epoch(self, loader):
        """ 모델 학습용 로더를 가져와 한 에포크를 학습하고 평균 오차와 정확도를 반환하는 메서드
        """

        # 모델을 학습 모드로 설정해 드롭아웃 등을 활성화
        self.model.train()

        # BN 네트워크의 나머지 부분을 학습하는 동안 레이어를 고정하는 방법 - 파이토치 논의(질답)
        # https://discuss.pytorch.org/t/how-to-freeze-bn-layers-while-training-the-rest-of-network-mean-and-var-wont-freeze/89736/11
        self.model.apply(freeze_bn_buffers)
        
        # 총 오차와 총 정답 수 초기화
        loss_sum, answer_sum = 0.0, 0
        total_samples = 0 # 진짜 처리한 데이터 개수
        
        """ 참고한 구조
        컴퓨터 비전을 위한 전이 학습 튜토리얼 - 파이토치 공식 튜토리얼
        https://docs.pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
        해당 링크에서 if phase == 'train'과 같이 학습 단계인지 검사하는 부분은 전부 제거
        지금 있는 메서드는 학습만을 위한 메서드이므로 필요 없음
        """
        # 로더에서 이미지(입력)와 정답(레이블)을 배치 단위로 받아서 하나씩 훑기
        for inputs, labels in loader:
            # 마찬가지로 to()를 이용해 데이터를 설정한 장치로 보내기
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            
            if settings.CAN_USE_MIXUP_CUTMIX and self.mixup_cutmix is not None:
                # 이미지와 레이블을 믹스업 또는 컷믹스 기법에 맞게 합성
                inputs, labels = self.mixup_cutmix(inputs, labels)

            # 1. 이전 배치의 계산 결과가 영향을 주지 않도록 기울기 초기화
            self.optimizer.zero_grad()
            
            # 2. 모델에 이미지를 전달한 뒤 이를 통해서 반환된 예측 값을 저장  
            outputs = self.model(inputs)
            
            # 3. 손실 함수를 이용해 모델이 출력한 예측 값과 실제 정답 사이의 오차 계산
            loss = self.criterion(outputs, labels)
            
            # 4. 역전파로 뒤로 돌아가서 각 파라미터가 오차에 기여한 정도(기울기)를 계산
            # 이를 통해 파라미터의 수정 방향과 크기를 결정할 수 있음
            loss.backward()
            
            # 5. 역전파 결과와 옵티마이저의 규칙에 따라 파라미터를 실제로 갱신 
            self.optimizer.step()
            
            # loss.item()은 배치 전체의 평균 오차이므로, 데이터 개수인 inputs.size(0)을 곱한 것이 진짜 합계
            loss_sum += loss.item() * inputs.size(0)
            
            # 위에서 모델이 출력한 예측 값은 (배치 크기, 클래스 수) 형태의 행렬
            # 여기에서 max(예측 값, 1)은 행으로 최대값을 찾으라는 뜻
            # 따라서 이미지마다 가장 확률이 높은 것을 골라 (확률, 예측한 인덱스)로 내보냄
            # 여기에서는 예측 인덱스 리스트만 저장
            _, preds = torch.max(outputs, 1)

            if labels.ndim > 1: # 레이블이 2차원 확률 배열이면 믹스업 상태
                last_labels = torch.max(labels, 1)[1] # 가장 확률이 높은 클래스를 정답으로 간주
            else: # 1차원 정수 배열이면 믹스업 아니니 그대로
                last_labels = labels.data

            # 이후 예측 인덱스 리스트와 실제 정답 인덱스 리스트를 비교
            # 파이토치에서 True는 1로 계산되므로 그 합이 결국 이번 배치에서 맞춘 정답 개수
            answer_sum += torch.sum(preds == last_labels)
            total_samples += inputs.size(0) # 이번 배치 크기만큼 더해서 진짜 처리한 데이터 개수 계산
            
        # 전체 데이터 개수로 나눈 평균 오차와 정확도를 반환
        return loss_sum / total_samples, answer_sum.double() / total_samples

    def evaluate(self, loader):
        """ 모델 평가용 로더를 가져와 한 에포크의 평균 오차와 정확도를 반환하는 메서드
        """

        # 모델을 평가 모드로 설정
        self.model.eval()
        
        loss_sum, answer_sum = 0.0, 0
        total_samples = 0
        
        # with는 특정 상태나 환경을 잠깐 활성화했다가 블록을 벗어나면 자동으로 꺼주는 역할
        # 평가할 때는 가중치를 수정할 필요가 없으므로 기울기 계산을 꺼서 메모리를 절약하는 것이 나음
        # 코드 실행 중 예외가 발생하더라도 활성화 전 상태로 복구해 주니 안정성도 보장
        with torch.no_grad():
            # 나머지는 학습 과정에서 '학습에만 필요한 단계'를 뺀 것과 동일
            for inputs, labels in loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                
                loss_sum += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                answer_sum += torch.sum(preds == labels.data)
                total_samples += inputs.size(0)
        
        return loss_sum / total_samples, answer_sum.double() / total_samples