import torch

class Trainer:
    """
    모델, 설정, 손실 함수, 옵티마이저를 받아와 모델 학습 및 평가를 진행하는 클래스
    """
    def __init__(self, model, device, criterion, optimizer):
        # to(장치)로 모델을 설정에서 정한 최적의 장치로 보냄
        self.model = model.to(device)
        self.device = device
        self.criterion = criterion
        self.optimizer = optimizer

    def train_epoch(self, loader):
        """ 모델 학습용 로더를 가져와 한 에포크를 학습하고 평균 오차와 정확도를 반환하는 메서드
        """

        # 모델을 학습 모드로 설정해 드롭아웃 등을 활성화
        self.model.train()
        
        # 총 오차와 총 정답 수 초기화
        loss_sum, answer_sum = 0.0, 0
        
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

            # 이후 예측 인덱스 리스트와 실제 정답 인덱스 리스트를 비교
            # 파이토치에서 True는 1로 계산되므로 그 합이 결국 이번 배치에서 맞춘 정답 개수
            answer_sum += torch.sum(preds == labels.data)
            
        # 전체 데이터 개수로 나눈 평균 오차와 정확도를 반환
        return loss_sum / len(loader.dataset), answer_sum.double() / len(loader.dataset)

    def evaluate(self, loader):
        """ 모델 평가용 로더를 가져와 한 에포크의 평균 오차와 정확도를 반환하는 메서드
        """

        # 모델을 평가 모드로 설정
        self.model.eval()
        
        loss_sum, answer_sum = 0.0, 0
        
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
        
        return loss_sum / len(loader.dataset), answer_sum.double() / len(loader.dataset)