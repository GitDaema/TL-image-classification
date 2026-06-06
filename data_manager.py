from PIL import Image
from torchvision import datasets
from torch.utils.data import DataLoader
import settings

class DataManager:
    """
    설정된 경로를 따라 데이터셋의 정보를 가져오고 가공하는 클래스
    """
    def __init__(self):
        
        """ 
        원래는 로더 제작용 메서드에서 클래스 수도 반환하도록 구현했음
        그런데 로더 제작용 메서드의 매개변수는 전처리 규칙
        모델의 전처리 규칙 제작용 메서드의 매개변수는 클래스 수
        따라서 닭과 달걀 문제가 생겼기 때문에 이런 식으로 생성하자마자 클래스 수를 저장
        """
        # transform에 아무 값도 안 넣고 이미지 폴더를 열면 데이터 변환 없이 구조만 읽을 수 있음
        data = datasets.ImageFolder(root=settings.TRAIN_DIR)
        self._num_classes = len(data.classes) # 앞에 _를 붙이는 것은 밖에서는 읽기 전용이라는 뜻
        self._data_size = len(data)
        self._classes = data.classes
        
    @property
    def num_classes(self):
        # 이렇게 @property를 붙인 메서드로만 접근할 수 있게 하면 밖에서는 읽기 전용
        return self._num_classes
    
    @property
    def data_size(self):
        return self._data_size
    
    @property
    def classes(self):
        return self._classes

    def get_loaders(self, train_transform, val_transform):
        """
        각 전처리 규칙을 입력 받고, 설정된 경로를 따라 데이터셋을 읽어
        모델 학습용 객체로 변환해 반환하는 메서드
        """
        # 각 전처리 규칙 적용해서 폴더 구조 분석 및 클래스명 자동 할당 후 데이터셋을 생성
        train_data_set = datasets.ImageFolder(root=settings.TRAIN_DIR, transform=train_transform)
        val_data_set = datasets.ImageFolder(root=settings.VAL_DIR, transform=val_transform)
        
        # DataLoader를 이용해 데이터셋을 쪼개서 배치 단위로 묶어 모델 학습 및 평가용 객체로 저장
        # 배치는 한 번 학습할 때 동시에 처리하는 데이터의 묶음, 이번 프로젝트에서는 이미지 개수
        # 학습 때는 섞어야 편향을 막을 수 있음
        train_loader = DataLoader(train_data_set, batch_size=settings.BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_data_set, batch_size=settings.BATCH_SIZE, shuffle=False)
        
        return train_loader, val_loader