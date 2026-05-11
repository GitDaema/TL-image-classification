import torch

class Setting:
    """
    CPU GPU 디바이스 확인 기능까지 포함한 설정값 저장용 클래스 
    """
    # 경로 설정
    TRAIN_DIR = 'data/ani/train'
    VAL_DIR = 'data/ani/val'
    
    # 하이퍼파라미터
    BATCH_SIZE = 8
    EPOCHS = 5
    LEARNING_RATE = 0.001
    OPTIM_MOMENTUM = 0.9
    
    # 테스트할 모델 리스트
    MODEL_NAME_LIST = ['resnet50', 'densenet121']

    CAN_DRAW_PLOT = False # 임시 그래프 그리기 기능 온오프

    # 장치 기본값은 일단 CPU
    device_name = "cpu"

    # 엔비디아 GPU가 있는지 확인
    if torch.cuda.is_available():
        device_name = "cuda"
        print("엔비디아 GPU cuda 사용")
    elif torch.backends.mps.is_available(): # 없으면 애플 실리콘 GPU가 있는지 확인 
        device_name = "mps"
        print("애플 실리콘 GPU mps 사용")
    else: # GPU 다 없으면 기본값대로 CPU 사용
        print("cpu 사용")

    DEVICE = torch.device(device_name)