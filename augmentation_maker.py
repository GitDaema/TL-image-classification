from torchvision import transforms

"""
증강을 쉽게 스위칭하기 위한 증강 설정 저장용 파일
"""

MODE_SUPER_NAME = 'super'
MODE_FOR_GAST_NAME = 'for_gast'
MODE_DEFAULT_NAME = 'default'

MODE_NAME_LIST = [MODE_SUPER_NAME, MODE_FOR_GAST_NAME]

def get_augmentation(mode_name, image_size, normalize):
    """ 입력한 모드 이름 문자열에 따라 미리 설정된 증강 규칙을 반환하는 함수  
    """

    if mode_name == MODE_SUPER_NAME: # 기본보다 증강 수치만 강화된 버전
        # transform.Compose는 여러 전처리 및 데이터 증강 기법을 하나의 파이프라인으로 묶어주는 객체
        # 안에 들어간 리스트에 적힌 순서대로 적용되니 주의, ToTensor, 정규화 맨 뒤로
        # 단, 이미지의 일부를 무작위로 지우는 것은 '최종 결정 이후의 변화'이니 다 끝나고 진행해야 함

        # 크기 조절하는 이유는 여러 이미지를 동일 행렬 모양으로 묶어 계산하면서, 최종 분류층의 고정 입력에 맞추기 위함
        # 보통 224인 이유는 반으로 계속 나누어도 딱 떨어지다가 마지막에 홀수인 7이 남아서 사진의 정중앙을 찾을 수 있기 때문 
        return transforms.Compose([
                # scale(최소, 최대) 비율만큼의 이미지 영역을 무작위로 오려내서 알맞은 크기로 다시 확대 
                transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)), 
                # transforms.Resize(image_size),

                # 상하좌우로 이미지를 translate(최대, 최대)만큼 평행 이동
                transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),

                transforms.RandomVerticalFlip(p=0.5), # 상하 반전
                transforms.RandomHorizontalFlip(p=0.5), # 좌우 반전, p는 확률(0.0 ~ 1.0)
                transforms.RandomRotation(degrees=15), # 이미지 회전, -degrees ~ degrees 사이 회전각 랜덤

                # 밝기, 대비, 채도, 색조를 무작위로 변경
                # 최대 얼마나 변동(플러스 마이너스)할 것인지 비율
                transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),

                # 선명도 조절, sharpness_factor는 원본보다 몇 배 선명하게 만들지, p는 확률
                transforms.RandomAdjustSharpness(sharpness_factor=2.0, p=0.3),

                # 가우시안 블러(흐림 효과)를 적용해 저화질처럼 흐릿하게 보이게, p는 확률
                # 커널 크기는 흐림 필터 크기(중앙이 있어야 해서 반드시 홀수), sigma(최소, 최대) 범위 내 흐림 강도
                transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=(0.2, 3.0))], p=0.3),

                transforms.ToTensor(),
                normalize, 

                # p만큼의 확률로, 이미지 전체에서 scale(최소, 최대) 비율만큼의 영역을 무작위로 지움 
                # value 값을 random으로 하면 무작위 컬러 픽셀로 채우니 흑백/컬러 분류 상관 없이 0으로 통일 
                transforms.RandomErasing(p=0.4, scale=(0.02, 0.1), value=0)
            ])
    elif mode_name == MODE_FOR_GAST_NAME:
        return transforms.Compose([
                transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)), 

                transforms.RandomRotation(degrees=180), # 이미지 회전, -degrees ~ degrees 사이 회전각 랜덤

                # 상하좌우로 이미지를 translate(최대, 최대)만큼 평행 이동
                # scale(최소, 최대) 크기 확대 또는 축소, shear은 평행사변형 모양으로 비틀기
                transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.95, 1.05), shear=5),

                transforms.RandomVerticalFlip(p=0.5), # 상하 반전
                transforms.RandomHorizontalFlip(p=0.5), # 좌우 반전, p는 확률(0.0 ~ 1.0)

                transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.1, hue=0.02),

                transforms.ToTensor(),
                normalize,

                # ratio(최소, 최대)는 가로/세로 비율 범위, 가로로 길거나 세로로 긴 사각형 만들기용
                transforms.RandomErasing(p=0.4, scale=(0.02, 0.15), ratio=(0.3, 3.3), value=0)
            ])
    else: # 특정 모드가 아니면 기본 설정
        return transforms.Compose([
                # scale(최소, 최대) 비율만큼의 이미지 영역을 무작위로 오려내서 알맞은 크기로 다시 확대 
                transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)), 
                # transforms.Resize(image_size),

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

def get_augmentation_mode_name(mode_name):
    if mode_name in MODE_NAME_LIST:
        return mode_name

    return MODE_DEFAULT_NAME