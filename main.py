from data_manager import DataManager
from model_factory import ModelFactory
from trainer import Trainer
import settings
import visualizer

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler

import copy
import os

import numpy as np

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

device = torch.device(device_name)

data_manager = DataManager()

print(f"데이터셋 이름: {settings.DATA_SET_NAME}, 이미지: {data_manager.data_size} 장, 클래스 개수: {data_manager.num_classes} 개")

for cur_model_name in settings.MODEL_NAME_LIST:
    print(f"\n--- {cur_model_name} 모델 실험 시작 ---")
    
    # 모델과 전처리 규칙을 가져옴
    model, train_transform, val_transform = ModelFactory.create_model_and_transforms(cur_model_name, data_manager.num_classes)

    # model, train_transform, val_transform = ModelFactory.new_create_model_and_transforms(cur_model_name, data_manager.num_classes, settings.NUM_CHANNELS)
    
    # 가져온 전처리 규칙으로 데이터를 모델 학습용 및 평가용으로 가공한 객체인 로더를 가져옴
    train_loader, val_loader = data_manager.get_loaders(train_transform, val_transform)

    # 손실 함수는 획득한 결과와 실제 값 사이의 틀린 정도를 측정하는 함수
    # 학습 중에 이 값을 최소화하려고 하며, 예측과 정답을 비교해 손실을 계산
    # 모델 매개변수 최적화하기 - 파이토치 한국어 튜토리얼
    # https://tutorials.pytorch.kr/beginner/basics/optimization_tutorial.html

    # 데이터가 적은 것도 잘 학습하도록 가중치 조절
    target_list = train_loader.dataset.targets # 모든 정답 레이블 리스트 형태로 가져오기
    class_counts = np.bincount(target_list) # 각 클래스별 개수를 넘파이 배열로 가져오기

    weights = 1.0 / class_counts # 데이터가 적을수록 가중치가 높게 설정되도록 역수 취하기

    # PyTorch에서 클래스 불균형 문제 해결하기 - Kaggle
    # https://www.kaggle.com/code/syzygyfy/addressing-the-class-imbalance-in-pytorch
    if settings.CAN_USE_OVERSAMPLING:
        # 각 데이터 샘플마다 부여될 가중치 배열 생성
        samples_weight = np.array([weights[t] for t in target_list])
        samples_weight = torch.from_numpy(samples_weight).double()
        
        # 원본 데이터 수 * 배수로 총 데이터 수 계산
        total_num_samples = len(target_list) * settings.SAMPLER_MULTIPLIER
        
        # 덮어씌울 샘플러 새로 생성할 때 replacement=True로 중복 뽑기 허용
        sampler = WeightedRandomSampler(weights=samples_weight, num_samples=total_num_samples, replacement=True)
        
        # 기존 로더의 속성은 그대로 유지한 채 샘플러만 교체하여 덮어쓰기
        train_loader = DataLoader(
            train_loader.dataset, 
            batch_size=train_loader.batch_size, 
            sampler=sampler,
            num_workers=getattr(train_loader, 'num_workers', 0), 
            pin_memory=getattr(train_loader, 'pin_memory', False)
        )
        print(f"오버샘플링 적용 중: 이미지 {settings.SAMPLER_MULTIPLIER}배, 총 {total_num_samples}장")

    # 가중치 배열을 파이토치로 변환하고 현재 장치로 이동
    class_weights = torch.FloatTensor(weights).to(device) 

    # 조절된 가중치 반영
    # criterion = nn.CrossEntropyLoss(weight = class_weights, label_smoothing=settings.LABEL_SMOOTHING)

    criterion = nn.CrossEntropyLoss(label_smoothing=settings.LABEL_SMOOTHING)

    # 옵티마이저는 손실 함수의 최저점을 찾아주는 탐색기

    # 가중치 재학습 여부를 False로 한 것들은 옵티마이저에 넣을 필요가 없음
    # 따라서 실제로 학습할 파라미터들만 모으는 리스트를 따로 생성
    # 설정에 따라 마지막 레이어만 들어가거나, 추가로 지정한 레이어들이 함께 포함됨
    param_list = []

    # 최종 분류층과 추가로 연 다른 레이어들의 학습률을 다르게 주는 차등 학습률 적용
    # 앞쪽 레이어는 가장자리, 모양 같은 일반적인 특징을 학습
    # 앞쪽에 학습된 가중치는 이미 제 역할을 잘 수행 중이므로 크게 변경할 필요 없음
    # 차등 학습률을 이용한 전이 학습 - 미디움 TDS 아카이브 
    # https://medium.com/data-science/transfer-learning-using-differential-learning-rates-638455797f00
    if settings.CAN_USE_DIFFERENTIAL_LEARNING_RATE:
        # 마지막 레이어 이름만 가져오기
        _, _, last_layer_name, _ = settings.MODEL_INFO_DICT[cur_model_name]

        base_params = [] # 학습률 낮춰서 미세조정할 백본 레이어 가중치 리스트
        classifier_params = [] # 그대로 학습률 적용할 최종 레이어 가중치 리스트

        for name, param in model.named_parameters():
            if param.requires_grad: # 학습이 가능한 레이어면
                if name.startswith(last_layer_name): # 마지막 레이어 이름과 같은지 확인
                    classifier_params.append(param)
                else:
                    base_params.append(param)

        # ReferenceAPI / torch.potim - 파이토치 문서 
        # https://docs.pytorch.org/docs/2.12/optim.html
        param_list = [ # 마지막 레이어 이름이면 그대로, 아니면 학습률 낮게
            {'params': base_params, 'lr': settings.LEARNING_RATE * settings.BASE_LEARNING_RATE_MULTIPLIER },
            { 'params': classifier_params, 'lr': settings.LEARNING_RATE }
        ]
    else:
        for param in model.parameters():
            if param.requires_grad == True:
                param_list.append(param) 
    
    # 컴퓨터 비전을 위한 전이 학습 튜토리얼에서는 옵티마이저로 SGD 사용
    # SGD 옵티마이저에 재학습 가능한 파라미터만 있는 리스트, 학습률, 모멘텀(관성) 전달해 객체 생성
    # optimizer = optim.SGD(param_list, lr=settings.LEARNING_RATE, momentum=settings.OPTIM_MOMENTUM, weight_decay=settings.WEIGHT_DECAY)
    
    # 옵티마이저로 Adam을 쓴 버전, SGD와 바꿔가며 테스트
    optimizer = optim.Adam(param_list, lr=settings.LEARNING_RATE, weight_decay=settings.WEIGHT_DECAY)

    if settings.CAN_USE_SCHEDULER:
        # 학습률 스케줄러를 이용하면 성능 개선이 안 될 때 중간에 학습률을 자동으로 낮춰줄 수 있음
        # factor는 학습률 감소 비율(배수), patience는 val loss가 몇 에포크 동안 안 줄어들면 학습률을 줄일지를 나타냄
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=settings.SCHEDULER_FACTOR 
                                                                            , patience=settings.SCHEDULER_PATIENCE)

    # 학습용 엔진에 모델, 설정, 손실 함수, 옵티마이저를 전달해 객체 생성
    trainer = Trainer(model, device, criterion, optimizer)

    # 불러올 모델이 있는지 확인하기 위해 폴더와 이름 경로 합치기
    model_path = os.path.join(settings.SAVED_MODELS_FOLDER_NAME, settings.LOADING_MODEL_NAME + ".pth")

    is_my_model_mode = False # 이미 저장된 모델을 불러올지 여부
    if os.path.isfile(model_path): # 실제로 파일 존재하는지 확인
        print(f"\n[학습 대신 저장된 모델 {settings.LOADING_MODEL_NAME} 로딩]")
        try:
            # 가중치 로드, 이때 파이토치는 학습 때의 메모리 장치까지 기록한 뒤 이에 따라 복원하려고 함
            # 따라서 학습된 환경이 다른 환경(CPU/GPU)일 수 있으니 현재 장치로 직접 지정해야 함
            model.load_state_dict(torch.load(model_path, map_location=device))
            is_my_model_mode = True
        except Exception as e:
            print(f"[저장된 모델 불러오던 중 오류 발생: {e}]")
    else:
        print(f"\n[불러올 모델 이름이 비어 있거나 잘못되어 있으므로 새로 학습 시작]")

    if is_my_model_mode:
        print(f"\n[불러온 모델 검증]")
        val_loss, val_acc = trainer.evaluate(val_loader)
        print(f"[ Val ] Loss: {val_loss:.3f} | Acc: {val_acc * 100:.3f}%")
    else:
        # 인공지능 강의 #6 5장 딥러닝과 텐서플로 참고
        # 텐서플로에서는 model.fit() 메서드가 학습 도중에 발생한 정보를 hist 객체에 저장해 둠
        # hist.history['accuracy']처럼 쓰기만 해도 바로 시각화에 쓸 수 있음
        # 그런데 파이토치에는 그런 기능이 없으므로 따로 딕셔너리 만들어 기록할 필요가 있음
        history = {
            'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []
        }

        # 최고 정확도, 최고 성능 모델 초기화
        # 그리고 최고 성능 모델의 최저 손실 저장용 변수(진짜 최저 손실 기록 아님)
        best_val_acc = 0.0
        best_model = copy.deepcopy(model.state_dict())
        best_model_loss = float('inf')

        min_val_loss = float('inf') # 이게 진짜 조기 종료 판단용 최저 손실 기록
        early_stop_count = 0 # 조기 종료용 카운트 초기화

        try: # 중간에 학습이 끊겨도 모델을 저장할 수 있도록 try문 사용
            # 정해진 횟수만큼 에포크 반복
            for epoch in range(settings.EPOCHS):
                print(f"\n[Epoch {epoch+1}/{settings.EPOCHS} 시작]")

                # 먼저 전체 학습 데이터셋에 대해 한 바퀴 훈련한 뒤 전체 평균 오차와 정확도 출력
                train_loss, train_acc = trainer.train_epoch(train_loader)
                print(f"[Train] Loss: {train_loss:.3f} | Acc: {train_acc * 100:.3f}%")

                # 그 다음 가중치 수정 없이 현재 모델 평가 후 평균 오차와 정확도 출력
                val_loss, val_acc = trainer.evaluate(val_loader)
                print(f"[ Val ] Loss: {val_loss:.3f} | Acc: {val_acc * 100:.3f}%")

                if settings.CAN_USE_SCHEDULER:
                    scheduler.step(val_loss) # val loss를 기준으로 학습률을 조정하겠다는 뜻
     
                history['train_loss'].append(train_loss)
                # 정확도 객체는 파이토치의 텐서 객체로, 그대로 들어가면 충돌 또는 메모리 점유 위험 가능성 있음
                # 따라서 item()을 붙여 순수한 숫자 데이터로 바꿔 넣는 것이 안전
                history['train_acc'].append(train_acc.item())
                history['val_loss'].append(val_loss)
                history['val_acc'].append(val_acc.item())

                # 최고 성능 모델 저장은 정확도가 올랐을 때
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_model_loss = val_loss
                    best_model = copy.deepcopy(model.state_dict()) # 최고 성능 모델 새로 저장

                # 설정한 횟수만큼 연속으로 val loss의 최저 기록을 갱신하지 못하면 조기종료하도록 설정
                if val_loss < min_val_loss:
                    min_val_loss = val_loss
                    early_stop_count = 0
                else:
                    early_stop_count += 1
                    if early_stop_count >= settings.EARLY_STOP_PATIENCE: 
                        print(f"{early_stop_count}회 연속 최저 손실({min_val_loss:.3f}) 갱신 실패, 학습 조기 종료")
                        break  # 이번 모델 학습 반복문 탈출
                    else:
                        print(f"최저 손실({min_val_loss:.3f}) 갱신 실패, 조기 종료 카운트: {early_stop_count} / {settings.EARLY_STOP_PATIENCE}")
        except KeyboardInterrupt: # 학습을 끊는 건 컨트롤 C 눌러서 중간에 끈 경우가 대표적
            print("\n[학습 중 키보드 인터럽트로 중단]")
        except Exception as e:
            print(f"\n[학습 중 오류 발생: {e}]")

        if best_val_acc > 0.0: # 최고 모델의 최고 정확도가 0보다 크면
            model.load_state_dict(best_model)

            # 폴더 생성하기(exist_ok를 켜면 이미 폴더가 있어도 에러 없이 넘어감)
            os.makedirs(settings.SAVED_MODELS_FOLDER_NAME, exist_ok=True) 

            # 학습한 데이터셋 이름, 학습시킨 모델 이름, 이 모델의 최저 손실값, 최고 정확도를 합친 이름을 지닌 pytorch 파일 이름
            save_file_name = f"{settings.DATA_SET_NAME}_{cur_model_name}_loss_{best_model_loss:.3f}_acc_{best_val_acc*100:.2f}.pth"
            save_path = os.path.join(settings.SAVED_MODELS_FOLDER_NAME, save_file_name) # 경로로 합치기

            torch.save(best_model, save_path) # 실제 모델 저장

            # 모델하고 똑같은 이름을 가진 txt 파일에 하이퍼파라미터와 규제 기법 적용 여부 기록
            setting_path = save_path.replace('.pth', '.txt') # 위의 경로에서 확장자만 txt로 대체

            # with open()은 블록이 끝나면 파일을 자동으로 닫아줘서 close 생략 가능한 안전한 함수
            with open(setting_path, "w", encoding="utf-8") as file: # w(쓰기 모드), 인코딩은 한글 안 깨지는 utf-8
                file.write(f"학습률: {settings.LEARNING_RATE}\n") 
                file.write(f"차등 학습률: {f'적용({settings.BASE_LEARNING_RATE_MULTIPLIER})' if settings.CAN_USE_DIFFERENTIAL_LEARNING_RATE else '미적용'}\n")

                file.write(f"드롭아웃 비율: {settings.DROPOUT_RATE}, 가중치 감쇠 정도: {settings.WEIGHT_DECAY}, 레이블 스무딩 비율: {settings.LABEL_SMOOTHING}\n")

                if not settings.CAN_FREEZE_LAYERS:
                    layer_state_string = "미세 조정(전부 unfreeze)"
                elif settings.CAN_GRAD_MORE_LAYERS:
                    can_grad_layers = settings.MODEL_INFO_DICT[cur_model_name][3]
                    # 현재 모델에서 추가 학습 가능, 즉 동결 해제된 레이어 리스트를 가져와 join으로 쉼표 구분하며 항목 나열
                    layer_state_string = f"부분 미세 조정(unfreeze 층: {', '.join(can_grad_layers)})"
                else:
                    layer_state_string = "동결(마지막 레이어만 unfreeze)"
                file.write(f"학습 방식: {layer_state_string}\n")

                if settings.CAN_USE_SCHEDULER:
                    scheduler_state_string = f"적용(factor: {settings.SCHEDULER_FACTOR}, patience: {settings.SCHEDULER_PATIENCE})"
                else:
                    scheduler_state_string = " 미적용"
                file.write(f"학습률 스케줄러: {scheduler_state_string}\n")

                file.write(f"데이터 증대: {'적용' if settings.CAN_USE_AUGMENTATION else '미적용'}\n")
                file.write(f"오버샘플링: {f'적용({settings.SAMPLER_MULTIPLIER}배)' if settings.CAN_USE_OVERSAMPLING else '미적용'}\n")

                file.write("\n===학습 로그===\n")

                # CSV 스타일로 써야 나중에 그래프 그려주는 사이트에서 적용하기 간편
                file.write("Epoch,Train Loss,Train Accuracy,Val Loss,Val Accuracy\n")
                for epo in range(len(history['train_loss'])):
                    t_loss = history['train_loss'][epo]
                    t_acc = history['train_acc'][epo] * 100
                    v_loss = history['val_loss'][epo]
                    v_acc = history['val_acc'][epo] * 100
                    file.write(f"{epo+1},{t_loss:.4f},{t_acc:.2f},{v_loss:.4f},{v_acc:.2f}\n")

            print(f"\n[{cur_model_name} 최고 성능 모델 기록 및 저장]")
            print(f"[ Val ] Loss: {best_model_loss:.3f} | Acc: {best_val_acc * 100:.3f}%")

            if settings.CAN_DRAW_PLOT:
                visualizer.draw_plot(history)
        else: # 최고 모델의 정확도가 0이라면 학습이 정상 진행되지 않은 것
            print("\n[최고 정확도가 0이므로 모델 저장 없이 종료]")        
        
        # 여러 모델을 연속으로 학습시키다 보니 속도 저하 및 메모리 부족 현상 발생
        del best_model # del로 남아 있는 메모리 상 객체 참조를 제거
    del model, optimizer, trainer, train_loader, val_loader

    # GPU도 장치별 명령어로 캐시 메모리 강제 비우기
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps":
        torch.mps.empty_cache()

    print(f"\n--- {cur_model_name} 모델 실험 끝 ---")


