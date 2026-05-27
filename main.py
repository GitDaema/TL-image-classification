from data_manager import DataManager
from model_factory import ModelFactory
from trainer import Trainer
import settings
import visualizer

import torch
import torch.nn as nn
import torch.optim as optim

import copy
import os

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

print(f"이미지: {data_manager._data_size} 장, 클래스 개수: {data_manager.num_classes} 개")
print(f"학습률: {settings.LEARNING_RATE}")
print(f"드롭아웃 비율: {settings.DROPOUT_RATE}, 가중치 감쇠 정도: {settings.WEIGHT_DECAY}")
print(f"레이블 스무딩 비율: {settings.LABEL_SMOOTHING}")
if not settings.CAN_FREEZE_LAYERS:
    layer_state_string = "미세 조정(전부 unfreeze)"
elif settings.CAN_GRAD_MORE_LAYERS:
    layer_state_string = "부분 미세 조정(몇몇 지정 레이어들만 unfreeze)"
else:
    layer_state_string = "동결(마지막 레이어만 unfreeze)"
print(f"학습 방식: {layer_state_string}")

if settings.CAN_USE_SCHEDULER:
    print(f"학습률 스케줄러: 적용(factor: {settings.SCHEDULER_FACTOR}, patience: {settings.SCHEDULER_PATIENCE})")
else:
    print(f"학습률 스케줄러: 미적용")

print(f"데이터 증대: {'적용' if settings.CAN_USE_AUGMENTATION else '미적용'}")

for cur_model_name in settings.MODEL_NAME_LIST:
    print(f"\n--- {cur_model_name} 모델 실험 시작 ---")
    
    # 모델과 전처리 규칙을 가져옴
    model, train_transform, val_transform = ModelFactory.create_model_and_transforms(cur_model_name, data_manager.num_classes)
    
    # 가져온 전처리 규칙으로 데이터를 모델 학습용 및 평가용으로 가공한 객체인 로더를 가져옴
    train_loader, val_loader = data_manager.get_loaders(train_transform, val_transform)

    # 손실 함수는 획득한 결과와 실제 값 사이의 틀린 정도를 측정하는 함수
    # 학습 중에 이 값을 최소화하려고 하며, 예측과 정답을 비교해 손실을 계산
    # 모델 매개변수 최적화하기 - 파이토치 한국어 튜토리얼
    # https://tutorials.pytorch.kr/beginner/basics/optimization_tutorial.html

    criterion = nn.CrossEntropyLoss(label_smoothing=settings.LABEL_SMOOTHING)

    # 옵티마이저는 손실 함수의 최저점을 찾아주는 탐색기

    # 가중치 재학습 여부를 False로 한 것들은 옵티마이저에 넣을 필요가 없음
    # 따라서 실제로 학습할 파라미터들만 모으는 리스트를 따로 생성
    # 설정에 따라 마지막 레이어만 들어가거나, 추가로 지정한 레이어들이 함께 포함됨
    param_list = [] 

    for param in model.parameters():
        if param.requires_grad == True:
            param_list.append(param) 
    
    # 컴퓨터 비전을 위한 전이 학습 튜토리얼에서는 옵티마이저로 SGD 사용
    # SGD 옵티마이저에 재학습 가능한 파라미터만 있는 리스트, 학습률, 모멘텀(관성) 전달해 객체 생성
    # optimizer = optim.SGD(param_list, lr=settings.LEARNING_RATE, momentum=settings.OPTIM_MOMENTUM,
    #                       weight_decay=settings.WEIGHT_DECAY)
    
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
        # 최저 val 손실, 최고 성능 모델 초기화
        # 그리고 최고 성능 모델의 val 정확도 저장용 변수(최고 val 정확도 기록 아님)
        best_val_loss = float('inf')
        best_model = copy.deepcopy(model.state_dict())
        best_model_acc = 0.0

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

                if settings.CAN_DRAW_PLOT:
                    history['train_loss'].append(train_loss)
                    # 정확도 객체는 파이토치의 텐서 객체로, 그대로 들어가면 충돌 또는 메모리 점유 위험 가능성 있음
                    # 따라서 item()을 붙여 순수한 숫자 데이터로 바꿔 넣는 것이 안전
                    history['train_acc'].append(train_acc.item())
                    history['val_loss'].append(val_loss)
                    history['val_acc'].append(val_acc.item())

                # 설정한 횟수만큼 연속으로 val loss의 최저 기록을 갱신하지 못하면 조기종료하도록 설정
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    early_stop_count = 0
                    best_model_acc = val_acc
                    best_model = copy.deepcopy(model.state_dict()) # 최고 성능 모델 새로 저장
                else:
                    early_stop_count += 1
                    if early_stop_count >= settings.EARLY_STOP_PATIENCE: 
                        print(f"{early_stop_count}회 연속 최저 손실({best_val_loss:.3f}) 갱신 실패, 학습 조기 종료")
                        break  # 이번 모델 학습 반복문 탈출
                    else:
                        print(f"최저 손실({best_val_loss:.3f}) 갱신 실패, 조기 종료 카운트: {early_stop_count} / {settings.EARLY_STOP_PATIENCE}")
        except KeyboardInterrupt: # 학습을 끊는 건 컨트롤 C 눌러서 중간에 끈 경우가 대표적
            print("\n[학습 중 키보드 인터럽트로 중단]")
        except Exception as e:
            print(f"\n[학습 중 오류 발생: {e}]")

        if best_val_loss != float('inf'): # 최고 모델의 최저 손실이 초기화값(무한대)가 아니면
            model.load_state_dict(best_model)

            # 폴더 생성하기(exist_ok를 켜면 이미 폴더가 있어도 에러 없이 넘어감)
            os.makedirs(settings.SAVED_MODELS_FOLDER_NAME, exist_ok=True) 

            # 학습한 데이터셋 이름, 학습시킨 모델 이름, 최저 손실값, 이 모델의 정확도를 합친 이름을 지닌 pytorch 파일 이름
            save_file_name = f"{settings.DATA_SET_NAME}_{cur_model_name}_loss_{best_val_loss:.3f}_acc_{best_model_acc*100:.2f}.pth"
            save_path = os.path.join(settings.SAVED_MODELS_FOLDER_NAME, save_file_name) # 경로로 합치기

            torch.save(best_model, save_path) # 실제 모델 저장

            print(f"\n[{cur_model_name} 최고 성능 모델 기록 및 저장]")
            print(f"[ Val ] Loss: {best_val_loss:.3f} | Acc: {best_model_acc * 100:.3f}%")

            if settings.CAN_DRAW_PLOT:
                visualizer.draw_plot(history)
        else: # 최저 손실이 초기값(무한대)라면, 아예 학습이 1에포크도 안 된 것
            print("\n[최저 손실이 무한대이므로 모델 저장 없이 종료]")        

    print(f"\n--- {cur_model_name} 모델 실험 끝 ---")


