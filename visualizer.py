import matplotlib.pyplot as plt

def draw_plot(history):
    """ 인공지능 강의 #6 5장 딥러닝과 텐서플로를 참고
    결과를 매번 Gemini에게 넣어서 그래프를 그려달라고 하니 시간이 걸림
    그래서 넣은 임시 그래프 그리기 기능(설정에서켜고 끌 수 있음)
    """
    
    # 손실 함수 곡선
    plt.plot(history['train_loss'])
    plt.plot(history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')

    # 손실은 정해진 최대값이 없지만 우선 3.0을 y축 최고점으로 고정
    plt.ylim(0.0, 3.0)

    plt.legend(['Train', 'Validation'], loc='upper right')
    plt.grid()
    plt.show()

    # 정확률 곡선
    plt.plot(history['train_acc'])
    plt.plot(history['val_acc'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')

    # y축 범위를 0.0에서 1.0으로 고정
    # 이렇게 안 하면 할 때마다 y축 최고점 위치 바뀌어서 정확한 비교가 안 됨
    plt.ylim(0.0, 1.0)

    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.grid()
    plt.show()