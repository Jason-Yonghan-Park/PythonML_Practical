# CNN을 이용한 Fashion MNIST 분류 - 98%
# tensorflow와 tf.keras 임포트
import tensorflow as tf
from tensorflow import keras
 
# 추가로 필요한 라이브러리 임포트
import numpy as np
import matplotlib.pyplot as plt

# 학습 완료 후 저장된 모델 읽어옴
model = keras.models.load_model('D:\PythonML_Practical\modelsave/tensorflow_model01/fashionmnist_cnn01.h5')

# 패션 MNIST는 keras.datasets 패키지에서 읽어 적재할 수 있다.
# load_data() 함수를 호출하면 네 개의 넘파이(NumPy) 배열이 반환된다.
# train_X와 train_y 배열은 모델 학습에 사용되는 training set
# test_X와 test_y 배열은 모델 테스트에 사용되는 test set
fasion_mnist = keras.datasets.fashion_mnist
(train_X, train_y), (test_X, test_y) = fasion_mnist.load_data()

# 학습 데이터 60,000개, 테스트 데이터 10,000개, 클래스는 0 ~ 9까지 10개
print(train_X.shape, test_X.shape)
print(set(train_y))
"""
# 첫 번째 이미지를 흑백으로 화면에 출력하면서 그 옆에 색상 바를
# 같이 출력해 보면 이미지의 각 픽셀은 0 ~ 255까지의 값을 가지는
# 28 x 28 = 768 픽셀의 이미지 데이터인 것을 확인 할 수 있다.
plt.imshow(train_X[0], cmap="gray")
plt.colorbar()
plt.show()
"""
# 신경망 모델에 주입하기 전에 픽셀 값의 범위를 0~1 사이로 조정해 정규화 한다.
# training set와 test set 둘 다 255로 나누어 정규화(normalize) 한다.
train_X = train_X / 255.0
test_X = test_X / 255.0

# Fashion MNIST 데이터는 흑백 이미지로 색상에 대한 1개 채널을 갖기
# 때문에 reshape() 함수를 사용해 데이터의 가장 뒤 쪽에 채널에 대한 차원
# 정보를 추가 한단. 데이터 수는 달라지지 않지만 차원이 다음과 같이 바뀐다.
# (60000, 28, 28) -> (60000, 28, 28, 1)
print("reshape 이전 : ", train_X.shape, test_X.shape, train_X[0].shape)
train_X = train_X.reshape(-1, 28, 28, 1)
test_X = test_X.reshape(-1, 28, 28, 1)
print("reshape 이후 : ", train_X.shape, test_X.shape, train_X[0].shape)

# 읽어온 모델 성능 평가
test_loss, test_acc = model.evaluate(test_X, test_y)
print('eval acc: {}'.format(test_acc))

# 테스트 데이터 사용하여 예측
pred = model.predict(test_X)
print(np.argmax(pred[7]))
print([round(p, 4) for p in pred[7]])