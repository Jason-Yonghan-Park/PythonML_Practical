# CNN을 이용한 Fashion MNIST 분류 - 98%
# tensorflow와 tf.keras 임포트
import tensorflow as tf
from tensorflow import keras
 
# 추가로 필요한 라이브러리 임포트
import numpy as np
import matplotlib.pyplot as plt

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

# 2. tf.keras.Sequential을 이용해 모델을 만든다.
# model을 생성하면서 아래와 같이 다양한 레이어를 추가할 수 있다.
# Conv2D 레이어는 컨볼루션 레이어에서 가장 많이 사용되는 레이어로
# 다음과 같은 파라미터를 설정할 수 있다.
# kernel_size : 필터 행렬의 가로 세로 크기
# strides : 필터가 한 스텝에 이동하는 크기, default=(1, 1)
# padding : 컨볼루션 연산 전에 이미지 경계 부분에 0으로 채울지를 
# 여부를 지정하는 속성, valid(경계를 추가하지 않음), same(경계를
# 추가해 출력 이미지의 크기를 입력 이미지의 크기와 같도록 설정) 
# filters  : 필터의 개수, 많을 수록 좋지만 너무 많으면 학습이 속도가 느림
#
# 아래에서 Flatten 레이어 앞 쪽이 특징 추출기(Feature Extractor)
# 그 뒷쪽이 이미지를 분류하는 분류기(Classifier) 부분이다. 
# 특징 추출기(Feature Extractor) 출력된 데이터를 Flatten 레이어를
# 통해서 이미지 분류기(Classifier)의 입력에 맞게 1차원 배열로 변환해 준다. 
# Flatten 레이어는 다 차원 배열을 1차원 배열로 펼쳐서 늘려주는  
# 레이어로 학습에 사용되는 가중치는 없고 데이터만 변환해 준다.
#
# Dense 레이어는 Fully Connected Layer를 구성해 주는 레이어로
# 활성 함수는 주로 성능이 뛰어난 relu를 사용하지만 분류 문제이므로
# 마지막은 소프트맥스를 사용해 10개의 클래스의 각각의 확률을 출력하면 된다.
model = tf.keras.Sequential([
    keras.layers.Conv2D(input_shape=(28, 28, 1), kernel_size=(3, 3), filters=64, padding='same', activation='relu'),
    keras.layers.MaxPool2D(pool_size=(2, 2)),
    keras.layers.Dropout(rate=0.3),
    keras.layers.Conv2D(kernel_size=(3, 3), filters=32, padding='same', activation='relu'),
    keras.layers.MaxPool2D(pool_size=(2, 2)),
    keras.layers.Dropout(rate=0.3),
    keras.layers.Conv2D(kernel_size=(3, 3), filters=32, activation='relu'),
    keras.layers.MaxPool2D(pool_size=(2, 2)),
    keras.layers.Flatten(),
    keras.layers.Dense(units=256, activation=tf.nn.relu),
    keras.layers.Dropout(rate=0.3),
    keras.layers.Dense(units=128, activation=tf.nn.relu),
    keras.layers.Dropout(rate=0.3),
    keras.layers.Dense(units=10, activation=tf.nn.softmax)
])    

# cost function / loss function 설정
# 비용(cost) 최소화 방식은 분류 문제이므로 크로스 엔트로피(Crossentroy)를
# 지정한다. - sparse_categorical_crossentropy 를 지정
# FashinMNIST의 레이블은 원핫 인코딩(one-hot encoding )을 사용하지
# 않고 0 ~ 9 사이의 값으로 레이블링 되어 있으므로 원핫 인코딩이 아닌 데이터를
# 받아서 계산하기 위해서 sparse_categorical_crossentropy를 지정 하였다.
# 이 설정은 희소 행렬을 나타내는 데이터를 별도의 처리 없이 사용할 수 있도록
# 해주는 설정이다. 원핫 인코딩 예) 3을 표현 - [0, 0, 0, 1, 0, 0, 0, 0, 0, 0])
# 참고로 categorical_crossentropy는 원핫 인코딩을 적용하는 설정이다.
#
# compile() 함수는 모델을 이용해 신경망 학습 과정을 설정하는 함수 이다.
# metrics에 accuracy를 지정하면 매 epoch 당 정확도를 출력해 준다.
# 옵티마이저를 별도로 생성해 학습률(learning rate) 등을 따로 지정할 수도
# 있고 tf.keras.optimizers.Adam()으로 지정하 수 있다. 또한 아래와 같이
# 문자열로 "adam"을 지정할 수도 있다. 학습률의 기본 값은 0.001 이다.
model.compile(loss="sparse_categorical_crossentropy", 
              optimizer="adam", metrics=["accuracy"])

# 훈련 동안 체크포인트 저장할 path -> 파일 이름에 epoch 지정
import os
checkpoint_path = 'D:/PythonML_Practical/modelsave/tensorflow_model02/cp-{epoch:04d}.ckpt'
checkpoint_dir = os.path.dirname(checkpoint_path)

# 체크포인트 콜백 만들기
cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path, save_weights_only=True, verbose=1, period=3)

# 1st 체크포인트 파일 명
model.save_weights(checkpoint_path.format(epoch=0))

#model.summary()
# fit() 함수를 이용해 모델을 학습 시킨다.
# epochs : 훈련은 epoch로 구성되며 1epoch은 전체 입력 데이터를 한 번
# 학습하는 것을 의미, batch_size : 한 번에 학습할 사이즈를 지정
# validation_split : 학습 데이터에서 검증 데이터로 사용할 비율을 지정
# 각 epoch의 학습 결과 출력에 loss, accuracy와 같이 검증 데이터의 
# val_loss와 val_accuracy가 같이 출력 된다.
history = model.fit(train_X, train_y, epochs=10, validation_split=0.25, callbacks=[cp_callback])

print('모델 저장 완료: {}'.format('D:/PythonML_Practical/modelsave/tensorflow_model02 폴더에 체크포인트 저장 완료'))