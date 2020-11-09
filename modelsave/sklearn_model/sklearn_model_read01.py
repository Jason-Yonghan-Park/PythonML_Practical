######################## 학습 후 저장 모델 읽어 예측하기

# 필요한 패키지 임포트
from sklearn.datasets import load_digits
import joblib

# 저장된 모델 읽음
model = joblib.load("/modelsave/sklearn_model/randomforest01.pkl")
print('D:/PythonML_Practical/modelsave/sklearn_model/randomforest01.pkl 모델 읽기 완료')

# 손글씨 숫자 이미지 데이터 읽어옮
digits = load_digits()

# 이미 학습된 model -> 예측
pre = model.predict(digits.data)

# 예측 정확도 출력
from sklearn.metrics import accuracy_score
print('정확도: ', accuracy_score(digits.target, pre))

# 이미지 한개를 꺼내와서 예측
nPred = model.predict(digits.images[100].reshape(-1, 64))
nLabel = digits.target[100]
print('nPred: {}, nLabel: {}'.format(nPred, nLabel), end=' - ')
print('정답' if(nPred == nLabel) else '오답')

# 시각화
import matplotlib.pyplot as plt
plt.imshow(digits.images[100], cmap='gray')
plt.text(0.05, 0.05, str(nPred), color='green' if(nPred == nLabel) else 'red')
plt.show()
plt.imsave('D:/PythonML_Practical/modelsave/sklearn_model/num100.jpg', digits.images[100], cmap='gray')




