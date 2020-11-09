# 손 글씨 이미지 숫자 분류 및 히트 맵 시각화  - 정확도 97.7
# digits 데이터를 랜덤 포레스트를 이용한 분류
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

# 손 글씨 숫자 이미지 데이터를 읽어온다.
digits = load_digits()

# 1. 랜덤 포레스트 모델 클래스 선택
from sklearn.ensemble import RandomForestClassifier

# 2. 모델 클래스의 인스턴스를 생성 - 의사결정 추정기 1000개 지정
model = RandomForestClassifier(n_estimators=1000)     

# 3. 데이터를 2차원 구조(data)와 1차원 벡터(label)로 구성해 배치
# 데이터를 학습 데이터와 테스트 데이터로 나눈다.
# 이때 이미지 데이터도 같이 나눠야 뒤에서 시각화에 사용할 수 있다.
Xtrain, Xtest, ytrain, ytest, trainImg, testImg =\
    train_test_split(digits.data, digits.target, digits.images, 
                     random_state=0)

# 4. 모델에 데이터를 적합시켜 학습 시킨다.
model.fit(Xtrain, ytrain)

# 학습 완료 -> 모델 저장
#### sklearn.externals 중 joblib 사용
import joblib
joblib.dump(model, '/modelsave/sklearn_model/randomforest01.pkl', compress = True)
print('D:/PythonML_Practical/modelsave/sklearn_model/randomforest01.pkl 모델 저장 완료')

# 5. 새로운 데이터에 모델을 적용시킨다.
pre = model.predict(Xtest)

# 예측결과 정확도 출력
# accuracy(정확도) : 전체 입력 데이터에서 모델이 정확하게 예측한 비율
from sklearn.metrics import accuracy_score
print("정확도 : ", accuracy_score(ytest, pre))

# 랜덤 포레스트의 분류 결과 모델 성능 평가 보고서 출력
# precision(정밀도, 적합률) : 예측 값 중에서 모델이 정확하게 예측한 비율
# recall(재현률) : 실제 값 중에서 모델이 정확하게 예측한 비율 
# F 값 : 정밀도와 재현율의 조화평균, 두 가지를 종합적으로 볼 때 사용
from sklearn import metrics
print(metrics.classification_report(pre, ytest))

# 현재 시스템에 설치되어 있는 맑은 고딕 폰트를 설정한다.
# 탐색기에서 Windows/fonts/ 폴더로 이동해 폰트의 영문 이름 확인
from matplotlib import font_manager, rc
fontLocation = "C:/Windows/fonts/malgun.ttf"

# matplotlib 패키지는 폰트 관리를 위해 font_manager와 리소스
# 관리를 위해서 rc를 제공하고 있다. 한글 폰트 파일 명을 읽어온 후 
# 리소스에 할당하면 지정한 폰트를 사용할 수 있다.
fontName = font_manager.FontProperties(fname=fontLocation).get_name()
rc("font", family=fontName)

# 오차 행렬(Confusion Matrix) 생성
from sklearn.metrics import confusion_matrix
mat = confusion_matrix(ytest, pre)

# seaborn의 히트맵 생성
import matplotlib.pyplot as plt
import seaborn as sns
sns.heatmap(mat, square=True, annot=True, cbar=False)
plt.xlabel("실제 데이터")
plt.ylabel("예측 데이터")
plt.show()


