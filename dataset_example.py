# Use dataset provided
from sklearn.datasets import load_diabetes
import numpy as np
import matplotlib.pyplot as plt
plt.style.use(['seaborn-whitegrid'])

diabetes = load_diabetes()
# dictionary type

print(diabetes.DESCR)  # describe the diabetes dataset.
print("")
print(diabetes.feature_names)
print(diabetes.data_filename)
print(diabetes.target_filename)
print("")

# model_selection module
# 학습용 데이터와 테스트 데이터로 분리, 교차검증 분할 및 평가,
# Estimator의 하이퍼 파라미터 튜닝을 위한 다양한 함수와 클래스 제공

# train_test_split() : 학습/테스트 세트 분리
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes

diabetes = load_diabetes()
# train_test_split returns 4 data sets as follows
X_train, X_test, y_train, y_test = train_test_split(diabetes.data, diabetes.target, test_size=0.3)
# train data : 70%, test data: 30%
# 실행할때마다 비율대로 random으로 분류해 줌.

model = LinearRegression()
model.fit(X_train, y_train)

print("학습 데이터 점수 : {}".format(model.score(X_train, y_train)))
print("평가 데이터 점수 : {}".format(model.score(X_test, y_test)))

predicted = model.predict(X_test)
expected = y_test

plt.figure(figsize=(8, 4))
plt.scatter(expected, predicted)
plt.plot([30, 350], [30, 350], '--r')
plt.tight_layout()
plt.show()

# cross_val_score() : 교차검증
from sklearn.model_selection import cross_val_score, cross_validate

scores = cross_val_score(model, diabetes.data, diabetes.target, cv=5)  # cross validation을 위하여 5개 선택

print("교차 검증 정확도 : {}".format(scores))
print("교차 검증 정확도 : {} +/- {}".format(np.mean(scores), np.std(scores)))
print("1")
