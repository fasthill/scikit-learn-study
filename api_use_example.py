# Estimator api
import numpy as np
import matplotlib.pyplot as plt
plt.style.use(['seaborn-whitegrid'])

x = 10 * np.random.rand(50)
y = 2 * x + np.random.rand(50)
plt.scatter(x, y)
plt.tight_layout()
plt.show()

# 1. 적절한 estimator 클래스를 import해서 모델의 class 선택
from sklearn.linear_model import LinearRegression

# 2. class를 원하는 값으로 인스턴스화해서 모델의 하이퍼파라미터 선택
model = LinearRegression(fit_intercept=True)

# 3. 데이터를 특징 배열과 대상 배열로 배치
X = x[:, np.newaxis]

# 4. 모델 인스턴스의 fit() 메서드를 호출해 모델을 데이터에 적합
model.fit(X, y)
# model.coef_
# model.intercept_

# 5. 모델을 새 데이터에 대하여 적용
xfit = np.linspace(-1, 11)
Xfit = xfit[:, np.newaxis]
yfit = model.predict(Xfit)

plt.scatter(x, y)
plt.plot(xfit, yfit, '--r')
plt.show()
