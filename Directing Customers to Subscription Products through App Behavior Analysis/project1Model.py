# %%
import pandas as pd
import numpy as np
import seaborn as sn
import matplotlib.pyplot as plt
import time

dataset = pd.read_csv('new_appdata10')

# Data Preprocessing

# 1. 반응변수를 독립 특성에서 분리
response = dataset['enrolled']
dataset = dataset.drop(columns = 'enrolled')

# 2. 데이터 세트 분리 - 학습 세트 / 테스트 세트
# %%
# 데이터의 80% : 학습세트 / 데이터의 20% : 테스트 세트
# random_state = 0 : 무작위 선택 이후 다음 실행 시에도 복제될 수 있도록
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(dataset, response, 
                                                    test_size = 0.2,
                                                    random_state = 0)
# %%
# 유저 아이디는 그저 식별 용도이므로 삭제
# 그러나, 마지막에는 예측 값에 대해서 유저와 연결시키는 과정이 필요 -> 다른 곳에 저장 후 삭제
train_identifier = X_train['user']
X_train = X_train.drop(columns = 'user')

test_identifier = X_test['user']
X_test = X_test.drop(columns = 'user')
# %%
# 특성 스케일링 -> 표준 스케일러 함수 활용
# 왜냐, 특정 수치 필드가 그저 큰 절댓값을 갖고 있다는 이유로 모델에 영향을 미치게 하지 않기 위해
# ex) 나이 : 18-100 / 화면 수 : 0,1 => 나이에 가중치 값 부여해서는 안 됨
# -> 따라서, 모든 특성을 정규화 하는 것 -> 모든 특성은 수적인 양이 아닌 상관관계에 따라서만 반응변수에 영향을 줄 수 있게 됨
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
# %%
# fit_transform : 데이터세트를 표준 스케일러에 적합시킨 후 적절한 스케일로 변환
X_train2 = pd.DataFrame(sc_X.fit_transform(X_train))
X_test2 = pd.DataFrame(sc_X.transform(X_test)) # 이미 학습세트에 적합됐으므로 transform만 해주면 됨
# %%
X_train2.columns = X_train.columns.values
X_test2.columns = X_test2.columns.values

# %%
# 원래 학습 및 테스트의 인덱스를 다시 통합시키기
X_train2.index = X_train.index.values
X_test2.index = X_test.index.values
# %%
# 원래 학습 및 테스트 세트를 새로운 학습 및 테스트 세트로 변환
# -> 모든 특성이 정규화과정을 거침
X_train = X_train2
X_test = X_test2
# %%

## Model Building
from sklearn.linear_model import LogisticRegression
# 분류기 지정
# random_state = 0 : 재실행 할 때도 이전과 동일한 결과 출력
# penalty : 정규화된 로지스틱 회귀 모델을 L1('라쏘'파일로 불림) 정규화 모델로 바꿔줌
# -> 반응 변수에 강하게 연관이 되어 있는 특정 필드에 패널티를 부과
classifier = LogisticRegression(random_state = 0, penalty = 'l1', solver= 'liblinear') 
classifier.fit(X_train, y_train)
# %%
y_pred = classifier.predict(X_test)
# %%
# confusion matrix : 예상된 값의 수와 실제 값의 수를 보여주는 표
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score
cm = confusion_matrix(y_test, y_pred)
accuracy_score(y_test, y_pred)
# %%
# 과적합됐는지 확인하기 위해 정밀도도 검사
# 정밀도 : 정확도의 정확도
precision_score(y_test, y_pred)
recall_score(y_test, y_pred)
f1_score(y_test, y_pred)
# %%
# 실제 혼동 행렬 도식화
# 1. 실제 pandas 데이터 프레임에 혼동 행렬을 생성하여 수치 생성
df_cm = pd.DataFrame(cm, index = (0,1), columns = (0,1))
plt.figure(figsize = (10,7))
# 2. Seaborn에 세트를 세팅한 뒤 Heatmap 생성
sn.set(font_scale = 1.4)
sn.heatmap(df_cm, annot = True, fmt = 'g')
print("Test Data Accuracy: %0.4f" %accuracy_score(y_test, y_pred))

# %%
# 예측한 숫자들이 실제로 맞는지 & 과적합 여부 확인 -> K겹 교차 검증
# K겹 교차 검증 : 모델을 학습 세트의 다른 하위 세트에 적용시켜 보는 기법
# 학습 세트를 10겹으로 나눈 후 그 중 9개를 선택한 뒤 남은 하나를 이용해 테스트 -> 1개의 세트를 고르는 이 방법을 반복
from sklearn.model_selection import cross_val_score # -> k겹 교차로 검증한 모든 정확도를 저장
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
# 실행되는 동안 각각의 정확도 출력 - 평균 점수 및 평균 표준편차
print("Logistic Accuracy : %0.3f (+/- %0.3f)" %(accuracies.mean(), accuracies.std() * 2))

# %%
######### Formatting the Final Results #########
# 유저 식별자와 유저의 실제 결과값 배치 , 각각을 열로 생성
final_results = pd.concat([y_test, test_identifier], axis = 1).dropna()
# %%
# 최종 열 생성
final_results['predicted_results'] = y_pred
final_results[['user', 'enrolled', 'predicted_results']].reset_index(drop = True)
# %%
