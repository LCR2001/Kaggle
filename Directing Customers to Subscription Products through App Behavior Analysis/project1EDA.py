#%%
import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
from dateutil import parser

dataset = pd.read_csv('C:/Users/CHAERIN/Desktop/Inflearn_ML/P39-CS3-Data/appdata10.csv')

## EDA 과정
# hour 열에 관한 내용은 출력X -> 왜냐, 문자열 형태이기 때문
dataset.head()
dataset.describe()
# %%
## Data Cleaning
dataset['hour'] = dataset.hour.str.slice(1,3).astype(int)

# %%
## Plotting
# 관련 없는 데이터 제거 후 새롭게 데이터 세트 생성
dataset2 = dataset.copy().drop(columns = ['user', 'screen_list','enrolled_date', 'first_open', 
'enrolled'])

# %%
dataset2.head()
# %%
# Histograms -> 독립적인 특성 중 수치적 특성 분포 파악에 용이
# 데이터세트의 분포에 익숙해지기 위해 파악해야함
plt.suptitle('Histograms of Numerical Columns', fontsize = 10)
for i in range(1, dataset2.shape[1]+1):
    plt.subplot(3,3,i)    # 이미지가 몇 차원이어야 하는지 알려주기
    f = plt.gca()   # 정리 후 필드 생성
    f.set_title(dataset2.columns.values[i-1])

    vals = np.size(dataset2.iloc[:, i-1].unique())  # 전체 열을 쿼리 , unique 함수로 고유값을 가져온 후 이를 size함수에 넣어 나온 고유 값의 크기를 넣어주기

    plt.hist(dataset2.iloc[:, i-1], bins = vals, color = '#3F5D7D')

# %%
# Correlation with Response-> 수치적 특성 분석 및 반응변수 간의 상관관계 살피기
# corrwith() : 데이터 프레임의 각 필드와 반응변수(함수 작성 시 지정해줌) 간의 상관관계 값을 리스트로 반환 
# rot : X축 레이블을 몇 도 정도 회전할 것인지 , gird = True : 그래프 내 격자 유무 설정
dataset2.corrwith(dataset.enrolled).plot.bar(figsize = (20,10), 
                                             title = 'Correlation with Response Variable', 
                                             fontsize = 15, rot = 45, grid = True)
# %%
# Correlation Matrix(상관 행렬) : 각 개별 필드 사이의 상관 관계를 제공(반응변수와의 상관관계X)
# -> 각 필드 사이의 선형적인 관계를 알 수 있음
# 머신러닝 모델 구축 시 전제 가정 : 특성이 독립변수 = 즉, 서로 독립적인 관계
# 인자를 외울 필요X. 인터넷에서 찾아서 그때끄때 쓰기O

# 상관 행렬 만들기 시작 
# 배경 만들기
sn.set(style="white", font_scale = 2)

# 상관 행렬 계산 및 2차원 배열 생성
corr = dataset2.corr()

# 상삼각행렬을 위한 틀 생성
mask = np.zeros_like(corr, dtype = np.bool_)
mask[np.triu_indices_from(mask)] = True

# matplotlib 세팅
f, ax = plt.subplots(figsize = (18,15))
f.suptitle("Correlation Matrix", fontsize = 40)

# Generate a custom diverging colormap
cmap = sn.diverging_palette(220, 10, as_cmap=True)

# 히트맵 생성 (상관행렬은 히트맵임)
sn.heatmap(corr, mask = mask, cmap = cmap, vmax = 1, center = 0, 
           square = True, linewidths = 5, cbar_kws={"shrink": .5})
# %%
## 변수 가공 - 반응변수 미세 조정
# ex) 등록 기간을 1주간만 두기
# 목표 : 처음 앱을 연 날짜와 등록 날짜 사이의 시간 차이 분포 도식화
dataset.dtypes
# %%
# 모든 것을 datetime 객체로 변환시키기
# -> 문자열인 행에만 util함수의 parser함수 적용 -> if isinstance / else
dataset["first_open"] = [parser.parse(row_data) for row_data in dataset["first_open"]]
dataset["enrolled_date"] = [parser.parse(row_data) if isinstance(row_data, str) else row_data for row_data in dataset["enrolled_date"]]
# %%
dataset.dtypes
# %%
# 두 날짜의 차이를 시간 단위로 계산
# -> timedelta 자료형으로 변환 : timedelta64[h] : 시간 단위로 변환하라는 의미
dataset["difference"] = (dataset.enrolled_date - dataset.first_open).astype('timedelta64[h]')
# dataset의 'difference'값 도식화하기
# -> dropna() : Na값 없애기 / 색상 추가
plt.hist(dataset['difference'].dropna(), color = '#3F5D7D')
plt.title("Distribution of Time-Since-Enrolled")
plt.show()
# %%
# 위의 과정을 통해 대부분의 사람들이 100시간 내에 등록하는 것을 확인했으므로 범위를 좁혀서 확인 : range = [0,100]
plt.hist(dataset['difference'].dropna(), color = '#3F5D7D', range = [0,100])
plt.title("Distribution of Time-Since-Enrolled")
plt.show()
# %%
# 위의 과정을 통해 대부분의 사람들이 50시간 내에 등록하는 것을 확인했으므로 범위를 좁혀서 확인 : range = [0,50]
plt.hist(dataset['difference'].dropna(), color = '#3F5D7D', range = [0,50])
plt.title("Distribution of Time-Since-Enrolled")
plt.show()
# %%
# 반응변수 값이 48보다 크면 0으로 셋팅 
# -> 왜냐, 실제로 등록한 사람일지라도 48시간 제한 시간을 초과할 시 미등록 처리로 넘기기 위해
dataset.loc[dataset.difference > 48, 'enrolled'] = 0

#더이상 필요하지 않은 칼럼들 제거
dataset = dataset.drop(columns = ['difference', 'enrolled_date', 'first_open'])
# %%
## Formatting the screen_list Field : 변수 가공 - 스크린
top_screens = pd.read_csv('C:/Users/CHAERIN/Desktop/Inflearn_ML/P39-CS3-Data/top_screens.csv').top_screens.values
# %%
# 인기있는 화면에 대한 열 생성 후 이를 설명하는 다른 열 생성

# 화면에서 각각의 작은 벡터 목록으로 분리하여 각 화면을 매핑
# 화면보다 콤마 개수가 늘 항상 한 개 모자르므로 한 개 따로 추가
dataset['screen_list'] = dataset.screen_list.astype(str) + ','
# 각 상위 화면에 대한 열 생성을 위해 for루프 활용
for sc in top_screens:
    # 해당 화면 이름으로 새로운 열 생성 - 문자열로 바꿔준 후(str), 그 화면 이름이 top_screen을 포함하는지 확인(contains(sc)) : bool형으로 반환
    # -> 정수로 변환
    # 이후, screen_list에서 해당 화면 제거
    dataset[sc] = dataset.screen_list.str.contains(sc).astype(int)
    dataset['screen_list'] = dataset.screen_list.str.replace(sc + ',', "")
# %%
dataset['Other'] = dataset.screen_list.str.count(",")
# %%
dataset = dataset.drop(columns = ['screen_list'])
# %%
# Funnels : 동일한 세트에 속하는 화면 그룹
# 상관 관계가 있는 그룹은 모델 구축에 있어서 도움이 되지X 
# -> 해당 상관관계 제거 후 해당 화면의 가치 유지를 위해 모든 화면을 하나의 퍼널로 그룹화
# -> 하나의 퍼널에 속할 경우, 포함된 화면 수로 열이 생성됨. 
# -> 상관관계를 제거해도 데이터는 유지됨

# 퍼널에 속하는 모든 화면의 리스트 생성
savings_screens = ['Saving1', 'Saving2', 'Saving2Amount', 'Saving4',
                   'Saving5', 'Saving6', 'Saving7', 'Saving8',
                   'Saving9', 'Saving10']

# %%
# saving_screen 개수를 포함하는 새로운 열 생성
# axis =  0 : 행 방향(가로) / axis = 1 : 열 방향(세로)
dataset['SavingsCount'] = dataset[savings_screens].sum(axis = 1)
dataset = dataset.drop(columns = savings_screens)

cm_screens = ["Credit1","Credit2","Credit3","Credit3Container","Credit3Dashboard"]
dataset['CMCount'] = dataset[cm_screens].sum(axis = 1)
dataset = dataset.drop(columns = cm_screens)

cc_screens = ["CC1", "CC1Category", "CC3"]
dataset["CCCount"] = dataset[cc_screens].sum(axis = 1)
dataset = dataset.drop(columns = cc_screens)

loan_screens = ["Loan", "Loan2", "Loan3", "Loan4"]
dataset["LoansCount"] = dataset[loan_screens].sum(axis = 1)
dataset = dataset.drop(columns = loan_screens)

# %%
dataset.head()
# %%
dataset.columns

# %%
# 자체 데이터 세트에 저장 - csv파일 형태로
# index = False : 기존의 인덱스 무시
dataset.to_csv("new_appdata10", index = False)
# EDA - 전처리 과정 끝
# %%
