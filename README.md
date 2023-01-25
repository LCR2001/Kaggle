# Kaggle
- 모델 개선 방법

1. 데이터 정규화(Data Normalization)
- Feature scaling : 통일성 기반 정규화(Unity-based normalization) 사용 -> 0~1 사이의 값으로 데이터를 만듦.
    - (X-X_min) / (X_max - X_min)

2. SVM 파라미터 조정(SVM Parameters Optimization)
- C parameter : 학습 지점의 올바른 분류와 원만한 결정경계 사이의 균형을 제어 -> 모델이 데이터 지점을 오분류할 시 패널티(cost) 적용
    - small C : 오분류에 대해 cost를 적게 부여. soft margin_두 개 클래스 사이에 더 평탄한 결정경계를 얻을 수 있음.
    - large C : 오분류에 대해 cost를 크게. hard margin_두 개 클래스 사이에 복잡한(더 엄격하거나 더 잘 맞는) 경계를 얻을 수 있음. 
        - 학습 데이터 세트에만 특정 ->  과적합에 취약함.
    
- Gamma parameter : 단일 학습 세트의 영향이 미치는 범위를 제어함(데이터 포인트의 영향 범위를 지정)
    - small Gamma : 범위가 넓음. 더 일반화된 솔루션을 가짐.
        - 결정경계만 고려하는 것이 아닌, 훨씬 더 많은 지점(공간)을 고려.
    - large Gamma : 범위가 짧음_초평면(결정경계)에 가까운 값에만 집중
        - 과적화될 위험 높음 <- 초평면에 멀리 떨어진 다른 지점은 무시

- C, Gamma parameter 값 조정 방법: 그리드 검색(Grid search)
## Breast Cancer

- 암 또는 종양을 양성(benign_양성 종양 : 전이가 되지 않는 종양)과 악성(malignant) 여부 판단
- 30가지 특성세트 : 세포핵의 특성 의미
- 30 Features examples:
        - radius (반경)
        - texture (질감)
        - perimeter (둘레)
        - area (면적) 
        - smoothness (local variation in radius lengths)
        - compactness (perimeter^2 / area - 1.0)
        - concavity (severity of concave portions of the contour)
        - concave points (number of concave portions of the contour)
        - symmetry 
        - fractal dimension ("coastline approximation" - 1)

- 데이터셋 -> 선형 분리 가능
- 데이터셋 -> 569개의 인스턴스 존재
- 클래스 분류 -> 악성(Malignant) : 212 , 양성(Benigh) : 357 
- Target class:
         - Malignant
         - Benign

### Improving the Model
1) Feature Scaling_Unity-Based Normalization
- Before
![정규화1_before](https://user-images.githubusercontent.com/91776093/214646717-749f3b02-0bc8-41cc-8fee-61f8a930c44c.png)
- After
![정규화1_after](https://user-images.githubusercontent.com/91776093/214646807-693675eb-7924-4f66-b4a5-522c74fc2424.png)
