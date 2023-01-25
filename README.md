# Kaggle
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
