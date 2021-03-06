## 학습조직 session #1

CH5/CH8/CH9  부분 학습 진행

### CH5 SVM

svm이란, 결정경계를 사용하여 데이터 사이를 분류하는 기법으로  마진을 최대화 하는 결정면을 찾는 것이 그 목적임. 결정경계 자체가 데이터 전체를 사용하여 구성되는 것이 아닌, 서포트벡터만을 활용하여 결정되기 때문에 sparse한 데이터를 지니고 있을때 유용함

가장 많이 사용되는 SVM은 RBF커널을 사용한 SVM이며, 해당 커널을 제대로 사용하기 위해서는 분석가가 하이퍼파라미터 gamma 값과 C값을 지정 해 주어야 함. 감마값과 씨 값은 간단하게 과적합과 관련이 있는 매개변수인데, **해당 값 모두 값이 커질수록 과적합의 가능성이 커지고, 작아질수록 과소적합의 위험성이 커지게 됨**

좀 더 자세하게 살펴보면, 데이터가 초평면에 의해 100% 분리된다고 한다면 해당 하이퍼파라미터가 별 의미가 없엇을테지만 대부분의 데이터는 그렇게 완벽하게 분리가 되는것이 불가능함

![img](https://t1.daumcdn.net/cfile/tistory/997D103359E9E0F323)

예를들어, 위와 같이 아웃라이어가 포함 되어 있는 데이터 같은 경우가 완벽하게 분리되지 못하는 경우에 해당됨.

이를 해결하기 위해서 약간의 오차를 허용하는 전략이 생성되게 되었는데,  이것과 관련된 파라미터가 바로 **cost(C)**임. **C는 얼마나 많은 데이터 샘플이 다른 클래스에 놓이는 것을 허용하는지를 결정함.**  작을 수록 많이 허용하고, 클 수록 적게 허용한다. 다른 말로, C값을 낮게 설정하면 이상치들이 있을 가능성을 크게 잡아 일반적인 결정 경계를 찾아내고, 높게 설정하면 반대로 이상치의 존재 가능성을 작게 봐서 좀 더 세심하게 결정 경계를 찾아내게 됨.

gamma 값은 RBF커널의 최적화와 관련된 매개변수로, **gamma값은 **하나의 데이터 샘플이 영향력을 행사하는 거리를 결정함. gamma는 가우시안 함수의 표준편차와 관련되어 있는데, 클수록 작은 표준편차를 갖는다. 즉, gamma가 클수록 한 데이터 포인터들이 영향력을 행사하는 거리가 짧아지는 반면, gamma가 낮을수록 커진다. 

 gamma 값이 커지게 되면  결정 경계가 결정 경계 가까이에 있는 데이터 샘플들에 영향을 크게 받기 때문에 점점 더 구불구불해지는 양상을 띄게 됨. 즉, **gamma 매개변수는 결정 경계의 곡률을 조정한다고 말할 수 있음.**



### CH8 차원축소

실제 분석가가 사용하게 되는 로 데이터에서, 대부분의 설명변수들은 실제 타겟변수와는 크게 관계가 없는 경우가 대단히 많음. 이러한 경우에서 원 변수를 그대로 사용하여 모델링을 진행하게 되면 과적합의 가능성 역시 높아지며, 예측오류 역시 높아질 가능성이 있음. 또한, 직관적으로 타겟 변수와 직접적으로 관련이 있지는 않으나 다른 설명변수들과의 결합하여 타겟변수에 영향을 미치는 latent factor 가 데이터 내에 존재할 가능성 역시 있음. 

데이터 차원 축소를 진행할때 가장 자주 사용하는 알고리즘은 다음과 같음.

1. PCA

   분산을 최대화 시키는 축을 계속해서 찾아 나가며 데이터의 주성분을 추출, 데이터의 차원을 축소시키는 기법으로, 가장 흔하게 사용됨. 분산 설명력을 이용하여 적절한 개수의 pc를 택할수 있으며, 일반적으로 eigen value가 1보다 큰 값을 유의한 pc라고 택하게 됨. 

   ![img](https://t1.daumcdn.net/cfile/tistory/2306BA4D594E8DD427?download)

   단, pca는 위의 데이터처럼 non-linear한 데이터에 대해서 차원 축소를 진행할때, 데이터의 형태를 반영하지 못하게 될 가능성이 높음. 각 색깔을 클래스라고 할 때 PCA로 차원을 축소하게 되면 분산이 최대 큰 방향으로 축소를 하게 될 것이고,  각 클래스가 섞이게 될 가능성이 높음. 차원축소라는 개념이 차원을 줄이면서도 이전의 속성과 모습을 최대한 유지하려는 것이 좋기 때문에, 이렇게 Nonlinear한 데이터의 형태일 경우에는 pca말고 다른 방법론을 사용하는 것이 좋으며, 추후 논의할 lle나 tsne등이 그 대표적인 예시임.

2. LLE(locally linear embedding)

   로컬 선형 임베딩(Local Linear Embedding)은 고차원의 공간에서 인접해 있는 데이터들 사이의 선형적 구조를 보존하면서 저차원으로 임베딩하는 방법론이며,  다음과 같은 장점을 지니고 있으며 주로 이미지 데이터의 차원 축소에 활용됨.

   1. 사용하기에 간단하다.
   2. 최적화가 국소최소점으로 가지 않는다.
   3. 비선형 임베딩 생성이 가능하다.
   4. 고차원의 데이터를 저차원의 데이터로 매핑이 가능하다.

3. t-SNE

   t-SNE의 아이디어는 데이터 포인트 사이의 거리를 가장 잘 보존하는 2차원 표현을 찾는것임. 
   먼저, 각 데이터 포인트를 2차원에 무작위로 표현한 후 원본 특성 공간에서 가까운 포인트는 가깝게, 멀리 떨어진 포인트는 멀어지게 만듦. 또한, t-SNE는 가까이 있는 포인트에 더 많은 비중을 두어 이웃 데이터 포인트에 대한 정보를 보존하려하는 특성을 가짐.

   이러한 특성 때문에 tsne는 차원을 축소하여 변수를 추출하는 목적보다는, 데이터를 2차원 또는 3차원의 저차원 데이터로 시각화하여 데이터 분석 과정에서 활용하는데 주로 이용됨. 

   그러나, tsne는 pca와 비교하여 계산이 오래걸리며 차원축소가 오직 2차원 내지 3차원으로만 가능하기 때문에 초 고차원의 데이터를 축소하는데는 적합하지 않다라는 단점이 존재함. 이와 같은 단점을 해결 한 것이 이후 논의할 umap임.

4. Umap

   Umap은 TSNE가 가진 단점을 해결한 차원 축소 방법론이며, 다음과 같은 장점을 가짐

   1. 연산 속도가 빠름
   2. 일반화된 embedding 차원: 시각화 용도인 t-SNE와 다르게 UMAP은 embedding 차원의 크기에 대한 제한이 없기 때문에, 일반적인 차원 축소 알고리즘으로 적용 가능함. (게다가 전체적인 구조를 더 잘 반영하기 때문에 시각화도 더 예쁘게 된다.)
   3. Global structure: 전체적인 manifold 구조를 더 잘 보존함.
   4. 탄탄한 이론적 배경: 리만 기하학과 위상 수학에 기반함(이해포기함)
   5. 코드로 바로 사용 가능하게 API로 존재함.(사용이 간편함)

   Umap은 tsne와 다르게 차원 축소의 크기 제한이 없기 때문에, 단순 시각화 뿐만 아니라 pca처럼 latent factor를 추출하여 새로운 변수로 활용하는데에도 사용 될 수 있음. 또한, tsne에는 존재하지 않는 n_components 라는 항목 역시 존재해 근처 몇개의 데이터를 한 군집으로 묶을 것인지를 설정 할 수 있기 때문에 데이터를 시각화 하는데에도 용이함.

   

### CH9 비지도 학습

위 챕터에선, 비지도 학습(주로 클러스터링)에 대해 학습함. 비지도 학습은 y라벨 없이 모델링을 진행하며 주로 이상감지 부분에 활용됨. 책에서는 K-means 도 있으나 이미 많이 아는 내용이므로 이 부분은 스킵하고 DBSCAN(+HDBSCAN) 과 GMM에 대해 설명을 진행함.

1. DBSCAN

   K means 가 거리를 기반으로 클러스터링을 진행하는 방법론이라면, DBSCAN은 데이터의 밀도를 기반으로 클러스터링을 진행하는 방법론임. 즉, 어느점을 기준으로 반경 x 내에 점이 n개 이상 있으면 하나의 군집으로 인식하는 방식임.

   dbscan은 kmeans와 비교하여 군집의 개수를 사전에 지정 할 필요가 없다는 장점을 가지고 있으나, 하이퍼파라미터인 epsilon과 minPts를 사용자가 지정해주어야 한다는 단점(?) 이 있음.

   여기서 epsilon이란, 점 p가 있다고 할 때 점 p에서부터 거리를 의미하며 minPts는  해당 거리 내에 존재하는 데이터 포인트의 개수를 의미함. DBSCAN 알고리즘을 사용하려면 기준점 부터의 거리 epsilon값과, 이 반경내에 있는 점의 수 minPts를 인자로 전달해야 하며 해당 인자가 어떻게 지정되냐에 따라 군집의 형태와 모양이 달라진다는 특징을 지니고 있음

   DBSCAN 은 K-means와 비교하여, 다음과 같은 장점을 가짐.

   1. 사전에 군집의 개수를 선정 할 필요가 없다.
   2. 클러스터의 밀도에 따라서 클러스터를 서로 연결하기 때문에 기하학적인 모양을 갖는 군집도 잘 찾을 수 있다.
   3. 자동으로 어떤 군집에도 속하지 않는 outlier를 검출해 준다.

   

   그러나, DBSCAN을 이상감지 영역에서 사용한다고 할 때 outlier의 개수를 지정할 수 없다는 점이 단점으로 작용하기도 함. 물론 epsilon값과 minPts값을 조정하여 outlier의 개수를 어느정도 조정 할 수는 있으나 시간이 대단히 오래걸린다는 단점이 있음.

   이러한 단점을 해결하여 이상감지 모형에서 사용할 수 있는 것이 바로 HDBSCAN임

   

2. HDBSCAN

   HDBSCAN은 학습 시간이 매우 오래걸리고, 조정해야할 매개변수가 많다는 DBSCAN의 단점을 어느정도 해결한 밀도 기반 클러스터링 모형임.

   HDBSCAN의 경우 입실론 파라미터는 더 이상 필요하지 않고 MinPts만 존재하여 하이퍼파라미터에 덜 민감하다는 장점이 있으며, outlier의 개수를 지정할 수 있다는 특징이 있음. 

   

3. GMM(Gaussian Mixture Model)

   GMM은 데이터가 여러개의 가우시안 분포에서부터 생성 되었을 것이라는 가정 하에 어떤 분포에서 나왔느냐를 기준으로 클러스터링을 진행하는 방법론임. 통계적인 가정이 필요 한 만큼, 해당 데이터가 다변량 가우시안 분포를 따르지 않을 시에는 정확도를 신뢰 할 수 없으며 분석가가 사전에 군집의 개수를 지정해 주어야 한다는 단점이 있음.

![img](https://t1.daumcdn.net/cfile/tistory/99E2AD335984306816)

위와 같은 데이터 분포가 존재한다고 할 때, 해당 데이터를 

![img](https://t1.daumcdn.net/cfile/tistory/9902AA3359842FDF2A)

위의 사진처럼 3개의 정규분포로 분류하는 것이 GMM의 주 개념이라고 할 수 있음. 

GMM을 사용하기 위해서는 두가지 모수를 추정해야 하는데, 첫번째는 어떤 정규분포에 속해있는가를 확률적으로 나타내는 weight값이고 두번째는 각각의 정규분포의 모수(평균, 분산)임. 그리고 해당 모수들을 추정하는데에는 EM(expectation Maximazation)임. EM이 무엇인지 간단하게 설명하면 다음과 같음.

> E 단계에서 모수를 임의의 값으로 설정한 다음, 해당 모수가 참이라는 가정 하에 잠재변수 Z(각 관측치가 어떤 정규분포에서 파생되었을 지에 대한 값)을 추정하고, M 단계에서는 추정한 Z값이 참이라는 가정 하에 해당 Z값을 최대화 할 수 있는 모수를 추정함. E와 M 과정을 반복하며, 더이상 추정한 모수의 값에 변화가 없을때까지 지속함

GMM모델을 피팅하게 되면 총 몇개의 군집으로 클러스터링이 되는지에 대한 정보와 각 관측치가 해당 군집에 속할 확률 값을 얻을 수 있는데, 

1. 분류된 특정 군집 자체를 모두 이상치로 봄
2. 각 군집에 속할 확률이 낮은 관측치를 이상치로 봄

의 두가지 방식대로 이상 감지를 진행 할 수 있음. 둘 중에 어떤 식으로 이상 감지를 진행 할지는 데이터의 형식과 분류된 군집의 특성에 따라 다를 수 있으므로 분석가의 개입이 필수적임. 

