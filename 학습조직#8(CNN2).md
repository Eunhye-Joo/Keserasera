# 학습조직#8(CNN2)

## 14.4 CNN 구조

### 14.4.5 ResNet

총 152개의 레이어를 가진 ultra-deep Network로, 더 적은 파라미터를 사용해 더 깊은 네트워크로 모델을 구성하는 알고리즘임. 

![image-20210927195634561](https://tva1.sinaimg.cn/large/008i3skNgy1guvdwyx9qlj61a80g8myl02.jpg)

일반적으로는 레이어가 깊어짐에 따라 모델 성능이 좋아질 것이라고 예상하지만, 위 사진과 같이 실제로 레이어가 많아진다고 해서 모델 성능이 linear하게 좋아지는 것이 아님. **이는 네트워크가 깊어질수록 최적화를 하는 것이 더 어렵기** 때문인데, ResNet은 위와 같은 문제를 해결하였음. 

ResNet은 스킵 연결을 통해 더 적은 파라미터로 더 깊은 학습을 진행하게끔 하였는데, 

![image-20210927200348439](https://tva1.sinaimg.cn/large/008i3skNgy1guve4gw67dj60sg0d60th02.jpg)

왼쪽이 일반 레이어, 오른쪽이 스킵 연결을 사용한 레이어에 해당함. 두 구조는 동일한 연산을 진행 한 후에 Input X를 더하냐(residual block), 더하지 않느냐(plain) 의 차이를 지님. 

즉 스킵 연결을 통해 각각의 layer(block)들이 작은 정보를 추가적으로 더 학습하게끔 하는것인데 이는 곧 각각의 레이어가 학습해야 할 정보량을 축소시키는 것이라고 할 수 있음. 

### 14.4.6 Xception

Xception은 Inception에서 파생된 것으로, 채널간의 관계를 찾는 것과 이미지의 local 정보를 찾는것을 완전히 분리하고자 하였음. 

![image-20210927201225776](https://tva1.sinaimg.cn/large/008i3skNgy1guvedii45lj60yi0lgta002.jpg)

기존의 컨볼루션이 모든 채널과 지역정보를 고려해 하나의 feature map을 생성 한 것과 달리, Xception은 각 채널별로 feature map을 생성하고 그 다음 1*1 conv연산을 사용해 출력되는 feature map 개수를 조정함. 

### 14.4.7 SENet

Senet은 기존의 인셉션 모듈이나 잔차 유닛에 se블록이라는 작은 신경망을 추가하여 성능을 향상한 알고리즘임. 

![img](https://tva1.sinaimg.cn/large/008i3skNgy1guveibhe95j60ng0553yu02.jpg)

특성맵은 X에서 컨볼루션을 통해 U로, U에서 SE block을 통해 ~XX~로 변환됨.

**SE block의 목적은 한마디로 컨볼루션을 통해 생성된 특성을 채널당 중요도를 고려해서 재보정(recalibration)하는 것.** 이러한 SE block을 컨볼루션 연산 뒤에 붙여줌으로써 성능 향상이 이루어짐.

## 14.5~ 14.8 python 실습 파일 진행

## 14.9 객체 탐지

CNN은 이미지 위를 슬라이딩 해 가며 여러 물체를 감지하는데, 조금씩 다른 위치에서 동일한 물체를 여러번 감지하기 때문에 중복이 생김. 이러한 중복을 제거하기 위해서는 추가적인 사후처리가 필요함. 

### 14.9.1 Fully connected CNN

FCN에서는 기존의 classification에 사용되던 모델들을 이용하여 tranfer learning을 함. 그러나 기존의 classification의 모델들은 class 분류를 위해 네트워크의 마지막엔 항상 Fully connected layer(이하 Fc layer)가 삽입되게 되는데, 이는 image segmentation에는 적합하지 않음. 왜냐하면 Fc layer를 사용하기 위해서는 고정된 크기의 input만을 받아야하며, 1차원적인 정보만을 가지고 있기 때문에 원하는 2차원적인 정보(위치정보 등,,)를 담을 수 없기 때문임. 

그래서 FCN은  classification에서 사용되었던 모델의 마지막 Fc layer들을 convolution layer로 대체하여 2차원적인 정보를 담아보자 하는 것임.  

![image-20210928194454151](https://tva1.sinaimg.cn/large/008i3skNgy1guwj77nnqbj61140n440j02.jpg)

위의 그림을 보면,  기존의 classification model 의 fc layer를 convolution layer로 바꿈. 원래 fc layer는 단순히 tabby cat 이라는 하나의 class로 분류될 score정보만을 출력해 주었다면, convolutionalization을 거친 아래의 모델은 conv layer를 사용했으므로 위치정보나 class에대한 정보 또한 잃어버리지 않고 지니고 있음.

tabby cat heatmap을 보면 tabby cat의 class에대한 정보뿐만아니라 그 위치정보까지 함께 가지고 있고 또한 일정한 크기의 input을 요하는 fc layer와는 다르게 conv layer는 filter의 크기만 맞다면 어떠한 input이 오더라도 수용할 수 있어서 input의 크기에 제약을 받지 않는다는 특징 역시 있음. 

### 14.9.2 YOLO

물체인식(Object Detection)을 수행하기 위해 고안된 심층 신경망으로서,
테두리상자 조정(Bounding Box Coordinate)과 분류(Classification)를 동일 신경망 구조를 통해
동시에 실행하는 통합인식(Unified Detection)을 구현하는 것이 가장 큰 특징.

![img](https://tva1.sinaimg.cn/large/008i3skNgy1guwjsm5xnjj60ij0ildip02.jpg)

예측하고자 하는 이미지를 SxS Grid cells로 나누고 각 cell마다 하나의 객체를 예측한다. 그리고 미리 설정된 개수의 boundary boxes를 통해 객체의 위치와 크기를 파악한다. 이때, 각 cell마다 하나의 객체만을 예측할 수 있기 때문에 여러 객체가 겹쳐있으면 몇몇의 객체는 탐지를 못 하게 될 수 있음. 

각 cell은 다음 조건 하에 예측을 진행함.

- **B**개의 boundary boxes를 예측하고 각 box는 하나의 **box confidence score**를 가지고 있다.
- 예측된 box 수에 관계없이 **단 하나**의 객체만 탐지한다.
- **C**개의 **conditional class probabilities**를 예측한다.

![img](https://tva1.sinaimg.cn/large/008i3skNgy1guwjtwvgtoj60jg0cgdge02.jpg)

각 boundary box는 객체의 위치 (x, y), 객체의 크기 (w, h), box confidence score로 구성되어 총 5개의 인자를 가지고 있음. 여기서 **box confidence score**는 box가 객체를 포함하고 있을 가능성(objectness)과 boundary box가 얼마나 정확한지를 반영. **Conditional class probabilities**는 탐지된 객체가 어느 특정 클래스에 속하는지에 대한 확률.

### 14.10 Semantic segmentaion

Semantic Image Segmentation의 목적은 사진에 있는 모든 픽셀을 해당하는 (미리 지정된 개수의) class로 분류하는 것. 이미지에 있는 모든 픽셀에 대한 예측을 하는 것이기 때문에 dense prediction 이라고도 불림.

AlexNet, VGG 등 분류에 자주 쓰이는 깊은 신경망들은 parameter의 개수와 차원을 줄이는 layer를 가지고 있어 자세한 위치정보를 잃을 수 있기 때문에 Semantic Segmentation 을 하는데 적합하지 않음.  따라서 보통 Semantic Segmentation 모델들은 보통 Downsampling & Upsampling 의 형태를 가지고 있음. 

![img](https://tva1.sinaimg.cn/large/008i3skNgy1guwjzhagn5j60uk076ta102.jpg)

- Downsampling: 주 목적은 차원을 줄여서 적은 메모리로 깊은 Convolution 을 할 수 있게 하는 것. 보통 stride 를 2 이상으로 하는 Convolution 을 사용하거나, pooling을 사용. 이 과정을 진행하면 어쩔 수 없이 feature 의 정보를 잃게됨.
  마지막에 Fully-Connected Layer를 넣지 않고, Fully Connected Network 를 주로 사용. [FCN](https://people.eecs.berkeley.edu/~jonlong/long_shelhamer_fcn.pdf) 모델에서 위와같은 방법을 제시한 후 이후에 나온 대부분의 모델들에서 사용하는 방법임.
- Upsampling: Downsampling 을 통해서 받은 결과의 차원을 늘려서 인풋과 같은 차원으로 만들어 주는 과정. 주로 Strided Transpose Convolution 을 사용

