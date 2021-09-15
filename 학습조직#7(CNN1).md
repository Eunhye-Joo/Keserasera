# 학습조직#7(CNN1)

## 14.1 시각 피질의 구조

David H. Hubel과 Torsten Wiesel은 1958년과 1959년에 시각 피질의 구조에 대한 결정적인 통찰을 제공한 고양이 실험을 수행했는데, 이들은 시각 피질 안의 많은 뉴런이 작은 **local receptive field**(국부 수용영역)을 가진다는 것을 보였음. 이는 뉴런의 수용영역(receptive field)들은 서로 겹칠수 있고 이렇게 겹쳐진 수용영역들이 전체 시야를 이루게 되며,  추가적으로 어떤 뉴런은 수직선의 이미지에만 반응하고, 다른 뉴런은 다른 각도의 선에 반응하는 뉴런이 있을 뿐만아니라, 어떤 뉴런은 큰 수용영역을 가져 저수준의 패턴(edge, blob 등)이 조합되어 복잡한 패턴(texture, object)에 반응하다는 것을 알게 되었음.  ==이러한 관찰을 통해 고수준의 뉴런이 이웃한 저수준의 뉴런의 출력에 기반한다는 아이디어를 도출해냄.==(아래 그림출처 : [brainconnection](https://brainconnection.brainhq.com/2004/03/06/overview-of-receptive-fields/))

![img](https://tva1.sinaimg.cn/large/008i3skNgy1gtxm2g1s5hj606v05rgll02.jpg)

이러한 아이디어가 바로 **합성곱 신경망(CNN, Convolutional Neural Network)**으로 점차 진화되어 왔으며, 1998년 Yann Lecn et al.의 논문에서 손글씨 숫자를 인식하는데 사용한 LeNet-5가 소개 되면서 CNN이 등장하게 됨.

## 14.2 합성곱 층

우선, CNN은 Fully connected neuralnet 구조가 아닌, 합성곱층과 풀링층으로 구성되어있는데 이런 CNN의 구조는 네트워크가 첫번째 은닉층에서는 작은 저수준 특성에 집중하고, 그 다음 은닉층에서는 더 고수준 특성으로 조합해나가도록 도와줌.

![img](https://tva1.sinaimg.cn/large/008i3skNgy1gtxm66mxluj60ms08r3z902.jpg)

첫번째 그림이 Fully connect 뉴럴넷의 구조고, 두번째 그림이 CNN의 구조임.

### 14.2.1 완전연결 계층의 문제점

완전연결 계층(fully connected layer)을 이용해 MNIST 데이터셋을 분류하는 모델을 만들 때, 3차원(세로, 가로, 채널)인 MNIST 데이터(28, 28, 1)를 입력층(input layer)에 넣어주기 위해서 아래의 그림(출처: [cntk.ai](https://cntk.ai/pythondocs/CNTK_103A_MNIST_DataLoader.html))처럼, 3차원 → 1차원의 평평한(flat) 데이터로 펼쳐줘야 함.  즉, (28, 28, 1)의 3차원 데이터를 $28 \times 28 \times 1 = 784$의 1차원 데이터로 바꾼다음 입력층에 넣어주었음(data flatten 과정)

[![img](https://tva1.sinaimg.cn/large/008i3skNgy1gtxm7jbu2wj60d20c2jrw02.jpg)

이러한 완전연결 계층의 문제점은 바로 **'데이터의 형상이 무시'**된다는 것. 이미지 데이터의 경우 3차원(세로, 가로, 채널)의 형상을 가지며, 이 형상에는 **공간적 구조(spatial structure)**를 가짐. 예를 들어 공간적으로 가까운 픽셀은 값이 비슷하거나, RGB의 각 채널은 서로 밀접하게 관련되어 있거나, 거리가 먼 픽셀끼리는 관련이 없는 등, 이미지 데이터는 3차원 공간에서 이러한 정보들이 내포 되어있음.  하지만, 완전연결 계층에서 1차원의 데이터로 펼치게 되면 이러한 정보들이 사라지게 되는 문제점이 생김

### 14.2.2 합성곱층

합성곱층은 CNN에서 가장 중요한 구성요소이며, 14.2.1의 완전연결 계층과는 달리 **합성곱층(convolutional layer)**은 아래의 그림과 같이 입력 데이터의 형상을 유지함. 3차원의 이미지 그대로 입력층에 입력받으며, 출력 또한 3차원 데이터로 출력하여 다음 계층(layer)으로 전달하기 때문에 CNN에서는 이미지 데이터처럼 형상을 가지는 데이터를 제대로 학습할 가능성이 높다고 할 수 있음.

![img](https://tva1.sinaimg.cn/large/008i3skNgy1gtxnd55wyjj60x608wwf302.jpg)합성곱층의 뉴런은 아래의 그림처럼(출처: [towardsdatascience.com](https://www.google.co.kr/url?sa=i&source=images&cd=&cad=rja&uact=8&ved=2ahUKEwiisMajvYzeAhWBzbwKHQwADpsQjhx6BAgBEAM&url=https%3A%2F%2Ftowardsdatascience.com%2Fintuitively-understanding-convolutions-for-deep-learning-1f6f42faee1&psig=AOvVaw2rBeiGhqGeRHABcckWUyi1&ust=1539831412136958)) 입력 이미지의 모든 픽셀에 연결되는 것이 아니라 합성곱층 뉴런의 **수용영역(receptive field)안에 있는 픽셀에만 연결**이 되기 때문에, 앞의 합성곱층에서는 저수준 특성에 집중하고, 그 다음 합성곱층에서는 고수준 특성으로 조합해 나가도록 해줌.

![img](https://tva1.sinaimg.cn/large/008i3skNgy1gtxndq3ut2g60wi0nl1er02.gif)

![img](https://tva1.sinaimg.cn/large/008i3skNgy1gtxnnf0ed2j60o90ayta402.jpg)

### 14.2.3 패딩 (padding)

패딩(Padding)은 합성곱 연산을 수행하기 전, 입력데이터 주변을 특정값으로 채워 늘리는 것으로, 패딩(Padding)은 주로 출력데이터의 공간적(Spatial)크기를 조절하기 위해 사용한다. 패딩을 할 때 채울 값은 hyper-parameter로 어떤 값을 채울지 결정할 수 있지만, 주로 **zero-padding**을 사용함.

패딩을 사용하는 이유는 패딩을 사용하지 않을 경우, 데이터의 Spatial 크기는 Conv Layer를 지날 때 마다 작아지게 되므로, 가장자리의 정보들이 사라지는 문제가 발생하기 때문에 패딩을 사용하며, 주로 합성곱 계층의 출력이 입력 데이터의 공간적 크기와 동일하게 맞춰주기 위해 사용함.

![padding](https://tva1.sinaimg.cn/large/008i3skNgy1gtxnyf4rwxj60b009egm202.jpg)

### 14.2.4 스트라이드(Stride)

스트라이드는 입력데이터에 필터를 적용할 때 이동할 간격을 조절하는 것, 즉 **필터가 이동할 간격을 의미함. **스트라이드 또한 출력 데이터의 크기를 조절하기 위해 사용. 스트라이드(Stride)는 보통 1과 같이 작은 값이 더 잘 작동하며, Stride가 1일 경우 입력 데이터의 spatial 크기는 pooling 계층에서만 조절하게 할 수 있음. 아래의 그림은 1폭 짜리 zero-padding과 Stride값을 1로 적용한 뒤 합성곱 연산을 수행하는 예제.

![convolution](https://tva1.sinaimg.cn/large/008i3skNgy1gtxnzg44hxg60b403ugod02.gif)



### 14.2.5 필터 (Filter)

위에서 설명한 수용영역(receptive field)을 합성곱층에서 **필터(filter)** 또는 커널(kernel)이라고 함. 아래의 그림처럼, 이 필터가 바로 합성곱층에서의 가중치 파라미터($\mathbf{W}$)에 해당하며, 학습단계에서 적절한 필터를 찾도록 학습되며, 합성곱 층에서 입력데이터에 필터를 적용하여 필터와 유사한 이미지의 영역을 강조하는 **특성맵(feature map)**을 출력하여 다음 층(layer)으로 전달함.

![img](https://tva1.sinaimg.cn/large/008i3skNgy1gtxnu27it1j60cn05naaa02.jpg)



## 14.3 풀링층 (Pooling Layer)

풀링의 배경에는 기술적인 이유와 이론적인 이유가 있음 기술적 측면에서 풀링은 차례로 처리되는 데이터의 크기를 줄이는데, 이 과정으로 모델의 전체 매개변수의 수를 크게 줄일 수 있음.  풀링에는 **Max-Pooling과 Average pooling**이 있는데 Max-Pooling은 해당영역에서 최대값을 찾는 방법이고, Average-Pooling은 해당영역의 평균값을 계산하는 방법. 이미지 인식 분야에서는 주로 Max-Pooling을 사용함. 아래의 그림은 풀링의 윈도우 사이즈는 (2, 2)이며 스트라이드는 2로 설정하여 맥스풀링을 한 예제.

![img](https://tva1.sinaimg.cn/large/008i3skNgy1gtxr1coldij60h0089mxp02.jpg)

풀링의 이론적 측면은 계산된 특징이 이미지 내의 위치에 대한 변화에 영항을 덜 받는다는 것임. 예를 들어 이미지의 우측 상단에서 눈을 찾는 특징은, 눈이 이미지의 중앙에 위치하더라도 크게 영향을 받지 않아야 함. 그렇기 때문에 풀링을 이용하여 불변성(invariance)을 찾아내서 공간적 변화를 극복할 수 있다.

![img](https://tva1.sinaimg.cn/large/008i3skNgy1gtxr2b0ks8j60s80bsq4602.jpg)

## 14.4 CNN 구조

![CNN, Convolutional Neural Network 요약](https://tva1.sinaimg.cn/large/008i3skNgy1gtxreazcjdj60om06bwf702.jpg)

전형적인 CNN 구조는 위 사진처럼 합성곱 층을 몇개 쌓고, (각각 Relu 층을 그 뒤에 놓고) 그 다음 다시 풀링층을 쌓고, 그 다음에 또 합성곱 층을 몇개 더 쌓고, 그 다음에 다시 풀링층을 쌓는 식임. 이러한 구조 때문에 네트워크를 통과하여 진행할수록 이미지는 점점 더 작아지지만, 합성곱 층 때문에 일반적으로 더 더 깊어짐. 

CNN은 여러가지 구조를 가진 모형들이 있는데, 몇가지를 살펴 보도록 함.

### 14.4.1 LeNet-5

![img](https://tva1.sinaimg.cn/large/008i3skNgy1gtz1uia29hj60n206tmxk02.jpg)

LeNet-5는 인풋, 3개의 컨볼루션 레이어(C1, C3, C5), 2개의 서브샘플링 레이어(S2, S4), 1층의 full-connected 레이어(F6), 아웃풋 레이어로 구성되어 있으며  C1부터 F6까지 활성화 함수로 tanh을 사용함. 

C3에 있는 대부분의 뉴런은 S2의 3개 또는 4개 맵에 있는 뉴런에만 연결되며, 출력층은 입력과 가중치 벡터를 행렬 곱셈하는 대신 각 뉴런에서 입력 벡터와 가중치 벡터 사이의 유클리드 거리를 출력함. 

### 14.4.2 AlexNet

![img](https://tva1.sinaimg.cn/large/008i3skNgy1gtz1y1bctyj60p00cgjtm02.jpg)

AlexNet의 기본구조는 [LeNet-5](https://bskyvision.com/418)와 크게 다르지 않음. 2개의 **GPU**로 병렬연산을 수행하기 위해서 병렬적인 구조로 설계되었다는 점이 가장 큰 차이점이라고 할 수 있음.

AlexNet은 8개의 레이어로 구성되어 있으며, 5개의 컨볼루션 레이어와 3개의 full-connected 레이어로 구성되어 있음. 두번째, 네번째, 다섯번째 컨볼루션 레이어들은 전 단계의 같은 채널의 특성맵들과만 연결되어 있는 반면, 세번째 컨볼루션 레이어는 전 단계의 두 채널의 특성맵들과 모두 연결되어 있다는 특징을 가짐.

### 14.4.3 GoogLeNet

![img](https://tva1.sinaimg.cn/large/008i3skNgy1gtz1zt5fgfj61zw0g4n3802.jpg)

GoogLeNet의 가장 큰 특징중 하나는 바로 1*1 합성곱 층을 가진다는 것임. GoogLeNet에서 **1 x 1 컨볼루션은 특성맵의 갯수를 줄이는 목적으로 사용됨.** 특성맵의 갯수가 줄어들면 그만큼 연산량이 줄어듬.

또한, 인셉션 모듈이라는 서브 네트워크를 가지고 있어서 파라미터를 더 효율적으로 사용 가능한데, 인셉션 모듈의 핵심 아이디어는  Convolution network를 최적의 sparse한 matrix를 만들고, 연산에서는 최대한 dense하게 만드는 것임. 인셉션 모듈은 1x1, 3x3, 5x5 세 개의 Conv layer와 1개의 Max-pooling을 사용하고, **여러 스케일의 Convolution 연산을 활용해 다양한 스케일에서 효율적으로 특징을 뽑아낸** 뒤 ReLU함수를 사용,  각각의 결과를 연결해(concat) 하나의 output을 생성하는 식으로 이루어짐. 

### 14.4.4 VGGNet

VGGNet은 매우 단순하고 고전적인 구조로, 2개 혹은 3개의 합성곱 층 뒤에 풀링 층이 나오고 다시 2개 또는 3개의 합성곱 층과 풀링 층이 등장하는 구조로 구성되어 있음. 마지막 밀집 네트워크는 2개의 은닉층과 출력층으로 이루어지며 3*3 필터만 사용한다는 특징이 있음.

