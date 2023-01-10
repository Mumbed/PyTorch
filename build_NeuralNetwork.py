import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

model = NeuralNetwork().to(device)
print(model)

#모델을 사용하기 위해 입력 데이터 전달 - 백그라운드 연산들고 함께 모델의 forward를 실행
X = torch.rand(1, 28, 28, device=device)
#print(X)
logits = model(X)
pred_probab = nn.Softmax(dim=1)(logits)
y_pred = pred_probab.argmax(1)
#각 분류(class)에 대한 원시(raw) 예측값이 있는 10-차원 텐서가 반환됨.
#원시 예측값을 nn.Softmax 모듈의 인스턴스에 통과시켜 예측 확률을 얻음
print(f"Predicted class: {y_pred}")

#모델 계층(Layer)
#FashionMNIST 모델의 계층들을 살표보자.
# 28x28 size의 이미지 3개로 구성된 미니배치를 가져와 신경망을 통과할때의 모습 확인하기
input_image = torch.rand(3,28,28) # 이미지 3개. 28 x 28 size
print(input_image.size())

#nn.Flatten 계층을 초기화하여 28x28의 2D 이미지를 784 픽셀 값을 갖는 연속된 배열로 변환
flatten = nn.Flatten()
flat_image = flatten(input_image)
print(flat_image.size())

#nn.Linear - 선형계층은 저장된 가중치(weight)와 편향(bias)를 사용하여 입력에 선형 변환(linear transformation)을 적용하는 모듈
layer1 = nn.Linear(in_features=28*28, out_features=20)
hidden1 = layer1(flat_image)
print(hidden1.size())

#활성화 함수를 위해선 선형 layer 를 비선형 layer로 바꾸어 주어야 함 따라서
print(f"Before ReLU: {hidden1}\n\n")
hidden1 = nn.ReLU()(hidden1)
#활성화 함수는 입력 데이터를 다음 레이어로 어떻게 출력하느냐를 결정하는 역할이기 때문에 매우 중요함
print(f"After ReLU: {hidden1}")

#nn.Sequential - 순서를 갖는 모듈의 컨테이너
# 데이터는 정의된 것과 같은 순서로 모든 모듈들을 통해 전달됨 - 순차 컨테이너를 사용해 아래와 같은 신경망을 빠르게 만들 수 있음
seq_modules = nn.Sequential(
    flatten,
    layer1,
    nn.ReLU(),
    nn.Linear(20, 10)
)
input_image = torch.rand(3,28,28)
logits = seq_modules(input_image)

#nn.Softmax - 신경망의 마지막 선형 계층은 nn.Softmax 모듈에 전달될 ([-infty, infty] 범위의 원시 값(raw value)인) logits 를 반환함.
#logits는 모델의 각 분류(class)dp eogks dPcmr ghkrfbfdmf skxksoehfhr [0,1]범위로 비례하여 조정됨.
softmax = nn.Softmax(dim=1)
pred_probab = softmax(logits)

#신경망 내부의 많은 계층들은 매개변수화(parameterize)됨. 즉, 학습 중에 최적화되는 가중치와 편향과 연관지어짐
#다음에서는 각 매개변수들을 순회하고(iterate), 매개변수의 크기와 값을 출력함
print(f"Model structure: {model}\n\n")

for name, param in model.named_parameters():
    print(f"Layer: {name} | Size: {param.size()} | Values : {param[:2]} \n")