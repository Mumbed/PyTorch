# 데이터에 매개변수를 최적화시켜 모델을 학습,검증,테스트해보자 epoch 실시
# 각 반복 단계에서 모델은 출력을 추측하고, 추측과 정답 사이의 오류(loss)를 계산하고 매개변수에 대한 오류의 도함수를 수집한 뒤 경사하강법을 사용해 optimize 하자

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda

training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)

train_dataloader = DataLoader(training_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)


class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


model = NeuralNetwork()

# 하이퍼 파라미터 - 모델 최적화 과정을 제어할 수 있는 조절 가능한 매개변수, 서로 다른 하이퍼파라미터 값은 모델 ㅏㄱ습과 수렴율에 영향을 미칠 수 있음 (하이퍼파라미터 튜닝)
# epoch 수- 데이터셋을 반복하는 횟수 , batch size = 매개변수가 갱신되기 전 신경망을 통해 전파된 데이터 샘플의 수
# learning rate = 각 배치/에폭에서 모델의 매개변수를 조절하는 비율, 값이 작을수록 학습 속도가 느려지고 ,값이 크면 학습 중 예측할 수 없는 동작이 발생할 수 있음

learning_rate = 1e-3
batch_size = 64
epochs = 5

# Optimization loop 에서 epoch 은 두가지로 구성됨 - 학습단계(train loop) 학습용 데이터셋을 반복하고 최적의 매개변수로 수렴함
# 검증 /테스트 단계 - 모델 성능이 개선되고 있는지를 확인하기 위해 테스트 데이터셋을 반복

# lose function - 획드한 결과와 실제 값 사이의 틀린 정도를 측정하여 이 값을 최소화시켜야함
# 일반적인 손실함수에는 회귀문제에 사용되는 nn.MSELoss(MSE)나 분류에 사용되는 nn.NLLLoss(음의 로그 우도)그리고 nn.CrossEntorpyLoss 등이 있다
# 모델의 출력 logit을 전달하여 logit 을 정규화하여 예측 오류를 계산 다음은 손실함수의 초기화 이다
loss_fn = nn.CrossEntropLoss()

# 최적화는 각 학습 단계에서 모델의 오류를 줄이기 위해 모델 매개변수를 조정하는 과정임
# 확률적 경사하강법을 정의할 예정이며 모든 최적화 절차(logic)은 optimizer 객체에 캡슐화 될 예정
# 다음은 SGD optimizer 이다
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)


# 학습 단계에서 최적화는 다음 세단계이다
# optimizer.zero_grad()를 호출하여 모델 매개변수의 변화도를 재설정함 기본적으로 변화도는 더해지기 때문에 중복 계산을 막기 위해 반복할 때마다 0으로 설정
# loss.backward()를 호출하여 예측 손실을 역전파함. Pytorch는 각 매개변수에 대한 손실의 변화도를 저장
# 변화도를 계산한 뒤에는 optimizer.step()을 호출하여 적전파 단계에서 수집된 변화도로 매개변수를 조정
def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        # 예측(prediction)과 손실(loss) 계산
        pred = model(X)
        loss = loss_fn(pred, y)

        # 역전파
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

epochs = 10
for t in range(epochs):
    print(f"Epoch {t + 1}\n-------------------------------")
    train_loop(train_dataloader, model, loss_fn, optimizer)
    test_loop(test_dataloader, model, loss_fn)
print("Done!")
