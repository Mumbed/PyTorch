import torch

#신경망을 학습할 때에는 역전파 알고리즘이 자주 사용됨
#역전파 알고리즘에서는 매개변수(모델 가중치)는 주어진 매개변수에 대한 손실 함수의 변화도(gradient)에 따라 조정됨
x = torch.ones(5)# input tensor
y = torch.zeros(3) #expected ouput
w = torch.randn(5,3,requires_grad = True) #w와 b는 매개변수
b = torch.randn(3, requires_grad = True)
z = torch.matmul(x,w)+b
loss = torch.nn.functional.binary_cross_entropy_with_logits(z,y)
#w와 b에에 대한 손실 함수의 변화도를 구할수 있어야 함 이를 위해 해당 tensor에 requires_grad 속성을 부여

print(f"Gradient function for z = {z.grad_fn}")
print(f"Gradient function for loss = {loss.grad_fn}")

#신경망에서 매개변수의 가중치를 최적화하려면 매개변수에 대한 손실함수의 도함수를 계산하여야함 - 이러한 도함수를 계산하기 위해 backward를 호출해
#w.grad와 b.grad에서 값을 가져옴

loss.backward()
print(f"{w.grad}")
print(b.grad)

#기본적으로 requires_grad=True 인 모든 tensor들은 연산기록을 추적하고 변화도 계산을 지원함
#순전파 연산만 필요한 경우엔 torch.no_grad() 을 사용해 연산 추적을 멈춤

z = torch.matmul(x, w)+b
print(z.requires_grad)

#torch.no_grad() 블록으로 감싸기
with torch.no_grad():
    z = torch.matmul(x, w)+b
print(z.requires_grad)
#다른 방법 detach()메서드 사용
z = torch.matmul(x, w)+b
z_det = z.detach()
print(z_det.requires_grad)

