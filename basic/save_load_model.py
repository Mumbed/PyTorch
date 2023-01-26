import torch
import torchvision.models as models

#모델의 상태를 유지(persist)하고 모델의 예측을 실행해보자


model = models.vgg16(pretrained=True)
torch.save(model.state.dict(), 'model_weights.pth')

model = models.vgg16() # 기본 가중치를 불러오지 않으므로 pretrained=True를 지정하지 않습니다.
model.load_state_dict(torch.load('model_weights.pth'))
model.eval()

torch.save(model, 'model.pth')
model = torch.load('model.pth')