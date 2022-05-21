import cv2
import torch
from torch.autograd import Variable
from torchvision import transforms

from models.cnn import Net

use_cuda = False
model = Net()
model.load_state_dict(torch.load('output/params_15.pth'))
model.eval()
if use_cuda and torch.cuda.is_available():
    model.cuda()

img = cv2.imread('6_00002.jpg')    # 28*28*3
img_tensor = transforms.ToTensor()(img)  # 3*28*28
img_tensor = img_tensor.unsqueeze(0)     # 1*3*28*28          # 以行为方向进行扩充
if use_cuda and torch.cuda.is_available():
    prediction = model(Variable(img_tensor.cuda()))
else:
    prediction = model(Variable(img_tensor))
pred = torch.max(prediction, 1)[1]
print(pred)
cv2.imshow("image", img)
cv2.waitKey(0)
