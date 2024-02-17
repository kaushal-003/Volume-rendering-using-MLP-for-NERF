import torch
from torch import nn
from PIL import Image
import torchvision.transforms as transforms


def encoder(model,filename):
    img = Image.open(filename)  
    transform = transforms.Compose([
        transforms.Resize((224, 224)), 
        transforms.ToTensor(),         
    ])
    img_tensor = transform(img)
    img_tensor=img_tensor.reshape([1,3,224,224])
    features = nn.Sequential(*list(model.children())[:7])
    out =features(img_tensor)
    return out

filename="DiLiGenT-MV\DiLiGenT-MV\mvpmsData\\bearPNG\\view_01\\001.png"
model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)

output=encoder(model=model,filename=filename)