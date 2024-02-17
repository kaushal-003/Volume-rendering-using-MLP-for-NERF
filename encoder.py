import torch
from torch import nn
from PIL import Image
import torchvision.transforms as transforms

#encoder function gives W matrix as an output
def encoder(model,filename):
    img = Image.open(filename)  
    transform = transforms.Compose([
        transforms.Resize((224, 224)), 
        transforms.ToTensor(),         
    ])
    img_tensor = transform(img)
    img_tensor=img_tensor.reshape([1,3,224,224])
    features1 = nn.Sequential(*list(model.children())[:3])
    features2 = nn.Sequential(*list(model.children())[4])
    features3 = nn.Sequential(*list(model.children())[5])
    features4 = nn.Sequential(*list(model.children())[6])
    out1 =features1(img_tensor)
    out2 =features2(out1)
    out3 =features3(out2)
    out4 =features4(out3)
    latent=[out1,out2,out3,out4]
    latent_sz=latent[0].shape[-2:]
    for i in range(len(latent)):
        latent[i]=torch.nn.functional.interpolate(
            latent[i],latent_sz,mode="bilinear",align_corners=True
        )
    latent=torch.cat(latent,dim=1)
    return latent

#an example
#insert file path here
# filename="DiLiGenT-MV\DiLiGenT-MV\mvpmsData\\bearPNG\\view_01\\001.png"
# model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)

# output=encoder(model=model,filename=filename)
