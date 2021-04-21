import torch
from torch import nn
from torchvision import models
import numpy

# Model
deeplab = models.segmentation.deeplabv3_resnet50(pretrained=False, progress=True, num_classes=2)
class HandSegModel(nn.Module):
    def __init__(self):
        super(HandSegModel,self).__init__()
        self.dl = deeplab
        
    def forward(self, x):
        y = self.dl(x)['out']
        return y

checkpoint = torch.load('checkpoints/handseg_noaug_00005_8_4iter_4.pt') # load the checkpoint, works in both original or modified model. https://pytorch.apachecn.org/docs/1.4/84.html#id24
model = HandSegModel()

i = 0
for key in checkpoint['state_dict']:
    if(key.split(".")[-1] == "weight"):
        i = i + 1

for key in checkpoint['state_dict']:
    if(key.split(".")[-1] == "weight"):
        print("remaining weight: ", i)
        print("org_shape: ", checkpoint['state_dict'][key].shape)
        
        w = checkpoint['state_dict'][key]
        new_w = torch.flatten(torch.normal(0,1,w.shape))
        org_w = torch.flatten(w)

        for idx, a in enumerate(org_w):
            new_w[idx] = a

        checkpoint['state_dict'][key] = new_w.reshape(w.shape)
        
        print("new_shape: ",checkpoint['state_dict'][key].shape)
        i = i - 1

model.load_state_dict(checkpoint['state_dict'])
x = torch.rand(2, 3, 400, 225)
ts_model = torch.jit.trace(model, x)
ts_model.save("TorchScript_Handseg_1.pt")

print("TorchScript Exported.")