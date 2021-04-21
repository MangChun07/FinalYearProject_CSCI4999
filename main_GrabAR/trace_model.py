import torch
import torchvision
from torchvision import transforms
from torch.autograd import Variable
import sys
sys.path.insert(0, "network")

from handnet_mask import HandNetInitial
from handnet_s import HandNet

import inspect

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net = HandNetInitial().to(device)
net.load_state_dict(torch.load('44.pth.tar')['state_dict'])  
net.eval()
print(inspect.getmro(net.__class__))

# handseg_net = HandNet().to(device)
# handseg_net.load_state_dict(torch.load('hand_seg.tar')['state_dict'])
# handseg_net.eval()
# print(inspect.getmro(handseg_net.__class__))

scripted_net = torch.jit.script(net)

# scripted_handseg_net = torch.jit.script(handseg_net)

scripted_net.save("model/TS_net.pt") # generate TorchScript format of PyTorch model.

# scripted_handseg_net.save("model/handseg_net.pt")
