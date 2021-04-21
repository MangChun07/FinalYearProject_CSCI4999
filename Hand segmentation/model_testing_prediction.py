import torch
from torch import nn, optim
from torchvision import transforms, models
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import glob, os
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import cv2
import time

WIDTH = 320
HEIGHT = 320

# Model
deeplab = models.segmentation.deeplabv3_resnet50(pretrained=0, 
                                                 progress=1, 
                                                 num_classes=2)
class HandSegModel(nn.Module):
    def __init__(self):
        super(HandSegModel,self).__init__()
        self.dl = deeplab
        
    def forward(self, x):
        y = self.dl(x)['out']
        return y

# testing
# add green mark to the image
# * not remove the background.
def SegmentHands(pathtest, model):
    
    if isinstance(pathtest, np.ndarray):
        img = Image.fromarray(pathtest)
    else :
        img = Image.open(pathtest)

    '''
    preprocess = transforms.Compose([transforms.Resize((WIDTH,HEIGHT), 2),
                                     transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    
    '''
    
    preprocess = transforms.Compose([transforms.Resize((HEIGHT, WIDTH), 2),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    

    Xtest = preprocess(img)
    
    with torch.no_grad():
        model.eval()
        if torch.cuda.is_available():
            dev = "cuda:0" 
        else:  
            dev = "cpu"
        device = torch.device(dev)
        model.to(device)
        Xtest = Xtest.to(device).float()    # input
        ytest = model(Xtest.unsqueeze(0).float()) # add 1 dimension

        ypos = ytest[0, 1, :, :].clone().detach().cpu().numpy()
        yneg = ytest[0, 0, :, :].clone().detach().cpu().numpy()
        #print(ypos.astype('float32'))
        #print(yneg.astype('float32'))
        ytest = ypos >= yneg
        #print(ytest.astype('float32'))
    
    mask = ytest.astype('float32')
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    mask = cv2.dilate(mask,kernel,iterations = 1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    return mask, ypos, yneg
'''
def getcoloredMask(image, mask):
    color_mask = np.zeros_like(image)
    color_mask[:, :, 1] += mask.astype('uint8') * 250
    masked = cv2.addWeighted(image, 1.0, color_mask, 1.0, 0.0)
    return masked
'''

cap = cv2.VideoCapture(0)
i = 0
cap_count = 0

'''
# one image
checkpoint = torch.load('checkpoints/handseg_aug2_00005_5_4iter_4.pt')
model = HandSegModel()
model.load_state_dict(checkpoint['state_dict'])

# example input
# x = torch.rand(2, 3, WIDTH, HEIGHT)
# ts_model = torch.jit.trace(model, x)
# ts_model.save("weighted_ts_handseg_model.pt")

# model = torch.jit.load('TorchScript_file/TS_Handseg.pt')

# input path here
img = cv2.imread('test/frame_raw_00.jpeg')

img = cv2.resize(img, (HEIGHT,WIDTH))
cv2.imshow('My Image', img)
rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
mask = SegmentHands(rgb, model)
hand_mask = np.zeros_like(img)
hand_mask[:,:,0] = mask.astype('uint8')*255
hand_mask[:,:,1] = mask.astype('uint8')*255
hand_mask[:,:,2] = mask.astype('uint8')*255

img[np.where(hand_mask != 255)[0],np.where(hand_mask != 255)[1],:] = np.array((119, 178, 78)).astype(np.uint8)
cv2.imshow('img, mask', np.hstack((img, hand_mask)))

cv2.waitKey(0)
cv2.destroyAllWindows()

'''

#video

checkpoint = torch.load('checkpoints/handseg_aug2_001_1_4iter_4.pt')
model = HandSegModel()
model.load_state_dict(checkpoint['state_dict'])

#model = torch.jit.load('TS_Handseg.pt')

# example input
# x = torch.rand(2, 3, WIDTH, HEIGHT)
# ts_model = torch.jit.trace(model, x)
# ts_model.save("weighted_ts_handseg_model.pt")

while(True):
    ret, frame = cap.read()
    
    frame = cv2.resize(frame, (WIDTH,HEIGHT))
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    copy = frame.copy()

    if i%3 == 0:
        i=0
        mask, ypos, yneg = SegmentHands(rgb, model)
        hand_mask = np.zeros_like(frame)
        hand_mask[:,:,0] = mask.astype('uint8')*255
        hand_mask[:,:,1] = mask.astype('uint8')*255
        hand_mask[:,:,2] = mask.astype('uint8')*255

    frame[np.where(hand_mask != 255)[0],np.where(hand_mask != 255)[1],:] = np.array((119, 178, 78)).astype(np.uint8)
    cv2.imshow('hand mask, hand', np.hstack((hand_mask, frame, copy)))
    cv2.imshow('ypos, yneg', np.hstack((ypos.astype('float32'), yneg.astype('float32'))))
    
    pos_mask = np.zeros_like(frame)
    pos_mask[:,:,0] = ypos.astype('uint8')*255
    pos_mask[:,:,1] = ypos.astype('uint8')*255
    pos_mask[:,:,2] = ypos.astype('uint8')*255
    neg_mask = np.zeros_like(frame)
    neg_mask[:,:,0] = yneg.astype('uint8')*255
    neg_mask[:,:,1] = yneg.astype('uint8')*255
    neg_mask[:,:,2] = yneg.astype('uint8')*255

    key = cv2.waitKey(24)
    if key & 0xFF == ord('q'):
        break

    if key & 0xFF == ord('e'):
        # cv2.imwrite("./imwrite/ymask.jpg", np.hstack((pos_mask, neg_mask)))
        cv2.imwrite("./imwrite/model_5.jpg", np.hstack((hand_mask, frame, copy)))
    
    if key & 0xFF == ord('w'):
    	path = 'all_hand_dataset/self/HAND_DATASET/RAW/'
    	name = 'frame_raw_' + str(cap_count) + '.jpg'
    	cv2.imwrite(os.path.join(path,name), frame)
    	path2 = 'all_hand_dataset/self/HAND_DATASET/MASK/'
    	name2 = 'frame_mask_' + str(cap_count) + '.jpg'
    	cv2.imwrite(os.path.join(path2,name2), hand_mask)
    	cap_count = cap_count + 1
    i+=1
    

cap.release()
cv2.destroyAllWindows()
