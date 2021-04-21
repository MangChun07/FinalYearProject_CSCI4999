import torch
from torch import nn, optim
from torchvision import transforms, models, utils
from PIL import Image
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import glob
from tqdm import tqdm
import numpy as np
import cv2
import time
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import albumentations as A

WIDTH = 320
HEIGHT = 320

# Model
deeplab = models.segmentation.deeplabv3_resnet50(pretrained=False, progress=True, num_classes=2)
class HandSegModel(nn.Module):
    def __init__(self):
        super(HandSegModel,self).__init__()
        self.dl = deeplab
        
    def forward(self, x):
        y = self.dl(x)['out']
        return y

# Dataset
class SegDataset(Dataset):    
    def __init__(self, parentDir, imageDir, maskDir, transform=None, aug = False):
        self.imageList = glob.glob(parentDir+'/'+imageDir+'/*')
        self.imageList.sort()
        self.maskList = glob.glob(parentDir+'/'+maskDir+'/*')
        self.maskList.sort()
        self.transform = transform
        self.aug = aug

    def __getitem__(self, index):
        image_resize = transforms.Resize((WIDTH,HEIGHT), 2)
        to_tensor = transforms.ToTensor()
        to_one_channel = transforms.Grayscale(num_output_channels=1)

        # X = Image.open(self.imageList[index]).convert('RGB')
        X = cv2.imread(self.imageList[index], cv2.IMREAD_COLOR)
        X = cv2.cvtColor(X, cv2.COLOR_BGR2RGB)

        # yimg = Image.open(self.maskList[index]).convert('L')
        yimg = cv2.imread(self.maskList[index], cv2.IMREAD_COLOR)
        yimg = cv2.cvtColor(yimg, cv2.COLOR_BGR2RGB)

        if self.aug and self.transform:
            transformed = self.transform(image=X, mask=yimg)
            X = to_tensor(image_resize(Image.fromarray(transformed['image'])))
            yimg = to_one_channel(Image.fromarray(transformed['mask']))

        elif self.transform:
            X = Image.fromarray(X)
            X = self.transform(X)
            yimg = to_one_channel(Image.fromarray(yimg))

        y1 = to_tensor(image_resize(yimg))
        y1 = y1.type(torch.BoolTensor)
        y2 = torch.bitwise_not(y1)
        y = torch.cat([y2, y1], dim=0)
        
        return X, y
            
    def __len__(self):
        return len(self.imageList)

def dataset_imshow(imageList, maskList):
    grid = image_grid(imageList, maskList)
    np_img = grid.permute(1,2,0).numpy()
    print(np_img.shape)
    plt.imshow(np_img)
    plt.show()

def image_grid(imageList, maskList, num_image = 25):
    to_tensor = transforms.Compose([transforms.ToTensor()])

    transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.ShiftScaleRotate(border_mode=cv2.BORDER_CONSTANT, 
                            rotate_limit=(10, 30),
                            p=0.5)
    ], p=1)

    image_list = []
    mask_list = []
    for i in range(num_image):
        img = cv2.imread(imageList[i], cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(maskList[i], cv2.IMREAD_COLOR)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)

        transformed = transform(image=img, mask=mask)

        t_image = to_tensor(transformed['image'])
        t_mask = to_tensor(transformed['mask'])

        image_list.append(t_image)
        mask_list.append(t_mask)
    
    concat = image_list + mask_list

    grid = utils.make_grid(concat, nrow=5, padding=3)
    return grid
    
def Data_preprocess(image_folder, preprocess, aug_times = 0):
    tempDataset = []
    for i in image_folder:
        nor_dataset = SegDataset(i,'RAW', 'MASK', preprocess)

        # plot the image and mask
        # dataset_imshow(nor_dataset.imageList, nor_dataset.maskList)

        tempDataset.append(nor_dataset)
        print(i)
        print("# of Normal data", len(nor_dataset))

        for j in range(aug_times):
            augmentation = A.Compose([
                    A.HorizontalFlip(p=0.5),
                    A.VerticalFlip(p=0.5),
                    A.ShiftScaleRotate( border_mode=cv2.BORDER_CONSTANT, 
                                        rotate_limit=(5, 25),
                                        p=0.5)
                ], p=1)
            aug_dataset = SegDataset(i,'RAW', 'MASK', augmentation, aug=True)
            tempDataset.append(aug_dataset)
            print("# of Augmented data",len(aug_dataset))

    mergeDataset = ConcatDataset(tempDataset)
    print("# of all data", len(mergeDataset))
    return mergeDataset

preprocess = transforms.Compose([transforms.Resize((WIDTH,HEIGHT), 2),
                                 transforms.ToTensor(),
                                 transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

'''
augmentation = transforms.Compose([ transforms.Resize((WIDTH,HEIGHT), 2),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                    transforms.ColorJitter(hue=.05, saturation=.05),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.RandomRotation(20, resample=Image.BILINEAR)
                                    ])
'''

image_folder = {'ALL_HAND_DATASET/EGO/HAND_DATASET',  # egohand_dataset
                'ALL_HAND_DATASET/EYTH/HAND_DATASET', # eyth_dataset
                'ALL_HAND_DATASET/HOF/HAND_DATASET',  # hof_dataset
                'ALL_HAND_DATASET/SELF/HAND_DATASET'}

mergeDataset = Data_preprocess(image_folder, preprocess, aug_times = 0)

# Split the dataset into two: training and validation
# TTR: Train Test Ratio
def trainTestSplit(dataset, TTR):
    trainDataset = torch.utils.data.Subset(dataset, range(0, int(TTR * len(dataset))))
    valDataset = torch.utils.data.Subset(dataset, range(int(TTR*len(dataset)), len(dataset)))
    return trainDataset, valDataset
  
batchSize = 2
trainDataset, valDataset = trainTestSplit(mergeDataset, 0.8)
trainLoader = DataLoader(trainDataset, batch_size = batchSize, shuffle=True, drop_last=True)
valLoader = DataLoader(valDataset, batch_size = batchSize, shuffle=True, drop_last=True)

print("training dataset no:", len(trainLoader.dataset))
print("validation dataset no:", len(valLoader.dataset))

# Performance
# Mean Intersection over Union, meanIOU
def meanIOU(target, predicted):
    if target.shape != predicted.shape:
        print("target has dimension", target.shape, ", predicted values have shape", predicted.shape)
        return
        
    if target.dim() != 4:
        print("target has dim", target.dim(), ", Must be 4.")
        return
    
    iousum = 0
    for i in range(target.shape[0]):
        target_arr = target[i, :, :, :].clone().detach().cpu().numpy().argmax(0)
        predicted_arr = predicted[i, :, :, :].clone().detach().cpu().numpy().argmax(0)
        
        intersection = np.logical_and(target_arr, predicted_arr).sum()
        union = np.logical_or(target_arr, predicted_arr).sum()
        if union == 0:
            iou_score = 0
        else :
            iou_score = intersection / union
        iousum += iou_score
        
    miou = iousum/target.shape[0]
    return miou

# Pixel accuracy
def pixelAcc(target, predicted):    
    if target.shape != predicted.shape:
        print("target has dimension", target.shape, ", predicted values have shape", predicted.shape)
        return
        
    if target.dim() != 4:
        print("target has dim", target.dim(), ", Must be 4.")
        return
    
    accsum=0
    for i in range(target.shape[0]):
        target_arr = target[i, :, :, :].clone().detach().cpu().numpy().argmax(0)
        predicted_arr = predicted[i, :, :, :].clone().detach().cpu().numpy().argmax(0)
        
        same = (target_arr == predicted_arr).sum()
        a, b = target_arr.shape
        total = a*b
        accsum += same/total
    
    pixelAccuracy = accsum/target.shape[0]        
    return pixelAccuracy

# Training
def training_loop(n_epochs, optimizer, lr_scheduler, model, loss_fn, train_loader, val_loader, lastCkptPath = None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tr_loss_arr = []
    val_loss_arr = []
    meanioutrain = []
    pixelacctrain = []
    meanioutest = []
    pixelacctest = []
    prevEpoch = 0

    if lastCkptPath != None :
        checkpoint = torch.load(lastCkptPath)
        prevEpoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)
                    tr_loss_arr =  checkpoint['Training Loss']
        val_loss_arr =  checkpoint['Validation Loss']
        meanioutrain =  checkpoint['MeanIOU train']
        pixelacctrain =  checkpoint['PixelAcc train']
        meanioutest =  checkpoint['MeanIOU test']
        pixelacctest =  checkpoint['PixelAcc test']
        print("loaded model, ", checkpoint['description'], "at epoch", prevEpoch)
        
    model.to(device)
    sum_time = 0

    for epoch in range(0, n_epochs):
        train_loss = 0.0
        pixelacc = 0
        meaniou = 0
        
        pbar = tqdm(train_loader, total = len(train_loader))
        for X, y in pbar:
            torch.cuda.empty_cache()
            model.train()
            X = X.to(device).float()
            y = y.to(device).float()
            ypred = model(X)
            loss = loss_fn(ypred, y)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            tr_loss_arr.append(loss.item())
            meanioutrain.append(meanIOU(y, ypred))
            pixelacctrain.append(pixelAcc(y, ypred))
            pbar.set_postfix({'Epoch':epoch+1+prevEpoch, 
                              'Training Loss': np.mean(tr_loss_arr),
                              'Mean IOU': np.mean(meanioutrain),
                              'Pixel Acc': np.mean(pixelacctrain)
                             })
            
        with torch.no_grad():
            val_loss = 0
            pbar = tqdm(val_loader, total = len(val_loader))
            for X, y in pbar:
                torch.cuda.empty_cache()
                X = X.to(device).float()
                y = y.to(device).float()
                model.eval()
                ypred = model(X)
                
                val_loss_arr.append(loss_fn(ypred, y).item())
                pixelacctest.append(pixelAcc(y, ypred))
                meanioutest.append(meanIOU(y, ypred))
                
                pbar.set_postfix({'Epoch':epoch+1+prevEpoch, 
                                  'Validation Loss': np.mean(val_loss_arr),
                                  'Mean IOU': np.mean(meanioutest),
                                  'Pixel Acc': np.mean(pixelacctest)
                                 })
        
        checkpoint = {
            'epoch':epoch+1+prevEpoch,
            'description':"add your description",
            'state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'Training Loss': tr_loss_arr,
            'Validation Loss':val_loss_arr,
            'MeanIOU train':meanioutrain, 
            'PixelAcc train':pixelacctrain, 
            'MeanIOU test':meanioutest, 
            'PixelAcc test':pixelacctest
        }
        torch.save(checkpoint, 'checkpoints/handseg_aug2_00005_5_4iter_'+str(epoch+1+prevEpoch)+'.pt')
        lr_scheduler.step()
        
    return tr_loss_arr, val_loss_arr, meanioutrain, pixelacctrain, meanioutest, pixelacctest

model = HandSegModel()

optimizer = optim.Adam(model.parameters(), lr=0.00005)
loss_fn = nn.BCEWithLogitsLoss ()
lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.5)

retval = training_loop(4, 
                       optimizer, 
                       lr_scheduler, 
                       model, 
                       loss_fn, 
                       trainLoader, 
                       valLoader, 
                        )

print("finish training")
