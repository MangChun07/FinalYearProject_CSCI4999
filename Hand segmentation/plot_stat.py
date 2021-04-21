import torch
import numpy as np

def get_stat(CkptPath):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tr_loss_arr = []
    val_loss_arr = []
    meanioutrain = []
    pixelacctrain = []
    meanioutest = []
    pixelacctest = []
    
    checkpoint = torch.load(CkptPath)
    tr_loss_arr =  checkpoint['Training Loss']
    val_loss_arr =  checkpoint['Validation Loss']
    meanioutrain =  checkpoint['MeanIOU train']
    pixelacctrain =  checkpoint['PixelAcc train']
    meanioutest =  checkpoint['MeanIOU test']
    pixelacctest =  checkpoint['PixelAcc test']
    
    print('Training Loss: ', np.mean(tr_loss_arr))
    print('Validation Loss: ',np.mean(val_loss_arr))
    print('MeanIOU train: ',np.mean(meanioutrain))
    print('PixelAcc train: ',np.mean(pixelacctrain))
    print('MeanIOU test: ',np.mean(meanioutest))
    print('PixelAcc test: ',np.mean(pixelacctest))

    return tr_loss_arr, val_loss_arr, meanioutrain, pixelacctrain, meanioutest, pixelacctest

retval = get_stat("checkpoints/handseg_aug2_00005_5_4iter_4.pt")

import matplotlib.pyplot as plt
def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0)) 
    return (cumsum[N:] - cumsum[:-N]) / float(N)

fig, ax = plt.subplots(ncols = 3, nrows = 2, figsize = (20,10))
N = 1000
ax[0][0].plot(running_mean(retval[0], N), 'r.', label='training loss')
ax[1][0].plot(running_mean(retval[1], N), 'b.', label='validation loss')
ax[0][1].plot(running_mean(retval[2], N), 'g.', label='meanIOU training')
ax[1][1].plot(running_mean(retval[4], N), 'r.', label='meanIOU validation')
ax[0][2].plot(running_mean(retval[3], N), 'b.', label='pixelAcc  training')
ax[1][2].plot(running_mean(retval[5], N), 'b.', label='pixelAcc validation')
for i in ax:
    for j in i:
        j.legend()
        j.grid(True)
plt.show()

import matplotlib as mpl
mpl.rcParams['axes.grid'] = False
mpl.rcParams['image.interpolation'] = 'nearest'
mpl.rcParams['figure.figsize'] = 15, 25