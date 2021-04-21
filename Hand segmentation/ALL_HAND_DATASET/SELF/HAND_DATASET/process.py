import glob
import cv2
import os


mask_image_list = glob.glob('./MASK/*')
raw_image_list = glob.glob('./image/*')
i = 0
'''
for image in mask_image_list:
    print(image)
    mask_path = './MASK/'+sub+'_'+image[12::]+'.jpg'
    print(mask_path)
    i = i + 1
    img = cv2.imread(image)
    cv2.imwrite(mask_path, img)
print(i)
'''
i = 0

for image in raw_image_list:
    print(image)
    if(i<10):
        raw_path = './RAW/'+'frame_raw_'+'0'+str(i)+'.jpg'
    else:
        raw_path = './RAW/'+'frame_raw_'+str(i)+'.jpg'
    print(raw_path)
    i = i + 1
    img = cv2.imread(image)
    cv2.imwrite(raw_path, cv2.resize(cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE), (400, 225), cv2.INTER_AREA))
print(i)
