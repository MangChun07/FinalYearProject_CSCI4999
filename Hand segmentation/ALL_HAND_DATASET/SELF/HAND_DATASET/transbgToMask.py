import glob
import cv2
import os


transbg_image_list = glob.glob('./transbg/*')
i = 0

for image in transbg_image_list:
    print(image)
    if(i<10):
        mask_path = './MASK/'+image[10::]
    else:
        mask_path = './MASK/'+image[10::]
    print(mask_path)
    i = i + 1
    
    img = cv2.imread(image)
    if(img[0][0].all()==0):
        img = cv2.threshold(img, 5, 255, 0)[1]
    cv2.imshow("show",img)
    key = cv2.waitKey(90)
    if key == ord('q') or key == 27: # Esc
        break
    print(img[0][0])
    cv2.imwrite(mask_path, img)
    
print(i)
