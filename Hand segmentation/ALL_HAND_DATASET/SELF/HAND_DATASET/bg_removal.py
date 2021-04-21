import glob
import cv2
import os
# Requires "requests" to be installed (see python-requests.org)
import requests

raw_image_list = glob.glob('./RAW/*')
i=0
for image in raw_image_list:
    if(i<49):
        mask_path = './MASK/'+'frame_raw_'+'0'+str(i)+'.jpg'
        i = i + 1
        continue
    else:
        mask_path = './MASK/'+'frame_raw_'+str(i)+'.jpg'
    print(image)
    
    response = requests.post(
        'https://api.remove.bg/v1.0/removebg',
        files={'image_file': open(image, 'rb')},
        data={'size': 'auto'},
        headers={'X-Api-Key': 'N1WhpfK8h7fWjQg27vc1h1kV'},
    )
    i = i + 1
    
    print(mask_path)
    if response.status_code == requests.codes.ok:
        with open(mask_path, 'wb') as out:
            out.write(response.content)
    else:
        print("Error:", response.status_code, response.text)