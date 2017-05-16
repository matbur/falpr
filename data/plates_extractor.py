
# coding: utf-8

# In[1]:

import cv2
import random
from glob import glob
import matplotlib.pyplot as plt
import os


# In[2]:

haar_fn = 'haarcascade_russian_plate_number.xml'
haar = cv2.CascadeClassifier(haar_fn)


# In[3]:

files = sorted(os.listdir('images'))
files[:5]


# In[4]:

for ii, fn in enumerate(files):
    print(ii, fn)

    fp = os.path.join('images', fn)
    img = cv2.imread(fp, cv2.IMREAD_GRAYSCALE)
    print(img.shape)

    plate_fn = '_plate.'.join(fn.split('.'))
    plate_fp = fp = os.path.join('images_plates', plate_fn)

    plates = haar.detectMultiScale(img)
    if not len(plates):
        continue

    print(plates)

    for i, (x, y, w, h) in enumerate(plates):
        plate = img[y:y+h, x:x+w]
        plate_fp_i = f'_{i}.'.join(plate_fp.split('.'))
        plt.imsave(plate_fp_i, plate, cmap=plt.gray())

