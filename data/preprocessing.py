
# coding: utf-8

# In[1]:

from skimage import segmentation, filters, measure, transform
from skimage.feature import canny
from scipy import ndimage as ndi
import matplotlib.pyplot as plt
import cv2
import numpy as np
import os
import pandas as pd


plt.set_cmap(plt.gray())

# get_ipython().magic('matplotlib inline')


# In[2]:


DIR = '../data/images_letters/'
DIRS = [os.path.join(DIR, i) for i in os.listdir(DIR)]

dic = {}


for d in os.listdir(DIR):
    dd = os.path.join(DIR, d)
    print(d)
    l = []
    for fn in os.listdir(dd):
        fp = os.path.join(dd, fn)
        img = cv2.imread(fp, 0)
        l.append((img.shape[0] / img.shape[1], img.shape, img))


    ratios = [i[0] for i in l]
    mean = np.mean(ratios)
    std = np.std(ratios)

    lower = mean - std
    upper = mean + std


    ll = [i for i in l if lower < i[0] < upper]

    print(len(l), len(ll))


    height = 64

    if d not in dic:
        dic[d] = 0

    for r, _, i in ll:
        img = transform.resize(i, [height, int(height / r)], mode='constant')
        width = img.shape[1]
        missing = (height - width) // 2
        zer = np.ones([height, missing])
        img = np.append(zer, img, 1)
        img = np.append(img, zer, 1)
        if 2 * missing + width != height:
            img = np.append(img, np.ones([height, 1]), 1)

        dic[d] += 1
        plt.imsave(os.path.join('images_letters_normalized', d, f'{dic[d]}.jpg'), img)



