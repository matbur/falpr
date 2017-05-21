
# coding: utf-8

# In[1]:

from glob import glob
import os

import cv2
import numpy as np
import matplotlib.pyplot as plt

plt.set_cmap(plt.gray())

# get_ipython().magic('matplotlib inline')


# In[8]:

letters = {}


labels = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')

for letter in labels:
    print(letter)
    DIR = f'../data/images_letters_normalized/{letter}'
    DIR2 = f'../data/images_letters_normalized_cleaned/{letter}'


    l = []
    for i in glob(os.path.join(DIR, '*')):
        img = cv2.imread(i, 0)
        l.append(img)


# In[10]:

    mean = np.mean(l, 0)

# In[12]:

    ll = sorted(l, key=lambda x: np.sum((x - mean)**2), reverse=1)

    split_size = int(len(ll) / 10)
    ll = ll[split_size:]

    if letter not in letters:
        letters[letter] = 0

    for i in ll:
        letters[letter] += 1
        plt.imsave(os.path.join(DIR2, f'{letters[letter]}.jpg'), i)
