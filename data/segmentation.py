
# coding: utf-8

# In[1]:

from skimage import segmentation, filters, measure
from skimage.feature import canny
from scipy import ndimage as ndi
import matplotlib.pyplot as plt
import cv2
import numpy as np
import os

plt.set_cmap(plt.gray())

# get_ipython().magic('matplotlib inline')


# In[2]:

# fp = '../data/Z3_ICE_plate_0.jpg'
# fp = '../data/images_plates/D1_JZO14_plate_0.jpg'
# fp = '../data/images_plates/D1_MAKAR_plate_0.jpg'
# fp = '../data/images_plates/W0_IPECO_plate_0.jpg'

fn = 'Z3_ICE_plate_0.jpg'
# fn = 'W0_IPECO_plate_0.jpg'

DIR = '../data/images_plates/'
fp = os.path.join(DIR, fn)

good = 0
bad = 0
al = 0

letters = {}

def foo(fn):
    global good, bad, al, letters

    fp = os.path.join(DIR, fn)
    img = s = cv2.imread(fp, 0)

# sobel, threshold, sobel, watershed

    plt.imshow(img)
    img, img.shape


# In[3]:

    coins = img < filters.threshold_minimum(img)

    plt.imshow(coins)
    coins, coins.shape, coins.dtype


# In[4]:

    labeled_coins, a = ndi.label(coins)
    labeled_coins = (labeled_coins > 1).astype(np.int8)
    plt.imshow(labeled_coins)
    labeled_coins, labeled_coins.shape, labeled_coins.dtype


# In[5]:

    c = measure.find_contours(labeled_coins, .1)


# In[6]:

    l = []
    for i, v in enumerate(c):
        xs, ys = zip(*[i for i in v])
        x = int(min(xs))
        y = int(min(ys))
        w = int(max(xs) - x + 2)
        h = int(max(ys) - y + 2)
        if w < 15:
            continue
        l.append((y, x, h, w))

    l = sorted(l)
    l


# In[7]:

    ll = [img[x:x+w, y:y+h] for y,x,h,w in l]

    ll = [i for i in ll if i[0,0] > 127]


# In[12]:

    plate = fn.replace('_', '').split('plate')[0]

    al += 1

    if len(plate) == len(ll):
        good += 1
    else:
        bad += 1
        return

    print(f'good: {good}', f'bad: {bad}', f'all: {al}', f'percent: {good/al}')

    for i, (v, letter) in enumerate(zip(ll, plate)):
        if letter not in letters:
            letters[letter] = 0
        letters[letter] += 1
        plt.imsave(f'images_letters/{letter}/{letters[letter]}.jpg', v)


for fn in sorted(os.listdir(DIR)):
    try:
        foo(fn)
    except Exception as err:
        print(fn, err)

# In[14]:

# d = [(i.shape[0] * i.shape[1], i) for i in ll]

# fig, axs = plt.subplots(1, len(ll))
# for i, (v, letter) in enumerate(zip(ll, plate)):
#     axs[i].imshow(v)
# #     plt.imsave(f'd/{letter}.jpg', v)


# # In[9]:

# [i[0] for i in sorted(d, key=lambda x: x[0])]
# [i[0] for i in d]

