import cv2
import numpy as np
from scipy import ndimage as ndi
from skimage import filters, measure, transform
from tflearn import DNN
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import dropout, fully_connected, input_data
from tflearn.layers.estimator import regression


def foo(img_fn, model_fn='../data/model/model_weights'):
    img = cv2.imread(img_fn, cv2.IMREAD_GRAYSCALE)

    haar_fn = '../data/haarcascade_russian_plate_number.xml'
    haar = cv2.CascadeClassifier(haar_fn)
    detected = haar.detectMultiScale(img)
    plates = []
    for x, y, w, h in detected:
        obj = img[y:y + h, x:x + w]
        plates.append(obj)

    chars = plates[0] < filters.threshold_minimum(plates[0])

    labeled_chars, a = ndi.label(chars)
    labeled_chars = (labeled_chars > 1).astype(np.int8)

    c = measure.find_contours(labeled_chars, .1)

    letters = []
    for i, v in enumerate(c):
        xs, ys = zip(*[i for i in v])
        x = int(min(xs))
        y = int(min(ys))
        w = int(max(xs) - x + 2)
        h = int(max(ys) - y + 2)
        if w < 15:
            continue
        letters.append((y, x, h, w))

    letters = sorted(letters)

    letters_img = [plates[0][x:x + w, y:y + h] for y, x, h, w in letters]

    letters_img = [i for i in letters_img if i[0, 0] > 127]

    sizes = [image.size for image in letters_img]
    median = np.median(sizes)
    allowed_size = median + median / 4

    letters_img = [image for image in letters_img if image.size < allowed_size]

    size = 64

    normalized_img = []
    for i in letters_img:
        ratio = i.shape[0] / i.shape[1]
        img1 = transform.resize(i, [size, int(size / ratio)], mode='constant')
        width = img1.shape[1]
        missing = (size - width) // 2
        ones = np.ones([size, missing])
        img2 = np.append(ones, img1, 1)
        img3 = np.append(img2, ones, 1)
        if 2 * missing + width != size:
            one = np.ones([size, 1])
            img4 = np.append(img3, one, 1)
        else:
            img4 = img3
        normalized_img.append(img4 * 255)

    net_input = input_data(shape=[None, 64, 64, 1])

    conv1 = conv_2d(net_input, nb_filter=4, filter_size=5, strides=[1, 1, 1, 1], activation='relu')
    max_pool1 = max_pool_2d(conv1, kernel_size=2)

    conv2 = conv_2d(max_pool1, nb_filter=8, filter_size=5, strides=[1, 2, 2, 1], activation='relu')
    max_pool2 = max_pool_2d(conv2, kernel_size=2)

    conv3 = conv_2d(max_pool2, nb_filter=12, filter_size=4, strides=[1, 1, 1, 1], activation='relu')
    max_pool3 = max_pool_2d(conv3, kernel_size=2)

    fc1 = fully_connected(max_pool3, n_units=200, activation='relu')
    drop1 = dropout(fc1, keep_prob=.5)

    fc2 = fully_connected(drop1, n_units=36, activation='softmax')
    net = regression(fc2)

    model = DNN(network=net)
    model.load(model_file=model_fn)

    labels = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')

    predicted = []
    for i in normalized_img:
        y = model.predict(i.reshape([1, 64, 64, 1]))
        y_pred = np.argmax(y[0])
        predicted.append(labels[y_pred])

    return ''.join(predicted)
