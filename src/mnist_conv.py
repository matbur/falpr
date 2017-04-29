#!/usr/bin/env python

import math

import tensorflow as tf
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets

tf.set_random_seed(0)

mnist = read_data_sets('data', one_hot=True, reshape=False)

K = 4
L = 8
M = 12
N = 200
S = 10

X = tf.placeholder(tf.float32, [None, 28, 28, 1])
Y_ = tf.placeholder(tf.float32, [None, 10])

lr = tf.placeholder(tf.float32)


def conv_layer(input, w_shape, strides):
    w = tf.Variable(tf.truncated_normal(shape=w_shape, stddev=.1))
    b_shape = w_shape[-1:]
    b = tf.Variable(tf.truncated_normal(shape=b_shape, mean=.1, stddev=.1))
    conv = tf.nn.conv2d(input=input, filter=w, strides=strides, padding='SAME')
    layer = tf.nn.relu(conv + b)
    return layer


def dense_layer(input, w_shape):
    w = tf.Variable(tf.truncated_normal(shape=w_shape, stddev=.1))
    b_shape = w_shape[-1:]
    b = tf.Variable(tf.truncated_normal(shape=b_shape, mean=.1, stddev=.1))
    layer = tf.nn.relu(features=tf.matmul(input, w) + b)
    return layer


def softmax_layer(input, w_shape):
    w = tf.Variable(tf.truncated_normal(shape=w_shape, stddev=.1))
    b_shape = w_shape[-1:]
    b = tf.Variable(tf.truncated_normal(shape=b_shape, mean=.1, stddev=.1))
    logits = tf.matmul(input, w) + b
    layer = tf.nn.softmax(logits=logits)
    return layer, logits


Y1 = conv_layer(X, w_shape=[5, 5, 1, K], strides=[1, 1, 1, 1])
Y2 = conv_layer(Y1, w_shape=[5, 5, K, L], strides=[1, 2, 2, 1])
Y3 = conv_layer(Y2, w_shape=[4, 4, L, M], strides=[1, 2, 2, 1])

YY = tf.reshape(Y3, [-1, 7 * 7 * M])

Y4 = dense_layer(YY, w_shape=[7 * 7 * M, N])

Y, Y_logits = softmax_layer(Y4, w_shape=[N, S])

cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=Y_logits, labels=Y_)
# cross_entropy = tf.reduce_mean(cross_entropy) * 100

correct_prediction = tf.equal(tf.argmax(Y, 1), tf.argmax(Y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

train_step = tf.train.AdamOptimizer(lr).minimize(cross_entropy)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

EPOCHS = 301
BATCH_SIZE = 100
EPOCH_PEEK = 20

max_learning_rate = .003
min_learning_rate = .0001
decay_speed = 2000.

for epoch in range(EPOCHS):
    batch_x, batch_y = mnist.train.next_batch(BATCH_SIZE)
    learning_rate = min_learning_rate + (max_learning_rate - min_learning_rate) * math.exp(-epoch / decay_speed)

    sess.run(train_step, feed_dict={
        X: batch_x,
        Y_: batch_y,
        lr: learning_rate,
    })

    if epoch % EPOCH_PEEK == 0:
        acc = sess.run(accuracy, feed_dict={
            X: mnist.test.images,
            Y_: mnist.test.labels,
            lr: learning_rate,
        })
        print(f'epoch: {epoch:>4}    lr: {learning_rate:.6f}    acc: {acc:.3}')
