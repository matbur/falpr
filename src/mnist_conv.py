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

W1 = tf.Variable(tf.truncated_normal([5, 5, 1, K], stddev=.1))
B1 = tf.Variable(tf.truncated_normal([K], stddev=.1))
Y1 = tf.nn.relu(tf.nn.conv2d(X, W1, strides=[1, 1, 1, 1], padding='SAME') + B1)
W2 = tf.Variable(tf.truncated_normal([5, 5, K, L], stddev=.1))
B2 = tf.Variable(tf.truncated_normal([L], stddev=.1))
Y2 = tf.nn.relu(tf.nn.conv2d(Y1, W2, strides=[1, 2, 2, 1], padding='SAME') + B2)
W3 = tf.Variable(tf.truncated_normal([4, 4, L, M], stddev=.1))
B3 = tf.Variable(tf.truncated_normal([M], stddev=.1))
Y3 = tf.nn.relu(tf.nn.conv2d(Y2, W3, strides=[1, 2, 2, 1], padding='SAME') + B3)

YY = tf.reshape(Y3, [-1, 7 * 7 * M])

W4 = tf.Variable(tf.truncated_normal([7 * 7 * M, N], stddev=.1))
B4 = tf.Variable(tf.truncated_normal([N], stddev=.1))
Y4 = tf.nn.relu(tf.matmul(YY, W4) + B4)

W5 = tf.Variable(tf.truncated_normal([N, S], stddev=.1))
B5 = tf.Variable(tf.truncated_normal([S], stddev=.1))
Y_logits = tf.matmul(Y4, W5) + B5
Y = tf.nn.softmax(Y_logits)

cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=Y_logits, labels=Y_)
cross_entropy = tf.reduce_mean(cross_entropy) * 100

correct_prediction = tf.equal(tf.argmax(Y, 1), tf.argmax(Y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

train_step = tf.train.AdamOptimizer(lr).minimize(cross_entropy)
# train_step = tf.train.GradientDescentOptimizer(.1).minimize(cross_entropy)

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
