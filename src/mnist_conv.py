#!/usr/bin/env python

import math

import tensorflow as tf
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets

tf.set_random_seed(0)

MNIST = read_data_sets('data', one_hot=True, reshape=False)


def conv_layer(input, w_shape, strides, name='conv'):
    with tf.name_scope(name):
        w = tf.Variable(tf.truncated_normal(shape=w_shape, stddev=.1), name='W')
        b_shape = w_shape[-1:]
        b = tf.Variable(tf.truncated_normal(shape=b_shape, mean=.1, stddev=.1), name='b')
        conv = tf.nn.conv2d(input=input, filter=w, strides=strides, padding='SAME')
        activation = tf.nn.relu(conv + b)

        tf.summary.histogram('weights', w)
        tf.summary.histogram('biases', b)
        tf.summary.histogram('activation', activation)
    return activation


def dense_layer(input, w_shape, name='dense'):
    with tf.name_scope(name):
        w = tf.Variable(tf.truncated_normal(shape=w_shape, stddev=.1), name='W')
        b_shape = w_shape[-1:]
        b = tf.Variable(tf.truncated_normal(shape=b_shape, mean=.1, stddev=.1), name='b')
        activation = tf.nn.relu(features=tf.matmul(input, w) + b)

        tf.summary.histogram('weights', w)
        tf.summary.histogram('biases', b)
        tf.summary.histogram('activation', activation)
    return activation


def softmax_layer(input, w_shape, name='softmax'):
    with tf.name_scope(name):
        w = tf.Variable(tf.truncated_normal(shape=w_shape, stddev=.1), name='W')
        b_shape = w_shape[-1:]
        b = tf.Variable(tf.truncated_normal(shape=b_shape, mean=.1, stddev=.1), name='b')
        logits = tf.matmul(input, w) + b
        activation = tf.nn.softmax(logits=logits)

        tf.summary.histogram('weights', w)
        tf.summary.histogram('biases', b)
        tf.summary.histogram('activation', activation)
    return activation, logits


X = tf.placeholder(tf.float32, [None, 28, 28, 1], name='X')
tf.summary.image('input', tensor=X, max_outputs=10)

Y_ = tf.placeholder(tf.float32, [None, 10], name='labels')

LR = tf.placeholder(tf.float32)

K = 4
L = 8
M = 12
N = 200
S = 10

Y1 = conv_layer(X, w_shape=[5, 5, 1, K], strides=[1, 1, 1, 1], name='conv1')
Y2 = conv_layer(Y1, w_shape=[5, 5, K, L], strides=[1, 2, 2, 1], name='conv2')
Y3 = conv_layer(Y2, w_shape=[4, 4, L, M], strides=[1, 2, 2, 1], name='conv3')

YY = tf.reshape(Y3, [-1, 7 * 7 * M])

Y4 = dense_layer(YY, w_shape=[7 * 7 * M, N], name='fc1')

Y, Y_logits = softmax_layer(Y4, w_shape=[N, S], name='fc2')

with tf.name_scope('cross_entropy'):
    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(logits=Y_logits, labels=Y_)
    )
tf.summary.scalar('cross_entropy', tensor=cross_entropy)

with tf.name_scope('train'):
    train_step = tf.train.AdamOptimizer(LR).minimize(cross_entropy)

with tf.name_scope('accuracy'):
    correct_prediction = tf.equal(tf.argmax(Y, 1), tf.argmax(Y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
tf.summary.scalar('accuracy', tensor=accuracy)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

summaries = tf.summary.merge_all()
writer = tf.summary.FileWriter('/tmp/mnist_conv/3')
writer.add_graph(sess.graph)

EPOCHS = 301
EPOCH_PEEK = 20
BATCH_SIZE = 100

MAX_LR = .003
MIN_LR = .0001
DECAY_SPEED = 2000.


def train(lr):
    batch_x, batch_y = MNIST.train.next_batch(BATCH_SIZE)

    sess.run(train_step, feed_dict={
        X: batch_x,
        Y_: batch_y,
        LR: lr,
    })

    return batch_x, batch_y


def check(epoch, lr):
    acc = sess.run(accuracy, feed_dict={
        X: MNIST.test.images,
        Y_: MNIST.test.labels,
        LR: lr,
    })
    print(f'epoch: {epoch:>4}    LR: {lr:.6f}    acc: {acc:.3}')


def main():
    for epoch in range(EPOCHS):
        learning_rate = MIN_LR + (MAX_LR - MIN_LR) * math.exp(-epoch / DECAY_SPEED)

        batch_x, batch_y = train(learning_rate)

        if epoch % EPOCH_PEEK == 0:
            check(epoch, learning_rate)

        if epoch % 5 == 0:
            s = sess.run(summaries, feed_dict={
                X: batch_x,
                Y_: batch_y,
            })
            writer.add_summary(s, epoch)


if __name__ == '__main__':
    main()
