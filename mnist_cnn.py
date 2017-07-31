import numpy as np
import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

x = tf.placeholder(tf.float32, [None, 784])
y_correct = tf.placeholder(tf.float32, [None, 10])  # correct labels


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    return tf.nn.conv2d(x, W,
                        strides=[1, 2, 2, 1],
                        padding='SAME')


def max_pooling(x):
    return tf.nn.max_pool(x,
                          ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1],
                          padding='SAME')

x_image = tf.reshape(x, [-1, 28, 28, 1])


# 1st layer: conv
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)  # relu before pooling
h_pool1 = max_pooling(h_conv1)

# 2nd layer: conv
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)  # relu before pooling
h_pool2 = max_pooling(h_conv2)

# 3rd layer: fc
# input shape is height x width x depth, output is size 1024
W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(W_fc1, [-1, 7 * 7 * 64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# 4th layer -- output: softmax
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

y = tf.matmul(h_fc1, W_fc2) + b_fc2

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_correct, logits=y))

train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_correct, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    batch = mnist.train.next_batch(50)
    for i in range(20000):
        if i % 100 == 0:
            acc = accuracy.eval({x: batch[0], y_correct: batch[1]})
            print('at iter {}, accuracy {}'.format(i, acc))
        train_step.run({x: batch[0], y_correct: batch[1]})
    print('training done')
    print('test accuracy {}'.format(accuracy.eval(
        {x: mnist.test.images, y_correct: mnist.test.labels})))
