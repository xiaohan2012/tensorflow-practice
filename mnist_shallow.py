import numpy as np
import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)


x = tf.placeholder(tf.float64, [None, 784])
y_ = tf.placeholder(tf.float64, [None, 10])  # correct labels

W = tf.Variable(np.zeros((784, 10)))
b = tf.Variable(np.zeros(10))

# 1st method
# y = tf.nn.softmax(tf.matmul(x, W) + b)
# cross_entropy = tf.reduce_mean(tf.reduce_sum(- y_ * tf.log(y), reduction_indices=[1]))

# 2nd method (more numerically stable)
y = tf.matmul(x, W) + b
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))

# the update step is some assign_{operator} function
train = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

for i in range(1000):
  xs, ys = mnist.train.next_batch(100)
  sess.run(train, {x: xs, y_: ys})

correctness = tf.equal(tf.argmax(y, 1) , tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correctness, tf.float32))

print('accuracy {}'.format(sess.run(
  accuracy, {x: mnist.test.images, y_: mnist.test.labels})))

sess.close()



