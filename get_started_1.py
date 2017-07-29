import tensorflow as tf

x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)
W = tf.Variable(0, dtype=tf.float32)
b = tf.Variable(0, dtype=tf.float32)

linear_model = W * x + b
loss = tf.reduce_sum(tf.square(linear_model - y))

optimizer = tf.train.GradientDescentOptimizer(0.01)

train = optimizer.minimize(loss)  # so train is an gradient update?

sess = tf.Session()

init = tf.global_variables_initializer()  # what's this function

sess.run(init)

fixW = W.assign(0)
fixb = W.assign(0)

for i in range(1000):
    sess.run(train, {x: [1, 2, 3, 4], y: [0, -1, -2, -3]})

print(sess.run([W, b]))
