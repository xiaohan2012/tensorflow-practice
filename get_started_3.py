import tensorflow as tf
import numpy as np


def model(features, labels, mode):
    W = tf.Variable(0, dtype=tf.float64)
    b = tf.Variable(0, dtype=tf.float64)
    x = features['x']
    y = W * x + b
    loss = tf.reduce_sum(tf.square(y - labels))
    optimizer = tf.train.GradientDescentOptimizer(0.01)

    global_step = tf.train.get_global_step()
    train = tf.group(optimizer.minimize(loss),
                     tf.assign_add(global_step, 1))  # assign add
    return tf.contrib.learn.ModelFnOps(
        mode=mode,
        predictions=y,  # eval purpose
        train_op=train,  # actual train
        loss=loss  # logging
    )
      

x_train = np.array([1., 2., 3., 4.])
y_train = np.array([0., -1., -2., -3.])
x_eval = np.array([2., 5., 8., 1.])
y_eval = np.array([-1.01, -4.1, -7, 0.])

input_fn = tf.contrib.learn.io.numpy_input_fn(
    {'x': x_train}, y_train,
    batch_size=4,  # why batch_size
    num_epochs=1000)

eval_fn = tf.contrib.learn.io.numpy_input_fn(
    {'x': x_eval}, y_eval,
    batch_size=4,  # why batch_size
    num_epochs=1000)

estimator = tf.contrib.learn.Estimator(model_fn=model)
estimator.fit(input_fn=input_fn, steps=1000)

train_loss = estimator.evaluate(input_fn=input_fn)
eval_loss = estimator.evaluate(input_fn=eval_fn)

print("train_loss: {}", train_loss)
print("eval_loss: {}", eval_loss)
