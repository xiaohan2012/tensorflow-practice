
# coding: utf-8

# In[ ]:

import tensorflow as tf
import numpy as np
import os
import time
import datetime
import data_helpers
from tensorflow.contrib import learn

from text_cnn import TextCNN


tf.flags.DEFINE_float("dev_sample_percentage", .1, "Percentage of the training data to use for validation")
tf.flags.DEFINE_string("positive_data_file", "./data/rt-polarity.pos", "Data source for the positive data.")
tf.flags.DEFINE_string("negative_data_file", "./data/rt-polarity.neg", "Data source for the negative data.")

# Model Hyperparameters
tf.flags.DEFINE_integer("embedding_dim", 128, "Dimensionality of character embedding (default: 128)")
tf.flags.DEFINE_string("filter_sizes", "3,4,5", "Comma-separated filter sizes (default: '3,4,5')")
tf.flags.DEFINE_integer("num_filters", 128, "Number of filters per filter size (default: 128)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularization lambda (default: 0.0)")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 200, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 100, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 100, "Save model after this many steps (default: 100)")
tf.flags.DEFINE_integer("num_checkpoints", 5, "Number of checkpoints to store (default: 5)")
# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")


FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")


text_x, y = data_helpers.load_data_and_labels(FLAGS.positive_data_file, FLAGS.negative_data_file)



# In[ ]:

# convert to matrix
max_doc_len = max(len(d.split()) for d in text_x)

vectorizer = learn.preprocessing.VocabularyProcessor(max_doc_len)
x = np.array(list(vectorizer.fit_transform(text_x)))


# In[ ]:

np.random.seed(12345)

ind = np.random.permutation(x.shape[0])
shuffled_x = x[ind, :]
shuffled_y = y[ind, :]

idx = int(FLAGS.dev_sample_percentage * x.shape[0])
train_x, train_y = shuffled_x[idx:, :], shuffled_y[idx:, :]
dev_x, dev_y = shuffled_x[:idx, :], shuffled_y[:idx, :]


# In[ ]:

with tf.Session() as sess:
    cnn = TextCNN(max_doc_len, 2, FLAGS.embedding_dim, len(vectorizer.vocabulary_), 
                 list(map(int, FLAGS.filter_sizes.split(','))), 
                 FLAGS.num_filters)            
    # train operation
    global_step = tf.Variable(0, name="global_step", trainable=False)
    optimizer = tf.train.AdamOptimizer(1e-3)
    grads_and_vars = optimizer.compute_gradients(cnn.loss)
    train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)
    
    # IO direction stuff
    timestamp = str(int(time.time()))    
    out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))    
    
    # summary writer
    train_summary_dir = os.path.join(out_dir, "summary/train")
    train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)
    
    dev_summary_dir = os.path.join(out_dir, "summary/dev")
    dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)
    
    # checkpoint writer
    checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))    
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)    
    checkpoint_prefix = os.path.join(checkpoint_dir, "model")
    saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)
    
    # summary operation
    grad_summaries = []
    for grad, v in grads_and_vars:
        if grad is not None:
            hist = tf.summary.histogram("{}/grad/hist".format(v.name), grad)
            sparsity = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(grad))
            grad_summaries.append(hist)
            grad_summaries.append(sparsity)
    grad_summary = tf.summary.merge(grad_summaries)
    
    acc_summary = tf.summary.scalar("accuracy", cnn.accuracy)
    loss_summary = tf.summary.scalar("loss", cnn.loss)
    
    train_summary_op = tf.summary.merge([grad_summary, acc_summary, loss_summary])
    dev_summary_op = tf.summary.merge([acc_summary, loss_summary])    
    
    # save vocabulary
    vectorizer.save(os.path.join(out_dir, "vocab")) 
    
    def train_step(batch_x, batch_y, writer=None):
        feed_dict = {
            cnn.input_x: batch_x, 
            cnn.input_y: batch_y,
            cnn.dropout_keep_prob: FLAGS.dropout_keep_prob
        }
        _, step, summaries, loss, accuracy = sess.run(
            [train_op, global_step, train_summary_op, cnn.loss, cnn.accuracy],
            feed_dict=feed_dict)
        time_str = datetime.datetime.now().isoformat()
        # print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))

        if writer:
            writer.add_summary(summaries, step)

            
    def dev_step(batch_x, batch_y, writer=None):
        feed_dict = {
            cnn.input_x: batch_x, 
            cnn.input_y: batch_y,
            cnn.dropout_keep_prob: 1.0
        }
        step, summaries, loss, accuracy = sess.run(
            [global_step, dev_summary_op, cnn.loss, cnn.accuracy],
            feed_dict=feed_dict)
        time_str = datetime.datetime.now().isoformat()
        print("dev {}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))

        if writer:
            writer.add_summary(summaries, step)

    sess.run(tf.global_variables_initializer())

    data = list(zip(train_x, train_y))
    batches = data_helpers.batch_iter(data, FLAGS.batch_size, FLAGS.num_epochs, shuffle=True)
    for batch in batches:
        batch_x, batch_y = zip(*batch)

        current_step = tf.train.global_step(sess, global_step)
        train_step(batch_x, batch_y, writer=train_summary_writer)
        
        if current_step % FLAGS.evaluate_every == 0:
            dev_step(dev_x, dev_y, writer=dev_summary_writer)

        if current_step % FLAGS.checkpoint_every == 0:
            saver.save(sess, checkpoint_prefix, global_step=global_step)

