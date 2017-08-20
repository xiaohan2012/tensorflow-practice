
# coding: utf-8

# In[1]:

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import math
import os
import random
import zipfile

import numpy as np
from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf


# In[2]:

filename = 'data/text8.zip'


# Read the data into a list of strings.
def read_data(filename):
    """Extract the first file enclosed in a zip file as a list of words."""
    with zipfile.ZipFile(filename) as f:
        data = tf.compat.as_str(f.read(f.namelist()[0])).split()
    return data

vocabulary = read_data(filename)
print('Data size', len(vocabulary))


# In[3]:

vocabulary_size = 5000
def build_dataset(words, n_words):
    """Process raw inputs into a dataset."""
    count = [['UNK', -1]]
    count.extend(collections.Counter(words).most_common(n_words - 1))
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)
    data = list()
    unk_count = 0
    for word in words:
        if word in dictionary:
            index = dictionary[word]
        else:
            index = 0  # dictionary['UNK']
        unk_count += 1
        data.append(index)

    count[0][1] = unk_count
    reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return data, count, dictionary, reversed_dictionary

data, count, dictionary, reverse_dictionary = build_dataset(vocabulary,
                                                            vocabulary_size)


# In[4]:

del vocabulary  # Hint to reduce memory.
print('Most common words (+UNK)', count[:5])
print('Sample data', data[:10], [reverse_dictionary[i] for i in data[:10]])


# In[5]:

# Step 3: Function to generate a training batch for the skip-gram model.
data_index = 1;
def generate_batch(batch_size, num_skips, skip_window):
    global data_index
    assert batch_size % num_skips == 0
    assert num_skips <= 2 * skip_window

    batch = np.ndarray(shape=(batch_size), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    span = 2 * skip_window + 1  # [ skip_window target skip_window ]
    buffer = collections.deque(maxlen=span)
    if data_index + span > len(data):
        # start over
        data_index = 0
    buffer.extend(data[data_index:data_index + span])
    data_index += span
    for i in range(batch_size // num_skips):
        target = skip_window  # target label at the center of the buffer
        targets_to_avoid = [skip_window]
        for j in range(num_skips):
            while target in targets_to_avoid:
                target = random.randint(0, span - 1)
            targets_to_avoid.append(target)
            batch[i * num_skips + j] = buffer[skip_window]
            labels[i * num_skips + j, 0] = buffer[target]
        if data_index == len(data):
            buffer[:] = data[:span]
            data_index = span
        else:
            buffer.append(data[data_index])
            data_index += 1
    # Backtrack a little bit to avoid skipping words in the end of a batch
    data_index = (data_index + len(data) - span) % len(data)
    return batch, labels


# In[6]:

batch, labels = generate_batch(batch_size=8, num_skips=2, skip_window=1)
for i in range(8):
    print(batch[i], reverse_dictionary[batch[i]],
          '->', labels[i, 0], reverse_dictionary[labels[i, 0]])


# In[7]:


# Step 4: Build and train a skip-gram model.

batch_size = 128
embedding_size = 128  # Dimension of the embedding vector.
skip_window = 1       # How many words to consider left and right.
num_skips = 2 # How many times to reuse an input to generate a label.

num_sampled = 64 # Number of negative examples to sample.

# validation

valid_size = 16
valid_window = 100
valid_examples = np.random.choice(valid_window, valid_size, replace=False)


# In[ ]:

train_input = tf.constant([[1,2], [0,1], [1, 3]])
# train_input = tf.constant([0, 1, 2, 3])
embedding = tf.constant(np.random.rand(4, 10))
embed = tf.nn.embedding_lookup(embedding, train_input)
with tf.Session() as sess:
    embed_val = sess.run([embed])[0]
    print(embed_val.shape)


# In[ ]:

m = tf.constant([[3, 1, 2], [0, 1, 0], [1, 1, 3]])
best = tf.nn.top_k(m, 2)[1]
with tf.Session() as sess:
    print(sess.run(best))


# In[ ]:

graph = tf.Graph()

with graph.as_default():
    train_input = tf.placeholder(tf.int32, shape=[batch_size], name="input")
    train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1], name="label")
    
    embedding = tf.Variable(tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))                            
    embed = tf.nn.embedding_lookup(embedding, train_input)
    
    nce_weight = tf.Variable(
        tf.truncated_normal([vocabulary_size, embedding_size], stddev=1/math.sqrt(embedding_size),
                           name='nce_weight'))
    nce_bias = tf.Variable(tf.zeros([vocabulary_size]), name='nce_bias')
    
    loss = tf.reduce_mean(
        tf.nn.nce_loss(nce_weight, nce_bias,
                       inputs=embed,
                       labels=train_labels, 
                       num_sampled=num_sampled,
                       num_classes=vocabulary_size))
    
    optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss)
    
    norm = tf.sqrt(tf.reduce_sum(tf.square(embedding), 1, keep_dims=True))
    normalized_embedding = embedding / norm

    # validation
    valid_embedding = tf.nn.embedding_lookup(normalized_embedding, valid_examples)
    similarity = tf.matmul(valid_embedding, normalized_embedding, transpose_b=True)
    top_k = tf.nn.top_k(similarity, 8)[1]
    init = tf.global_variables_initializer()
    


# In[ ]:

# Step 5: Begin training.
num_steps = 100001

with tf.Session(graph=graph) as sess:
    sess.run(init)
    batch_input, batch_labels = generate_batch(batch_size, num_skips, skip_window)    
    loss_sum = 0
    for step in range(num_steps):    
        _, loss_val = sess.run([optimizer, loss], feed_dict={train_input: batch_input, train_labels: batch_labels})
        loss_sum += loss_val
        if (step+1)% 2000 == 0:
            print("at {}, avg loss of last 2000 steps: {}".format(step+1, loss_sum / 2000))
            loss_sum = 0
        if (step+1) % 10000 == 0:
            top_k_val = top_k.eval()
            for i, similar_ids in zip(valid_examples, top_k_val):
                word = reverse_dictionary[i]
                similar_words = [reverse_dictionary[j] for j in similar_ids]
                print("nearest to {}: {}".format(word, " ".join(similar_words)))
                
    final_embedding = normalized_embedding.eval()

