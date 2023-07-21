# -*- coding: utf-8 -*-
"""
Created on Sun Sep 16 21:10:52 2018

@author: hunni
"""
import tensorflow as tf
import numpy as np
import pickle
import math
import os
from tempfile import gettempdir

tf.reset_default_graph()
gbatch = []
glabels = []
def generate_input(batch_size = 128):
    global ids
    global gbatch
    global glabels
    global korids
    while len(gbatch) < batch_size :
        if len(ids) > 1:
            gbatch.append(ids[0])
            glabels.append([ids[1]])
            for i in range(1,len(ids)-1):
                gbatch.append(ids[i])
                glabels.append([ids[i-1]])
                gbatch.append(ids[i])
                glabels.append([ids[i+1]])
            gbatch.append(ids[-1])
            glabels.append([ids[-2]])
        ids = pickle.load(korids)
        if ids == [0]:
            korids.seek(0)
            ids = pickle.load(korids)
            global loop
            loop += 1
            print(loop, "loop, baby")
    batch = gbatch[:batch_size]
    labels = glabels[:batch_size]
    gbatch = gbatch[batch_size:]
    glabels = glabels[batch_size:]
    return batch, labels

graph = tf.Graph()


with open("kor_dictionary.bin", "rb") as kd:
    kdictionary = pickle.load(kd) 

embedding_size = 128
with graph.as_default():
    train_inputs = tf.placeholder(tf.int32, shape=[None])
    train_labels = tf.placeholder(tf.int32, shape=[None,1])
    
    embeddings = tf.Variable(
            tf.random_uniform(
                    [len(kdictionary), embedding_size], -1.0, 1.0))
    embed = tf.nn.embedding_lookup(embeddings, train_inputs)
    nce_weights = tf.Variable(
            tf.truncated_normal(
                    [len(kdictionary), embedding_size], 
                    stddev=1.0 / math.sqrt(embedding_size)))
    nce_biases = tf.Variable(tf.zeros((len(kdictionary))))
    loss = tf.reduce_mean(
            tf.nn.nce_loss(
                    weights=nce_weights,
                    biases=nce_biases,
                    labels=train_labels,
                    inputs=embed,
                    num_sampled=64,
                    num_classes=len(kdictionary)))
    optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss)
    norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keepdims=True))
    normalized_embeddings = embeddings / norm
    init = tf.global_variables_initializer()

loop = 0
batch_size = 128
with tf.Session(graph=graph) as session:
    init.run()
    print('시작 ㅋ')
    step = 0
    std_step = 2000
    num_step = 100000 
    average_loss = 0
    with open('kor_indexed.bin', 'rb') as korids:
        ids = pickle.load(korids)
        while step <= num_step:
            batch_inputs, batch_labels = generate_input(batch_size)
            feed_dict = {train_inputs: batch_inputs, train_labels: batch_labels}
            _, loss_val = session.run(
                    [optimizer, loss],
                    feed_dict=feed_dict)
            step += 1
            average_loss += loss_val
            if step % std_step == 0:
                average_loss /= std_step
                print('step = ', step, " loss = ", average_loss)
                average_loss = 0
    print("마지막 임베딩값을 구합니다.")
    final_embeddings = normalized_embeddings.eval()
    print("마지막 임베딩 값을 구했습니다.")
with open("kor_embedding.bin", "wb") as kor_embedding:
    pickle.dump(final_embeddings, kor_embedding)
    

def plot_with_labels(low_dim_embs, labels, filename):
  assert low_dim_embs.shape[0] >= len(labels), 'More labels than embeddings'
  plt.figure(figsize=(18, 18))  # in inches
  for i, label in enumerate(labels):
    x, y = low_dim_embs[i, :]
    plt.scatter(x, y)
    plt.annotate(
        label,
        xy=(x, y),
        xytext=(5, 2),
        textcoords='offset points',
        ha='right',
        va='bottom')

  plt.savefig(filename)

r_dict = open("kor_reverse_dictionary.bin", "rb")
reverse_dictionary = pickle.load(r_dict)
r_dict.close()

try:
  # pylint: disable=g-import-not-at-top
  from sklearn.manifold import TSNE
  import matplotlib.pyplot as plt
  from matplotlib import font_manager, rc
  font_name = font_manager.FontProperties(fname="c:/Windows/Fonts/malgun.ttf").get_name()
  rc('font', family=font_name)
  tsne = TSNE(
      perplexity=30, n_components=2, init='pca', n_iter=5000, method='exact')
  plot_only = 500
  low_dim_embs = tsne.fit_transform(final_embeddings[:plot_only, :])
  labels = [reverse_dictionary[i] for i in range(plot_only)]
  plot_with_labels(low_dim_embs, labels, os.path.join(gettempdir(), 'tsne.png'))

except ImportError as ex:
  print('Please install sklearn, matplotlib, and scipy to show embeddings.')
  print(ex)    
    