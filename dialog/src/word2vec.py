#!/usr/bin/env python3
# coding=utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import math
import pickle
import copy
import random
import numpy as np
import tensorflow as tf
import jieba
import json
import matplotlib as mpl
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

with open("../definition/OTLG.json", 'r', encoding='utf-8') as f:
	OTLG = json.load(f)



def extract_OTLG_words(filename):
	# 添加ontology里的词
	with open(filename, 'w', encoding='utf-8') as f:
		for domain in OTLG:
			for slot in OTLG[domain]["informable"]:
				values = OTLG[domain]["informable"][slot]
				values.sort(key=lambda x: len(x))
				for value in values:
					f.write(value + "\n")
			for slot in OTLG[domain]["requestable"]:
				f.write(slot + "\n")
			f.write(domain + "\n")


def del_redundant_words(filename):
	with open(filename, 'r', encoding="utf-8") as f:
		words = json.load(f)
		for word in words:
			jieba.del_word(word)
			
			
extract_OTLG_words('../data/word2vec/OTLG_words.txt')
jieba.load_userdict("../data/word2vec/OTLG_words.txt")
jieba.load_userdict("../data/word2vec/words_to_add.txt")
del_redundant_words("../data/word2vec/words_to_del.json")

with open('../data/word2vec/OTLG_words.txt', 'r', encoding='utf-8') as f:
	OTLG_words = f.read().split()
OTLG_words.sort(key=lambda x: len(x), reverse=True)
with open('../data/word2vec/OTLG_words.json', 'w', encoding='utf-8') as f:
	json.dump(OTLG_words, f, ensure_ascii=False, indent=2)


# Step 1: Build the rawdata
def sentence2words(sentence):  # 分词
	def entity_filter(sentence, entity_set):
		for word in entity_set:
			if len(word) > 1:
				if word in sentence:
					sub_sentences = sentence.split(word)
					sentence = (' ' + word + ' ').join([entity_filter(sub_sentence, entity_set) for sub_sentence in sub_sentences])
					break
		return sentence
	sentence = sentence.replace('《', '').replace('》 ', '')
	words = ' '.join(jieba.cut(entity_filter(sentence, OTLG_words) + '。', HMM=False)).split()
	return words

def generate_raw_data(filename, rounds_dialog):
	"""生成经过分词后的数据集"""
	raw_data = []
	sentences = []
	for round in rounds_dialog:
		sentences.append(round["user_nl"])
	random.shuffle(sentences)
	
	max_len = 0
	for sentence in sentences:
		words = sentence2words(sentence)
		max_len = max(max_len, len(words))
		raw_data.extend(words)
	print("max length of all sentence: ", max_len)
	with open(filename, 'w', encoding='utf-8') as f:
		json.dump(raw_data, f, ensure_ascii=False, indent=2)
	return raw_data, max_len

with open("../data/rounds_dialog_20000.json", 'r', encoding='utf-8') as f:
	rounds_dialog_20000 = json.load(f)
with open("../data/rounds_dialog_50000.json", 'r', encoding='utf-8') as f:
	rounds_dialog_50000 = json.load(f)
# raw_data, max_len = generate_raw_data("../data/word2vec/raw_data.json", rounds_dialog_20000 + rounds_dialog_50000)
raw_data, max_len = generate_raw_data("../data/word2vec/raw_data.json", rounds_dialog_20000)


# Step 2: Build the dictionary and replace rare words with UNK token.
def build_dataset(words):
	"""
	Process raw inputs into a dataset
	:param words: rawdata
	:return: data, count, dictionary, reversed_dictionary
	"""
	count = [['unk', -1]]
	count.extend(collections.Counter(words).most_common())
	dictionary = dict()
	for word, _ in count:
		dictionary[word] = len(dictionary)
	reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
	data = []
	unk_count = 0
	for word in words:
		index = dictionary.get(word, 0)
		if index == 0:
			unk_count += 1
		data.append(index)
	count[0][1] = unk_count
	return data, count, dictionary, reversed_dictionary


with open("../data/word2vec/raw_data.json", 'r', encoding='utf-8') as f:
	raw_data = json.load(f)
data, count, dictionary, reverse_dictionary = build_dataset(raw_data)
vocabulary_size = len(dictionary)
print("vocab size:", vocabulary_size)
with open('../data/word2vec/vocab_dict.json', 'w', encoding='utf-8') as f:
	json.dump(dictionary, f, ensure_ascii=False)
with open('../data/word2vec/reverse_vocab_dict.json', 'w', encoding='utf-8') as f:
	json.dump(reverse_dictionary, f, ensure_ascii=False)

# # 处理近义词，用于后续词向量的学习
# synonym_inputs_ = []
# with open("../data/raw_data.txt", 'r', encoding='utf-8') as f:
# 	for line in f:
# 		line = line.lstrip('\ufeff')
# 		words = line.split()
# 		if words[0] in dictionary and words[1] in dictionary:
# 			synonym_inputs_.append([dictionary[words[0]], dictionary[words[1]]])


# Step 3: Function to generate a training batch for the skip-gram model.
data_index = 0


def generate_batch(batch_size, num_skips, skip_window):
	"""
	生成训练数据
	:param batch_size: minibatch大小
	:param num_skips: 跳词数目
	:param skip_window: 窗长
	:return: 数据
	"""
	global data_index
	assert batch_size % num_skips == 0
	assert num_skips <= 2 * skip_window
	batch = np.ndarray(shape=batch_size, dtype=np.int32)
	labels = np.ndarray(shape=[batch_size, 1], dtype=np.int32)
	span = 2 * skip_window + 1  # [ skip_window target skip_window ]
	buffer = collections.deque(maxlen=span)
	for _ in range(span):
		buffer.append(data[data_index])
		data_index = (data_index + 1) % len(data)
	for i in range(batch_size // num_skips):
		context_words = [w for w in range(span) if w != skip_window]
		words_to_use = random.sample(context_words, num_skips)
		for j, context_word in enumerate(words_to_use):
			batch[i * num_skips + j] = buffer[skip_window]
			labels[i * num_skips + j, 0] = buffer[context_word]
		buffer.append(data[data_index])
		data_index = (data_index + 1) % len(data)
	# Backtrack a little bit to avoid skipping words in the end of a batch
	data_index = (data_index - span + len(data)) % len(data)
	return batch, labels


# Step 4: Build and train a skip-gram model.
batch_size = 128
embedding_size = 25  # Dimension of the embedding vector.
skip_window = 1  # How many words to consider left and right.
num_skips = 2  # How many times to reuse an input to generate a label.
num_sampled = 64  # Number of negative examples to sample.

graph = tf.Graph()
with graph.as_default():
	# Input_data
	train_inputs = tf.placeholder(dtype=tf.int32, shape=[None], name="input")
	train_labels = tf.placeholder(dtype=tf.int32, shape=[None, 1], name="label")
	
	# Ops and variables pinned to the CPU because of missing GPU implementation
	with tf.device("/cpu:0"):
		# look up embeddings for inputs.
		embeddings = tf.Variable(tf.random_uniform(shape=[vocabulary_size, embedding_size], minval=-1.0, maxval=1.0),
		                         name='embeddings')
		embed = tf.nn.embedding_lookup(params=embeddings, ids=train_inputs)
		# Construct the variables for the NCE loss
		nce_weights = tf.Variable(tf.truncated_normal(shape=[vocabulary_size, embedding_size],
		                                              stddev=1.0 / math.sqrt(embedding_size)))
		nce_bias = tf.Variable(tf.zeros(shape=[vocabulary_size]))
	# Compute the average NCE loss for the batch
	# tf.nce_loss automatically draws a new sample of the negative labels each time we evaluate the loss.
	loss = tf.reduce_mean(
		tf.nn.nce_loss(weights=nce_weights,
		               biases=nce_bias,
		               labels=train_labels,
		               inputs=embed,
		               num_sampled=num_sampled,
		               num_classes=vocabulary_size)
	)
	optimizer = tf.train.AdamOptimizer(5e-4).minimize(loss)
	norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
	normalized_embeddings = tf.div(embeddings, norm)
	# Add variable initializer.
	init = tf.global_variables_initializer()

# Step 5: Begin training.
num_steps = 400000
with tf.Session(graph=graph) as sess:
	init.run()
	print("Initialized")
	# saver = tf.train.Saver([embeddings])
	# saver.restore(sess, "../model/tmp_final/model.ckpt")
	average_loss = 0
	for step in range(num_steps):
		batch_inputs, batch_labels = generate_batch(batch_size, num_skips, skip_window)
		feed_dict = {train_inputs: batch_inputs, train_labels: batch_labels}
		_, loss_val = sess.run([optimizer, loss], feed_dict=feed_dict)
		average_loss += loss_val
		
		if step % 2000 == 0 and step:
			average_loss /= 2000
			print("Average loss at step", step, ":", average_loss)
			average_loss = 0
			
	# save_path = saver.save(sess, "../model/tmp_final/model.ckpt")
	final_embeddings_norm = normalized_embeddings.eval()
	vocab_norm = dict()
	for ii, embedding in enumerate(final_embeddings_norm):
		vocab_norm[reverse_dictionary[ii]] = embedding
	with open("../data/word2vec/vocab_norm_25d.pkl", 'wb') as f:
		pickle.dump(vocab_norm, f)
	
	final_embeddings = embeddings.eval()
	vocab = dict()
	for ii, embedding in enumerate(final_embeddings):
		vocab_norm[reverse_dictionary[ii]] = embedding
	with open("../data/word2vec/vocab_25d.pkl", 'wb') as f:
		pickle.dump(vocab, f)

# Step6 : Visualize the embeddings.
mpl.rcParams['font.sans-serif'] = ['SimHei']


def plot_with_labels(low_dim_embs, labels, filename='tsne.png'):
	assert low_dim_embs.shape[0] >= len(labels), "More labels than embeddings"
	plt.figure(figsize=(18, 18), )  # in inches
	for i, label in enumerate(labels):
		x, y = low_dim_embs[i, :]
		plt.scatter(x, y)
		plt.annotate(
			label,
			xy=(x, y),
			xytext=(5, 2),
			textcoords="offset points",
			ha='right',
			va="bottom"
		)
	plt.savefig(filename)


try:
	tsne = TSNE(perplexity=30, n_components=2, init="pca", n_iter=5000)
	for i in range(4):
		plot_only = 1000
		if plot_only * (i + 1) > vocabulary_size:
			break
		low_dim_embs = tsne.fit_transform(final_embeddings[plot_only * i: plot_only * (i+1), :])
		labels = [reverse_dictionary[i] for i in range(plot_only * i, plot_only * (i+1))]
		plot_with_labels(low_dim_embs, labels, '../data/word2vec/tsne%d.png' % i)
	
		low_dim_embs = tsne.fit_transform(final_embeddings_norm[plot_only * i: plot_only * (i+1), :])
		labels = [reverse_dictionary[i] for i in range(plot_only * i, plot_only * (i+1))]
		plot_with_labels(low_dim_embs, labels, '../data/word2vec/tsne_norm%d.png' % i)
	
	low_dim_embs = tsne.fit_transform(final_embeddings[0: vocabulary_size, :])
	labels = [reverse_dictionary[i] for i in range(vocabulary_size)]
	plot_with_labels(low_dim_embs, labels, '../data/word2vec/tsne.png')
	
	low_dim_embs = tsne.fit_transform(final_embeddings_norm[0: vocabulary_size, :])
	labels = [reverse_dictionary[i] for i in range(vocabulary_size)]
	plot_with_labels(low_dim_embs, labels, '../data/word2vec/tsne_norm.png')

except ImportError:
	print("Please install sklearn, matplotlib, and scipy to show embeddings.")
