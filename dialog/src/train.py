#!/usr/bin/env python3
# coding=utf-8


import tensorflow as tf
from input_data import read_data_sets
import numpy as np
import os
import json
import copy
import re
import pickle
import random
import pprint
import collections
import jieba
from six.moves import xrange

# word embedding
print('prepare the vocabulary ...')
embedding_dim = 25
with open('../../data/word2vec/vocab_dict.json', 'r', encoding='utf-8') as f:
	vocab_dict = json.load(f)
with open('../../data/word2vec/vocab_norm_25d.pkl', 'rb') as f:
	embedding_table = pickle.load(f)
# semantic dict
with open('../../data/word2vec/semantic_dict_domain.json', 'r', encoding='utf-8') as f:
	semantic_dict_domain = json.load(f)
with open('../../data/word2vec/semantic_dict_request_slot.json', 'r', encoding='utf-8') as f:
	semantic_dict_request_slot = json.load(f)
with open('../../data/word2vec/semantic_dict_inform_slot.json', 'r', encoding='utf-8') as f:
	semantic_dict_inform_slot = json.load(f)
with open('../../data/word2vec/semantic_dict_value.json', 'r', encoding='utf-8') as f:
	semantic_dict_value = json.load(f)
with open('../../data/word2vec/semantic_dict_act.json', 'r', encoding='utf-8') as f:
	semantic_dict_act = json.load(f)
# ontology
with open('../../definition/OTLG.json', 'r', encoding='utf-8') as f:
	OTLG = json.load(f)
print('Done')
# prepare datasets
print('prepare the datasets...')
datasets = read_data_sets()
print('Done')
# domains, slots,
domain_set = ["电影", "音乐", "天气", "时间"]
request_slots = {domain: OTLG[domain]['requestable'] for domain in domain_set}
inform_slots = {domain: set(OTLG[domain]['informable'].keys()) - set(OTLG[domain]['major_key']) for domain in
                ['电影', '音乐']}
confirm_slots = {domain: set(OTLG[domain]['informable'].keys()) - set(OTLG[domain]['major_key']) for domain in
                 ['电影', '音乐']}
user_acts = ['first', 'second', 'third', 'last', 'other', 'affirm', 'deny']
# informable_slots = list(OTGY["informable"].keys())
# requestable_slots = OTGY["requestable"]

# labels
domain_label = ["MENTIONED", "NOT_MENTIONED"]
request_slot_label = ["MENTIONED", "NOT_MENTIONED"]
inform_slot_label = ["MENTIONED", "NOT_MENTIONED", "DONT_CARE"]
value_label = ["LIKE", "DISLIKE"]
act_label = ["MENTIONED", "NOT_MENTIONED"]

# parameters
learning_rate = 0.0005
display_step = 100
max_length = 60
batchsize_domain = {"MENTIONED": 50, "NOT_MENTIONED": 206}  # 1:4
batchsize_request_slot = {"MENTIONED": 32, "NOT_MENTIONED": 224}  # 1:7
batchsize_inform_slot = {"MENTIONED": 64, "NOT_MENTIONED": 160, 'DONT_CARE': 32}  # 2:5:1
batchsize_value = {"LIKE": 180, "DISLIKE": 76}  # 7:3
batchsize_confirm_slot = {"MENTIONED": 32, "NOT_MENTIONED": 224}  # 1:7
batchsize_act = {"MENTIONED": 32, "NOT_MENTIONED": 224}  # 1:7
hidden_size = 100


class DomainTracker(object):
	def __init__(self):
		# tf Graph input
		self._x_user_nl = tf.placeholder(dtype=tf.float32, shape=(None, max_length, embedding_dim))
		self._x_stringmatch = tf.placeholder(dtype=tf.float32, shape=(None, max_length, 1))  # whether MENTIONED
		# [request, confirm, inform, inform_no_match, inform_one_match, inform_some_match, none]
		self._x_sys_act = tf.placeholder(dtype=tf.float32, shape=(None, 7))
		self._is_training = tf.placeholder(dtype=tf.bool)
		self._y = tf.placeholder(dtype=tf.float32, shape=(None, 2))
		self._lr = tf.Variable(0.0003, trainable=False)
		
		self._input_cnn = tf.concat([self._x_user_nl, self._x_stringmatch], 2)  # -> (?, max_length, embedding_size + 1)
		kernel_sizes = [1, 2, 3]
		self._pools = []
		for kernel_size in kernel_sizes:
			self._conv = tf.layers.conv1d(
				inputs=self._input_cnn,
				filters=100,
				kernel_size=kernel_size,
				strides=1,
				padding='same',
				activation=tf.nn.relu,
				kernel_initializer=tf.random_normal_initializer(0, 0.01))
			self._pool = tf.layers.max_pooling1d(self._conv, pool_size=max_length, strides=1)
			self._pool = tf.squeeze(self._pool, [1])
			self._pools.append(self._pool)
		self._output_cnn = tf.concat(values=self._pools, axis=1)
		self._dnn_hiddenlayer = tf.layers.dense(inputs=self._x_sys_act,
		                                        units=self._x_sys_act.shape[1] * 2,
		                                        activation=tf.nn.sigmoid,
		                                        kernel_initializer=tf.random_normal_initializer(0, 0.5))
		self._output_cnn_W = tf.layers.dense(inputs=self._dnn_hiddenlayer, units=1,
		                                     activation=tf.nn.sigmoid,
		                                     kernel_initializer=tf.random_normal_initializer(0, 0.5), )
		self._user_nl_code = tf.multiply(self._output_cnn_W, self._output_cnn)
		# self._user_code = self._output_cnn
		
		self._dnn_inputs = []
		for i in range(self._x_sys_act.shape[1]):
			self._dnn_inputs.append(self._user_nl_code * tf.expand_dims(self._x_sys_act[:, i], axis=1))
		self._dnn_inputs = tf.concat(self._dnn_inputs, 1)
		
		self._dnn_hiddenlayer = tf.layers.dense(inputs=self._dnn_inputs, units=500, activation=tf.nn.relu,
		                                        kernel_initializer=tf.random_normal_initializer(0, 0.1))
		self._dnn_hiddenlayer = tf.layers.dropout(self._dnn_hiddenlayer, rate=0.5, training=self._is_training)
		self._pred = tf.layers.dense(inputs=self._dnn_hiddenlayer, units=2,
		                             kernel_initializer=tf.random_normal_initializer(0, 0.1))
		self._loss = tf.losses.softmax_cross_entropy(onehot_labels=self._y, logits=self._pred)  # compute cost
		
		# gradient clipping
		self._tvars = tf.trainable_variables()
		self._grads, _ = tf.clip_by_global_norm(tf.gradients(self._loss, self._tvars), clip_norm=5)
		self._optimizer = tf.train.AdamOptimizer(self._lr)
		self._train_op = self._optimizer.apply_gradients(
			grads_and_vars=zip(self._grads, self._tvars),
			global_step=tf.train.get_or_create_global_step())
		# learning rate update
		self._new_lr = tf.placeholder(dtype=tf.float32, shape=[], name="new_learning_rate")
		self._lr_update = tf.assign(self._lr, self._new_lr)
		
		# Evaluate model
		self._results = tf.argmax(self._pred, 1)
		self._probability = tf.nn.softmax(self._pred)
		self._correct_pred = tf.equal(self._results, tf.argmax(self._y, 1))
		self._accuracy = tf.reduce_mean(tf.cast(self._correct_pred, tf.float32))
	
	def assign_lr(self, session, lr_value):
		session.run(self._lr_update, feed_dict={self._new_lr: lr_value})
	
	def accuracy(self):
		return self._accuracy
	
	def train_op(self):
		return self._train_op
	
	def loss(self):
		return self._loss


class RequestSlotTracker(object):
	def __init__(self):
		# tf Graph input
		self._x_user_nl = tf.placeholder(dtype=tf.float32, shape=(None, max_length, embedding_dim))
		self._x_stringmatch = tf.placeholder(dtype=tf.float32, shape=(None, max_length, 1))  # whether MENTIONED
		self._is_training = tf.placeholder(dtype=tf.bool)
		self._y = tf.placeholder(dtype=tf.float32, shape=(None, 2))
		self._lr = tf.Variable(0.0003, trainable=False)
		
		self._input_cnn = tf.concat([self._x_user_nl, self._x_stringmatch], 2)  # -> (?, max_length, embedding_size + 1)
		kernel_sizes = [1, 2, 3]
		self._pools = []
		for kernel_size in kernel_sizes:
			self._conv = tf.layers.conv1d(
				inputs=self._input_cnn,
				filters=100,
				kernel_size=kernel_size,
				strides=1,
				padding='same',
				activation=tf.nn.relu,
				kernel_initializer=tf.random_normal_initializer(0, 0.01))
			self._pool = tf.layers.max_pooling1d(self._conv, pool_size=max_length, strides=1)
			self._pool = tf.squeeze(self._pool, [1])
			self._pools.append(self._pool)
		self._output_cnn = tf.concat(values=self._pools, axis=1)
		
		self._dnn_hiddenlayer = tf.layers.dense(inputs=self._output_cnn, units=500, activation=tf.nn.relu,
		                                        kernel_initializer=tf.random_normal_initializer(0, 0.1))
		self._dnn_hiddenlayer = tf.layers.dropout(self._dnn_hiddenlayer, rate=0.5, training=self._is_training)
		self._pred = tf.layers.dense(inputs=self._dnn_hiddenlayer, units=2,
		                             kernel_initializer=tf.random_normal_initializer(0, 0.1))
		self._loss = tf.losses.softmax_cross_entropy(onehot_labels=self._y, logits=self._pred)  # compute cost
		
		# gradient clipping
		self._tvars = tf.trainable_variables()
		self._grads, _ = tf.clip_by_global_norm(tf.gradients(self._loss, self._tvars), clip_norm=5)
		self._optimizer = tf.train.AdamOptimizer(self._lr)
		self._train_op = self._optimizer.apply_gradients(
			grads_and_vars=zip(self._grads, self._tvars),
			global_step=tf.train.get_or_create_global_step())
		# learning rate update
		self._new_lr = tf.placeholder(dtype=tf.float32, shape=[], name="new_learning_rate")
		self._lr_update = tf.assign(self._lr, self._new_lr)
		
		# Evaluate model
		self._results = tf.argmax(self._pred, 1)
		self._probability = tf.nn.softmax(self._pred)
		self._correct_pred = tf.equal(self._results, tf.argmax(self._y, 1))
		self._accuracy = tf.reduce_mean(tf.cast(self._correct_pred, tf.float32))
	
	def assign_lr(self, session, lr_value):
		session.run(self._lr_update, feed_dict={self._new_lr: lr_value})
	
	def accuracy(self):
		return self._accuracy
	
	def train_op(self):
		return self._train_op
	
	def loss(self):
		return self._loss


class InformSlotTracker(object):
	def __init__(self):
		# tf Graph input
		self._x_user_nl = tf.placeholder(dtype=tf.float32, shape=(None, max_length, embedding_dim))
		self._x_stringmatch = tf.placeholder(dtype=tf.float32, shape=(None, max_length, 1))  # whether MENTIONED
		# [request, confirm, inform]
		self._x_sys_act = tf.placeholder(dtype=tf.float32, shape=(None, 2))
		self._is_training = tf.placeholder(dtype=tf.bool)
		self._y = tf.placeholder(dtype=tf.float32, shape=(None, 3))
		self._lr = tf.Variable(0.0003, trainable=False)
		
		self._input_cnn = tf.concat([self._x_user_nl, self._x_stringmatch], 2)  # -> (?, max_length, embedding_size + 1)
		kernel_sizes = [1, 2, 3]
		self._pools = []
		for kernel_size in kernel_sizes:
			self._conv = tf.layers.conv1d(
				inputs=self._input_cnn,
				filters=100,
				kernel_size=kernel_size,
				strides=1,
				padding='same',
				activation=tf.nn.relu,
				kernel_initializer=tf.random_normal_initializer(0, 0.01))
			self._pool = tf.layers.max_pooling1d(self._conv, pool_size=max_length, strides=1)
			self._pool = tf.squeeze(self._pool, [1])
			self._pools.append(self._pool)
		self._output_cnn = tf.concat(values=self._pools, axis=1)
		self._dnn_hiddenlayer = tf.layers.dense(inputs=self._x_sys_act,
		                                        units=self._x_sys_act.shape[1] * 2,
		                                        activation=tf.nn.sigmoid,
		                                        kernel_initializer=tf.random_normal_initializer(0, 0.5))
		self._output_cnn_W = tf.layers.dense(inputs=self._dnn_hiddenlayer, units=1,
		                                     activation=tf.nn.sigmoid,
		                                     kernel_initializer=tf.random_normal_initializer(0, 0.5), )
		self._user_nl_code = tf.multiply(self._output_cnn_W, self._output_cnn)
		# self._user_code = self._output_cnn
		
		self._dnn_inputs = []
		for i in range(self._x_sys_act.shape[1]):
			self._dnn_inputs.append(self._user_nl_code * tf.expand_dims(self._x_sys_act[:, i], axis=1))
		self._dnn_inputs = tf.concat(self._dnn_inputs, 1)
		
		self._dnn_hiddenlayer = tf.layers.dense(inputs=self._dnn_inputs, units=500, activation=tf.nn.relu,
		                                        kernel_initializer=tf.random_normal_initializer(0, 0.1))
		self._dnn_hiddenlayer = tf.layers.dropout(self._dnn_hiddenlayer, rate=0.5, training=self._is_training)
		self._pred = tf.layers.dense(inputs=self._dnn_hiddenlayer, units=3,
		                             kernel_initializer=tf.random_normal_initializer(0, 0.1))
		self._loss = tf.losses.softmax_cross_entropy(onehot_labels=self._y, logits=self._pred)  # compute cost
		
		# gradient clipping
		self._tvars = tf.trainable_variables()
		self._grads, _ = tf.clip_by_global_norm(tf.gradients(self._loss, self._tvars), clip_norm=5)
		self._optimizer = tf.train.AdamOptimizer(self._lr)
		self._train_op = self._optimizer.apply_gradients(
			grads_and_vars=zip(self._grads, self._tvars),
			global_step=tf.train.get_or_create_global_step())
		# learning rate update
		self._new_lr = tf.placeholder(dtype=tf.float32, shape=[], name="new_learning_rate")
		self._lr_update = tf.assign(self._lr, self._new_lr)
		
		# Evaluate model
		self._results = tf.argmax(self._pred, 1)
		self._probability = tf.nn.softmax(self._pred)
		self._correct_pred = tf.equal(self._results, tf.argmax(self._y, 1))
		self._accuracy = tf.reduce_mean(tf.cast(self._correct_pred, tf.float32))
	
	def assign_lr(self, session, lr_value):
		session.run(self._lr_update, feed_dict={self._new_lr: lr_value})
	
	def accuracy(self):
		return self._accuracy
	
	def train_op(self):
		return self._train_op
	
	def loss(self):
		return self._loss


class ValueTracker(object):
	def __init__(self):
		# tf Graph input
		self._x_user_nl = tf.placeholder(dtype=tf.float32, shape=(None, max_length, embedding_dim))
		self._x_stringmatch = tf.placeholder(dtype=tf.float32, shape=(None, max_length, 1))  # whether MENTIONED
		self._is_training = tf.placeholder(dtype=tf.bool)
		self._y = tf.placeholder(dtype=tf.float32, shape=(None, 2))
		self._lr = tf.Variable(0.0001, trainable=False)
		
		self._input_cnn = tf.concat([self._x_user_nl, self._x_stringmatch], 2)  # -> (?, max_length, embedding_size + 1)
		kernel_sizes = [1, 2, 3, 4]
		self._pools = []
		for kernel_size in kernel_sizes:
			self._conv = tf.layers.conv1d(
				inputs=self._input_cnn,
				filters=100,
				kernel_size=kernel_size,
				strides=1,
				padding='same',
				activation=tf.nn.relu,
				kernel_initializer=tf.random_normal_initializer(0, 0.01))
			self._pool = tf.layers.max_pooling1d(self._conv, pool_size=max_length, strides=1)
			self._pool = tf.squeeze(self._pool, [1])
			self._pools.append(self._pool)
		self._output_cnn = tf.concat(values=self._pools, axis=1)
		
		self._dnn_hiddenlayer = tf.layers.dense(inputs=self._output_cnn, units=500, activation=tf.nn.relu,
		                                        kernel_initializer=tf.random_normal_initializer(0, 0.1))
		self._dnn_hiddenlayer = tf.layers.dropout(self._dnn_hiddenlayer, rate=0.5, training=self._is_training)
		self._pred = tf.layers.dense(inputs=self._dnn_hiddenlayer, units=2,
		                             kernel_initializer=tf.random_normal_initializer(0, 0.1))
		self._loss = tf.losses.softmax_cross_entropy(onehot_labels=self._y, logits=self._pred)  # compute cost
		
		# gradient clipping
		self._tvars = tf.trainable_variables()
		self._grads, _ = tf.clip_by_global_norm(tf.gradients(self._loss, self._tvars), clip_norm=5)
		self._optimizer = tf.train.AdamOptimizer(self._lr)
		self._train_op = self._optimizer.apply_gradients(
			grads_and_vars=zip(self._grads, self._tvars),
			global_step=tf.train.get_or_create_global_step())
		# learning rate update
		self._new_lr = tf.placeholder(dtype=tf.float32, shape=[], name="new_learning_rate")
		self._lr_update = tf.assign(self._lr, self._new_lr)
		
		# Evaluate model
		self._results = tf.argmax(self._pred, 1)
		self._probability = tf.nn.softmax(self._pred)
		self._correct_pred = tf.equal(self._results, tf.argmax(self._y, 1))
		self._accuracy = tf.reduce_mean(tf.cast(self._correct_pred, tf.float32))
	
	def assign_lr(self, session, lr_value):
		session.run(self._lr_update, feed_dict={self._new_lr: lr_value})
	
	def accuracy(self):
		return self._accuracy
	
	def train_op(self):
		return self._train_op
	
	def loss(self):
		return self._loss


class ConfirmSlotTracker(object):
	def __init__(self):
		# tf Graph input
		self._x_user_nl = tf.placeholder(dtype=tf.float32, shape=(None, max_length, embedding_dim))
		self._x_stringmatch = tf.placeholder(dtype=tf.float32, shape=(None, max_length, 1))  # whether MENTIONED
		self._is_training = tf.placeholder(dtype=tf.bool)
		self._y = tf.placeholder(dtype=tf.float32, shape=(None, 2))
		self._lr = tf.Variable(0.0003, trainable=False)
		
		self._input_cnn = tf.concat([self._x_user_nl, self._x_stringmatch], 2)  # -> (?, max_length, embedding_size + 1)
		kernel_sizes = [1, 2, 3]
		self._pools = []
		for kernel_size in kernel_sizes:
			self._conv = tf.layers.conv1d(
				inputs=self._input_cnn,
				filters=100,
				kernel_size=kernel_size,
				strides=1,
				padding='same',
				activation=tf.nn.relu,
				kernel_initializer=tf.random_normal_initializer(0, 0.01))
			self._pool = tf.layers.max_pooling1d(self._conv, pool_size=max_length, strides=1)
			self._pool = tf.squeeze(self._pool, [1])
			self._pools.append(self._pool)
		self._output_cnn = tf.concat(values=self._pools, axis=1)
		
		self._dnn_hiddenlayer = tf.layers.dense(inputs=self._output_cnn, units=500, activation=tf.nn.relu,
		                                        kernel_initializer=tf.random_normal_initializer(0, 0.1))
		self._dnn_hiddenlayer = tf.layers.dropout(self._dnn_hiddenlayer, rate=0.5, training=self._is_training)
		self._pred = tf.layers.dense(inputs=self._dnn_hiddenlayer, units=2,
		                             kernel_initializer=tf.random_normal_initializer(0, 0.1))
		self._loss = tf.losses.softmax_cross_entropy(onehot_labels=self._y, logits=self._pred)  # compute cost
		
		# gradient clipping
		self._tvars = tf.trainable_variables()
		self._grads, _ = tf.clip_by_global_norm(tf.gradients(self._loss, self._tvars), clip_norm=5)
		self._optimizer = tf.train.AdamOptimizer(self._lr)
		self._train_op = self._optimizer.apply_gradients(
			grads_and_vars=zip(self._grads, self._tvars),
			global_step=tf.train.get_or_create_global_step())
		# learning rate update
		self._new_lr = tf.placeholder(dtype=tf.float32, shape=[], name="new_learning_rate")
		self._lr_update = tf.assign(self._lr, self._new_lr)
		
		# Evaluate model
		self._results = tf.argmax(self._pred, 1)
		self._probability = tf.nn.softmax(self._pred)
		self._correct_pred = tf.equal(self._results, tf.argmax(self._y, 1))
		self._accuracy = tf.reduce_mean(tf.cast(self._correct_pred, tf.float32))
	
	def assign_lr(self, session, lr_value):
		session.run(self._lr_update, feed_dict={self._new_lr: lr_value})
	
	def accuracy(self):
		return self._accuracy
	
	def train_op(self):
		return self._train_op
	
	def loss(self):
		return self._loss


class ActTracker(object):
	def __init__(self):
		# tf Graph input
		self._x_user_nl = tf.placeholder(dtype=tf.float32, shape=(None, max_length, embedding_dim))
		self._x_stringmatch = tf.placeholder(dtype=tf.float32, shape=(None, max_length, 1))  # whether INCLUSIVE
		self._is_training = tf.placeholder(dtype=tf.bool)
		self._y = tf.placeholder(dtype=tf.float32, shape=(None, 2))
		self._lr = tf.Variable(0.0003, trainable=False)
		
		self._input_cnn = tf.concat([self._x_user_nl, self._x_stringmatch], 2)  # -> (?, max_length, embedding_size + 1)
		kernel_sizes = [1, 2, 3]
		self._pools = []
		for kernel_size in kernel_sizes:
			self._conv = tf.layers.conv1d(
				inputs=self._input_cnn,
				filters=100,
				kernel_size=kernel_size,
				strides=1,
				padding='same',
				activation=tf.nn.relu,
				kernel_initializer=tf.random_normal_initializer(0, 0.01))
			self._pool = tf.layers.max_pooling1d(self._conv, pool_size=max_length, strides=1)
			self._pool = tf.squeeze(self._pool, [1])
			self._pools.append(self._pool)
		self._output_cnn = tf.concat(values=self._pools, axis=1)
		
		self._dnn_hiddenlayer = tf.layers.dense(inputs=self._output_cnn, units=500, activation=tf.nn.relu,
		                                        kernel_initializer=tf.random_normal_initializer(0, 0.1))
		self._dnn_hiddenlayer = tf.layers.dropout(self._dnn_hiddenlayer, rate=0.5, training=self._is_training)
		self._pred = tf.layers.dense(inputs=self._dnn_hiddenlayer, units=2,
		                             kernel_initializer=tf.random_normal_initializer(0, 0.1))
		self._loss = tf.losses.softmax_cross_entropy(onehot_labels=self._y, logits=self._pred)  # compute cost
		
		# gradient clipping
		self._tvars = tf.trainable_variables()
		self._grads, _ = tf.clip_by_global_norm(tf.gradients(self._loss, self._tvars), clip_norm=5)
		self._optimizer = tf.train.AdamOptimizer(self._lr)
		self._train_op = self._optimizer.apply_gradients(
			grads_and_vars=zip(self._grads, self._tvars),
			global_step=tf.train.get_or_create_global_step())
		# learning rate update
		self._new_lr = tf.placeholder(dtype=tf.float32, shape=[], name="new_learning_rate")
		self._lr_update = tf.assign(self._lr, self._new_lr)
		
		# Evaluate model
		self._results = tf.argmax(self._pred, 1)
		self._probability = tf.nn.softmax(self._pred)
		self._correct_pred = tf.equal(self._results, tf.argmax(self._y, 1))
		self._accuracy = tf.reduce_mean(tf.cast(self._correct_pred, tf.float32))
	
	def assign_lr(self, session, lr_value):
		session.run(self._lr_update, feed_dict={self._new_lr: lr_value})
	
	def accuracy(self):
		return self._accuracy
	
	def train_op(self):
		return self._train_op
	
	def loss(self):
		return self._loss


def data2num_domain(batch_data, has_label=False):
	def is_zh(content):
		zhPattern = re.compile(u'^[\u4e00-\u9fa5]+$')
		match = zhPattern.search(content)
		return True if match else False
	
	batchsize = len(batch_data)
	x_user_nl = np.zeros((batchsize, max_length, embedding_dim))
	x_stringmatch = np.zeros((batchsize, max_length, 1))
	# [request, confirm, inform, inform_no_match, inform_one_match, inform_some_match, none]
	x_sys_act = np.zeros((batchsize, 7))
	y = np.zeros((batchsize, 2))
	
	for batch_id, data in enumerate(batch_data):
		# x_user_nl x_stringmatch
		words = filter(lambda x: is_zh(x), data['user_nl'])
		for word_id, word in enumerate(words):
			if word_id == max_length:
				break
			if word in vocab_dict:
				x_user_nl[batch_id, word_id, :] = embedding_table[word]
			else:
				x_user_nl[batch_id, word_id, :] = embedding_table['unk']
			if word in semantic_dict_domain[data["domain"]]:
				x_stringmatch[batch_id, word_id, 0] = 1
		
		# x_sys_act
		sys_act_set = ["request", "confirm", "inform", "inform_no_match", "inform_one_match", "inform_some_match"]
		for act_id, sys_act in enumerate(sys_act_set):
			if sys_act in data["sys_act"]:
				if data["domain"] in data["sys_act"][sys_act]:
					x_sys_act[batch_id, act_id] = 1.0
		if sum(x_sys_act[batch_id, :]) == 0:
			x_sys_act[batch_id, len(sys_act_set)] = 1.0
		
		# y
		if has_label:
			label_domain = {'MENTIONED': 0, 'NOT_MENTIONED': 1}
			y[batch_id, label_domain[data['label']]] = 1
	if has_label:
		return {'user_nl': x_user_nl, 'stringmatch': x_stringmatch, 'sys_act': x_sys_act, 'label': y}
	else:
		return {'user_nl': x_user_nl, 'stringmatch': x_stringmatch, 'sys_act': x_sys_act}


def data2num_request_slot(batch_data, has_label=False):
	def is_zh(content):
		zhPattern = re.compile(u'^[\u4e00-\u9fa5]+$')
		match = zhPattern.search(content)
		return True if match else False
	
	batchsize = len(batch_data)
	x_user_nl = np.zeros((batchsize, max_length, embedding_dim))
	x_stringmatch = np.zeros((batchsize, max_length, 1))
	x_user_nl_len = np.zeros(batchsize, dtype='int32')
	y = np.zeros((batchsize, 2))
	
	for batch_id, data in enumerate(batch_data):
		# x_user_nl, x_stringmatch, x_user_nl_len
		words = filter(lambda x: is_zh(x), data['user_nl'])
		for word_id, word in enumerate(words):
			if word_id == max_length:
				break
			if word in vocab_dict:
				x_user_nl[batch_id, word_id, :] = embedding_table[word]
			else:
				x_user_nl[batch_id, word_id, :] = embedding_table['unk']
			if word in semantic_dict_request_slot[data['domain']][data['slot']]:
				x_stringmatch[batch_id, word_id, 0] = 1
		x_user_nl_len[batch_id] = len(data['user_nl'])
		
		# y
		if has_label:
			label_requestable_slot = {'MENTIONED': 0, 'NOT_MENTIONED': 1}
			y[batch_id, label_requestable_slot[data['label']]] = 1
	if has_label:
		return {'user_nl': x_user_nl, 'stringmatch': x_stringmatch, 'user_nl_len': x_user_nl_len, 'label': y}
	else:
		return {'user_nl': x_user_nl, 'stringmatch': x_stringmatch, 'user_nl_len': x_user_nl_len}


def data2num_inform_slot(batch_data, has_label=False):
	def is_zh(content):
		zhPattern = re.compile(u'^[\u4e00-\u9fa5]+$')
		match = zhPattern.search(content)
		return True if match else False
	
	batchsize = len(batch_data)
	x_user_nl = np.zeros((batchsize, max_length, embedding_dim))
	x_stringmatch = np.zeros((batchsize, max_length, 1))
	# [request, confirm, inform]
	x_sys_act = np.zeros((batchsize, 2))
	y = np.zeros((batchsize, 3))
	
	for batch_id, data in enumerate(batch_data):
		# x_user_nl x_stringmatch
		words = filter(lambda x: is_zh(x), data['user_nl'])
		for word_id, word in enumerate(words):
			if word_id == max_length:
				break
			if word in vocab_dict:
				x_user_nl[batch_id, word_id, :] = embedding_table[word]
			else:
				x_user_nl[batch_id, word_id, :] = embedding_table['unk']
			if word in semantic_dict_inform_slot[data['domain']][data['slot']]:
				x_stringmatch[batch_id, word_id, 0] = 1
		
		# x_sys_act
		sys_act_set = ["request"]
		for act_id, sys_act in enumerate(sys_act_set):
			if sys_act in data["sys_act"]:
				if data["domain"] in data["sys_act"][sys_act]:
					if data["slot"] in data["sys_act"][sys_act][data["domain"]]:
						x_sys_act[batch_id, act_id] = 1.0
		if sum(x_sys_act[batch_id, :]) == 0:
			x_sys_act[batch_id, len(sys_act_set)] = 1.0
		
		# y
		if has_label:
			label_domain = {'MENTIONED': 0, 'NOT_MENTIONED': 1, 'DONT_CARE': 2}
			y[batch_id, label_domain[data['label']]] = 1
	if has_label:
		return {'user_nl': x_user_nl, 'stringmatch': x_stringmatch, 'sys_act': x_sys_act, 'label': y}
	else:
		return {'user_nl': x_user_nl, 'stringmatch': x_stringmatch, 'sys_act': x_sys_act}


def data2num_value(batch_data, has_label=False):
	def is_zh(content):
		zhPattern = re.compile(u'^[\u4e00-\u9fa5]+$')
		match = zhPattern.search(content)
		return True if match else False
	
	batchsize = len(batch_data)
	x_user_nl = np.zeros((batchsize, max_length, embedding_dim))
	x_stringmatch = np.zeros((batchsize, max_length, 1))
	y = np.zeros((batchsize, 2))
	
	for batch_id, data in enumerate(batch_data):
		# x_user_nl, x_stringmatch
		words = filter(lambda x: is_zh(x), data['user_nl'])
		for word_id, word in enumerate(words):
			if word_id == max_length:
				break
			if word == data['value']:
				x_user_nl[batch_id, word_id, :] = np.ones(embedding_dim)
			elif word in vocab_dict:
				x_user_nl[batch_id, word_id, :] = embedding_table[word]
			else:
				x_user_nl[batch_id, word_id, :] = embedding_table['unk']
			if word in semantic_dict_value['LIKE']:
				x_stringmatch[batch_id, word_id, 0] = 1
			elif word in semantic_dict_value['DISLIKE']:
				x_stringmatch[batch_id, word_id, 0] = -1
		
		# y
		if has_label:
			label_value = {'LIKE': 0, 'DISLIKE': 1}
			y[batch_id, label_value[data['label']]] = 1
	if has_label:
		return {'user_nl': x_user_nl, 'stringmatch': x_stringmatch, 'label': y}
	else:
		return {'user_nl': x_user_nl, 'stringmatch': x_stringmatch}


def data2num_confirm_slot(batch_data, has_label=False):
	def is_zh(content):
		zhPattern = re.compile(u'^[\u4e00-\u9fa5]+$')
		match = zhPattern.search(content)
		return True if match else False
	
	batchsize = len(batch_data)
	x_user_nl = np.zeros((batchsize, max_length, embedding_dim))
	x_stringmatch = np.zeros((batchsize, max_length, 1))
	x_user_nl_len = np.zeros(batchsize, dtype='int32')
	y = np.zeros((batchsize, 2))
	
	for batch_id, data in enumerate(batch_data):
		# x_user_nl, x_stringmatch, x_user_nl_len
		words = filter(lambda x: is_zh(x), data['user_nl'])
		for word_id, word in enumerate(words):
			if word_id == max_length:
				break
			if word in vocab_dict:
				x_user_nl[batch_id, word_id, :] = embedding_table[word]
			else:
				x_user_nl[batch_id, word_id, :] = embedding_table['unk']
			if word in semantic_dict_request_slot[data['domain']][data['slot']]:
				x_stringmatch[batch_id, word_id, 0] = 1
		x_user_nl_len[batch_id] = len(data['user_nl'])
		
		# y
		if has_label:
			label_requestable_slot = {'MENTIONED': 0, 'NOT_MENTIONED': 1}
			y[batch_id, label_requestable_slot[data['label']]] = 1
	if has_label:
		return {'user_nl': x_user_nl, 'stringmatch': x_stringmatch, 'user_nl_len': x_user_nl_len, 'label': y}
	else:
		return {'user_nl': x_user_nl, 'stringmatch': x_stringmatch, 'user_nl_len': x_user_nl_len}


def data2num_act(batch_data, has_label=False):
	def is_zh(content):
		zhPattern = re.compile(u'^[\u4e00-\u9fa5]+$')
		match = zhPattern.search(content)
		return True if match else False
	
	batchsize = len(batch_data)
	x_user_nl = np.zeros((batchsize, max_length, embedding_dim))
	x_stringmatch = np.zeros((batchsize, max_length, 1))
	x_user_nl_len = np.zeros(batchsize, dtype='int32')
	y = np.zeros((batchsize, 2))
	
	for batch_id, data in enumerate(batch_data):
		# x_user_nl, x_stringmatch, x_user_nl_len
		words = filter(lambda x: is_zh(x), data['user_nl'])
		for word_id, word in enumerate(words):
			if word_id == max_length:
				break
			if word in vocab_dict:
				x_user_nl[batch_id, word_id, :] = embedding_table[word]
			else:
				x_user_nl[batch_id, word_id, :] = embedding_table['unk']
			if word in semantic_dict_act[data['user_act']]:
				x_stringmatch[batch_id, word_id, 0] = 1
		x_user_nl_len[batch_id] = len(data['user_nl'])
		
		# y
		if has_label:
			label_requestable_slot = {'MENTIONED': 0, 'NOT_MENTIONED': 1}
			y[batch_id, label_requestable_slot[data['label']]] = 1
	if has_label:
		return {'user_nl': x_user_nl, 'stringmatch': x_stringmatch, 'user_nl_len': x_user_nl_len, 'label': y}
	else:
		return {'user_nl': x_user_nl, 'stringmatch': x_stringmatch, 'user_nl_len': x_user_nl_len}


with tf.Graph().as_default():
	with tf.name_scope("DomainTracker"):
		with tf.variable_scope("d_movie"):
			domain_tracker_movie = DomainTracker()
		with tf.variable_scope("d_music"):
			domain_tracker_music = DomainTracker()
		with tf.variable_scope("d_weather"):
			domain_tracker_weather = DomainTracker()
		with tf.variable_scope("d_time"):
			domain_tracker_time = DomainTracker()
	with tf.name_scope('RequestSlotTracker'):
		with tf.name_scope('movie'):
			with tf.variable_scope('r_s_actor'):
				request_slot_tracker_actor = RequestSlotTracker()
			with tf.variable_scope('r_s_director'):
				request_slot_tracker_director = RequestSlotTracker()
			with tf.variable_scope('r_s_genre'):
				request_slot_tracker_genre = RequestSlotTracker()
			with tf.variable_scope('r_s_area'):
				request_slot_tracker_area = RequestSlotTracker()
			with tf.variable_scope('r_s_score'):
				request_slot_tracker_score = RequestSlotTracker()
			with tf.variable_scope('r_s_era'):
				request_slot_tracker_era = RequestSlotTracker()
			with tf.variable_scope('r_s_year'):
				request_slot_tracker_year = RequestSlotTracker()
			with tf.variable_scope('r_s_payment'):
				request_slot_tracker_payment = RequestSlotTracker()
			with tf.variable_scope('r_s_length'):
				request_slot_tracker_length = RequestSlotTracker()
			with tf.variable_scope('r_s_intro'):
				request_slot_tracker_intro = RequestSlotTracker()
		with tf.name_scope('music'):
			with tf.variable_scope('r_s_artist'):
				request_slot_tracker_artist = RequestSlotTracker()
			with tf.variable_scope('r_s_album'):
				request_slot_tracker_album = RequestSlotTracker()
			with tf.variable_scope('r_s_style'):
				request_slot_tracker_style = RequestSlotTracker()
			with tf.variable_scope('r_s_musicera'):
				request_slot_tracker_music_era = RequestSlotTracker()
			with tf.variable_scope('r_s_publish'):
				request_slot_tracker_publish = RequestSlotTracker()
			with tf.variable_scope('r_s_musiclength'):
				request_slot_tracker_music_length = RequestSlotTracker()
		with tf.name_scope('weather'):
			with tf.variable_scope('r_s_weather'):
				request_slot_tracker_weather = RequestSlotTracker()
		with tf.name_scope('time'):
			with tf.variable_scope('r_s_date'):
				request_slot_tracker_date = RequestSlotTracker()
			with tf.variable_scope('r_s_week'):
				request_slot_tracker_week = RequestSlotTracker()
			with tf.variable_scope('r_s_time'):
				request_slot_tracker_time = RequestSlotTracker()
	with tf.name_scope('InformSlotTracker'):
		with tf.name_scope('movie'):
			with tf.variable_scope('i_s_actor'):
				inform_slot_tracker_actor = InformSlotTracker()
			with tf.variable_scope('i_s_director'):
				inform_slot_tracker_director = InformSlotTracker()
			with tf.variable_scope('i_s_genre'):
				inform_slot_tracker_genre = InformSlotTracker()
			with tf.variable_scope('i_s_area'):
				inform_slot_tracker_area = InformSlotTracker()
			with tf.variable_scope('i_s_era'):
				inform_slot_tracker_era = InformSlotTracker()
			with tf.variable_scope('i_s_payment'):
				inform_slot_tracker_payment = InformSlotTracker()
		with tf.name_scope('music'):
			with tf.variable_scope('i_s_artist'):
				inform_slot_tracker_artist = InformSlotTracker()
			with tf.variable_scope('i_s_album'):
				inform_slot_tracker_album = InformSlotTracker()
			with tf.variable_scope('i_s_style'):
				inform_slot_tracker_style = InformSlotTracker()
			with tf.variable_scope('i_s_musicera'):
				inform_slot_tracker_music_era = InformSlotTracker()
	with tf.name_scope('ValueTracker'):
		with tf.name_scope('movie'):
			with tf.variable_scope('v_actor'):
				value_tracker_actor = ValueTracker()
			with tf.variable_scope('v_director'):
				value_tracker_director = ValueTracker()
			with tf.variable_scope('v_genre'):
				value_tracker_genre = ValueTracker()
			with tf.variable_scope('v_area'):
				value_tracker_area = ValueTracker()
			with tf.variable_scope('v_era'):
				value_tracker_era = ValueTracker()
			with tf.variable_scope('v_payment'):
				value_tracker_payment = ValueTracker()
		with tf.name_scope('music'):
			with tf.variable_scope('v_artist'):
				value_tracker_artist = ValueTracker()
			with tf.variable_scope('v_album'):
				value_tracker_album = ValueTracker()
			with tf.variable_scope('v_style'):
				value_tracker_style = ValueTracker()
			with tf.variable_scope('v_musicera'):
				value_tracker_music_era = ValueTracker()
	with tf.name_scope('ConfirmSlotTracker'):
		with tf.name_scope('movie'):
			with tf.variable_scope('c_s_actor'):
				confirm_slot_tracker_actor = ConfirmSlotTracker()
			with tf.variable_scope('c_s_director'):
				confirm_slot_tracker_director = ConfirmSlotTracker()
			with tf.variable_scope('c_s_genre'):
				confirm_slot_tracker_genre = ConfirmSlotTracker()
			with tf.variable_scope('c_s_area'):
				confirm_slot_tracker_area = ConfirmSlotTracker()
			with tf.variable_scope('c_s_era'):
				confirm_slot_tracker_era = ConfirmSlotTracker()
			with tf.variable_scope('c_s_payment'):
				confirm_slot_tracker_payment = ConfirmSlotTracker()
		with tf.name_scope('music'):
			with tf.variable_scope('c_s_artist'):
				confirm_slot_tracker_artist = ConfirmSlotTracker()
			with tf.variable_scope('c_s_album'):
				confirm_slot_tracker_album = ConfirmSlotTracker()
			with tf.variable_scope('c_s_style'):
				confirm_slot_tracker_style = ConfirmSlotTracker()
			with tf.variable_scope('c_s_musicera'):
				confirm_slot_tracker_music_era = ConfirmSlotTracker()
	with tf.name_scope('ActTracker'):
		with tf.variable_scope('first'):
			act_tracker_first = ActTracker()
		with tf.variable_scope('second'):
			act_tracker_second = ActTracker()
		with tf.variable_scope('third'):
			act_tracker_third = ActTracker()
		with tf.variable_scope('last'):
			act_tracker_last = ActTracker()
		with tf.variable_scope('other'):
			act_tracker_other = ActTracker()
		with tf.variable_scope('affirm'):
			act_tracker_affirm = ActTracker()
		with tf.variable_scope('deny'):
			act_tracker_deny = ActTracker()
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		saver = tf.train.Saver()
		# saver.restore(sess, "../../model/tmp_final/model.ckpt")
		
		domain_tracker = {
			"电影": domain_tracker_movie,
			"音乐": domain_tracker_music,
			"天气": domain_tracker_weather,
			"时间": domain_tracker_time
		}
		request_slot_tracker = {
			"电影": {
				'主演': request_slot_tracker_actor,
				'导演': request_slot_tracker_director,
				'类型': request_slot_tracker_genre,
				'地区': request_slot_tracker_area,
				'评分': request_slot_tracker_score,
				'年代': request_slot_tracker_era,
				'上映日期': request_slot_tracker_year,
				'资费': request_slot_tracker_payment,
				'片长': request_slot_tracker_length,
				'简介': request_slot_tracker_intro
			},
			"音乐": {
				'歌手': request_slot_tracker_artist,
				'专辑': request_slot_tracker_album,
				'曲风': request_slot_tracker_style,
				'年代': request_slot_tracker_music_era,
				'发行日期': request_slot_tracker_publish,
				'时长': request_slot_tracker_music_length
			},
			"天气": {
				'天气': request_slot_tracker_weather
			},
			"时间": {
				'日期': request_slot_tracker_date,
				'星期': request_slot_tracker_week,
				'时刻': request_slot_tracker_time
			}
		}
		inform_slot_tracker = {
			'电影': {
				'主演': inform_slot_tracker_actor,
				'导演': inform_slot_tracker_director,
				'类型': inform_slot_tracker_genre,
				'地区': inform_slot_tracker_area,
				'年代': inform_slot_tracker_era,
				'资费': inform_slot_tracker_payment
			},
			'音乐': {
				'歌手': inform_slot_tracker_artist,
				'专辑': inform_slot_tracker_album,
				'曲风': inform_slot_tracker_style,
				'年代': inform_slot_tracker_music_era
			}
		}
		value_tracker = {
			'电影': {
				'主演': value_tracker_actor,
				'导演': value_tracker_director,
				'类型': value_tracker_genre,
				'地区': value_tracker_area,
				'年代': value_tracker_era,
				'资费': value_tracker_payment
			},
			'音乐': {
				'歌手': value_tracker_artist,
				'专辑': value_tracker_album,
				'曲风': value_tracker_style,
				'年代': value_tracker_music_era
			}
		}
		confirm_slot_tracker = {
			'电影': {
				'主演': confirm_slot_tracker_actor,
				'导演': confirm_slot_tracker_director,
				'类型': confirm_slot_tracker_genre,
				'地区': confirm_slot_tracker_area,
				'年代': confirm_slot_tracker_era,
				'资费': confirm_slot_tracker_payment
			},
			'音乐': {
				'歌手': confirm_slot_tracker_artist,
				'专辑': confirm_slot_tracker_album,
				'曲风': confirm_slot_tracker_style,
				'年代': confirm_slot_tracker_music_era
			}
		}
		act_tracker = {
			'first': act_tracker_first,
			'second': act_tracker_second,
			'third': act_tracker_third,
			'last': act_tracker_last,
			'other': act_tracker_other,
			'affirm': act_tracker_affirm,
			'deny': act_tracker_deny
		}
		# we use early stopping for all of the trackers
		# training & evaluate
		pass  # train domain tracker
		print('\n\n------train domain tracker------')
		domain = '电影'
		data = [
			{'user_nl': ['想看', '喜剧片', '或', '恐怖片', '。'],
			 'sys_act': {"hello": {}},
			 'domain': '电影'},
			{'user_nl': ['想看', '喜剧片', '或', '恐怖片', '。'],
			 'sys_act': {"request": {"电影": "类型"}},
			 'domain': '电影'},
			{'user_nl': ['想看', '喜剧片', '或', '恐怖片', '。'],
			 'sys_act': {"inform_no_match": {"电影": {"片名": []}}},
			 'domain': '电影'},
			{'user_nl': ['想看', '喜剧片', '或', '恐怖片', '。'],
			 'sys_act': {"inform_one_match": {"电影": {"片名": ['战狼']}}},
			 'domain': '电影'},
			{'user_nl': ['想看', '喜剧片', '或', '恐怖片', '。'],
			 'sys_act': {"inform_some_match": {"电影": {"片名": ['战狼', '战狼二']}}},
			 'domain': '电影'},
			{'user_nl': ['想看', '喜剧片', '或', '恐怖片', '。'],
			 'sys_act': {"inform": {"电影": ["演员"]}},
			 'domain': '电影'},
			{'user_nl': ['想看', '喜剧片', '或', '恐怖片', '。'],
			 'sys_act': {"confirm": {"电影": {"导演": "提莫贝克曼贝托夫"}}},
			 'domain': '电影'}
		]
		num_valid_domain = data2num_domain(data, has_label=False)
		print(num_valid_domain['sys_act'])
		valid_pro = sess.run(domain_tracker[domain]._probability,
		                     feed_dict={
			                     domain_tracker[domain]._x_user_nl: num_valid_domain['user_nl'],
			                     domain_tracker[domain]._x_stringmatch: num_valid_domain['stringmatch'],
			                     domain_tracker[domain]._x_sys_act: num_valid_domain['sys_act'],
			                     domain_tracker[domain]._is_training: False
		                     }
		                     )
		del num_valid_domain
		print("---例子---\n", valid_pro)

		data_valid_domain = datasets.valid._domain_data
		for domain in domain_set:
			patience, patience_count = 5, 0
			previous_valid_accuracy = 0
			for batch_num in range(10000):
				# input the minibatch data
				next_batch_data = datasets.train.next_batch_domain(batchsize_domain, domain)
				num = data2num_domain(next_batch_data, has_label=True)  # x_usr, x_usr_len, x_slot
				sess.run(domain_tracker[domain].train_op(),
				         feed_dict={
					         domain_tracker[domain]._x_user_nl: num['user_nl'],
					         domain_tracker[domain]._x_stringmatch: num['stringmatch'],
					         domain_tracker[domain]._x_sys_act: num['sys_act'],
					         domain_tracker[domain]._is_training: True,
					         domain_tracker[domain]._y: num['label']
				         }
				         )
				if (batch_num + 1) % 100 == 0:
					data = data_valid_domain[domain]["MENTIONED"] + data_valid_domain[domain]["NOT_MENTIONED"]
					num_valid_domain = data2num_domain(data, has_label=True)
					len1 = len(data_valid_domain[domain]["MENTIONED"])
					# train value-specific tracker
					valid_pred = sess.run(domain_tracker[domain]._correct_pred,
					                      feed_dict={
						                      domain_tracker[domain]._x_user_nl: num_valid_domain["user_nl"],
						                      domain_tracker[domain]._x_stringmatch: num_valid_domain["stringmatch"],
						                      domain_tracker[domain]._x_sys_act: num_valid_domain["sys_act"],
						                      domain_tracker[domain]._is_training: False,
						                      domain_tracker[domain]._y: num_valid_domain["label"]
					                      }
					                      )
					del num_valid_domain
					valid_accuracy = np.mean(valid_pred)
					print(domain, "batchnum", (batch_num + 1), "domain accuracy: + ", np.mean(valid_pred[:len1]), ' - ',
					      np.mean(valid_pred[len1:]), ' total ', valid_accuracy)

					# whether stop
					if valid_accuracy > previous_valid_accuracy:
						previous_valid_accuracy = valid_accuracy
						patience_count = 0
						# save model
						if not os.path.exists('../../model/tmp_final/'):
							os.mkdir('../../model/tmp_final/')
						save_path = saver.save(sess, "../../model/tmp_final/model.ckpt")
					else:
						patience_count += 1
						if patience_count == patience:
							print(domain, "best domain accuracy: ", previous_valid_accuracy)
							break
			saver.restore(sess, "../../model/tmp_final/model.ckpt")
			data = data_valid_domain[domain]["MENTIONED"] + data_valid_domain[domain]["NOT_MENTIONED"]
			num_valid_domain = data2num_domain(data, has_label=True)
			valid_pred, valid_pro = sess.run(
				(domain_tracker[domain]._correct_pred, domain_tracker[domain]._probability),
				feed_dict={
					domain_tracker[domain]._x_user_nl: num_valid_domain["user_nl"],
					domain_tracker[domain]._x_stringmatch: num_valid_domain["stringmatch"],
					domain_tracker[domain]._x_sys_act: num_valid_domain["sys_act"],
					domain_tracker[domain]._is_training: False,
					domain_tracker[domain]._y: num_valid_domain['label']
				}
			)
			valid_accuracy = np.mean(valid_pred)
			for index, correct_flag in enumerate(valid_pred):
				if correct_flag == 0:
					print(num_valid_domain['label'][index], "MENTIONED %.4f  NOT_MENTIONED %.4f"
					      % (valid_pro[index, 0], valid_pro[index, 1]))
					print(data[index])
			del num_valid_domain
		pass  # train requestable slot tracker
		print('\n\n------train requestable slot tracker------')
		data_valid_request_slot = datasets.valid._request_slot_data
		domain = '电影'
		slot = '主演'
		data = [
			{'user_nl': ['有', '哪些', '演员', '？', '。'],
			 'domain': '电影', 'slot': '主演'},
			{'user_nl': ['有', '哪些', '演员', '？', '。'],
			 'domain': '电影', 'slot': '导演'},
			{'user_nl': ['有', '哪些', '演员', '？', '。'],
			 'domain': '电影', 'slot': '评分'}
		]
		num_valid_request_slot = data2num_request_slot(data, has_label=False)
		valid_pro = sess.run(request_slot_tracker[domain][slot]._probability,
		                     feed_dict={
			                     request_slot_tracker[domain][slot]._x_user_nl: num_valid_request_slot['user_nl'],
			                     request_slot_tracker[domain][slot]._x_stringmatch: num_valid_request_slot[
				                     'stringmatch'],
			                     request_slot_tracker[domain][slot]._is_training: False
		                     }
		                     )
		del num_valid_request_slot
		print("---例子---\n", valid_pro)

		for domain in domain_set:
			for slot in request_slots[domain]:
				patience, patience_count = 3, 0
				previous_valid_accuracy = 0
				for batch_num in range(10000):
					# input the minibatch data
					next_batch_data = datasets.train.next_batch_request_slot(batchsize_request_slot, domain, slot)
					num = data2num_request_slot(next_batch_data, has_label=True)  # x_usr, x_usr_len, x_slot
					sess.run(request_slot_tracker[domain][slot].train_op(),
					         feed_dict={
						         request_slot_tracker[domain][slot]._x_user_nl: num['user_nl'],
						         request_slot_tracker[domain][slot]._x_stringmatch: num['stringmatch'],
						         request_slot_tracker[domain][slot]._is_training: True,
						         request_slot_tracker[domain][slot]._y: num['label']
					         }
					         )
					if (batch_num + 1) % 100 == 0:
						data = data_valid_request_slot[domain][slot]["MENTIONED"] + \
						       data_valid_request_slot[domain][slot]["NOT_MENTIONED"]
						num_valid_request_slot = data2num_request_slot(data, has_label=True)
						len1 = len(data_valid_request_slot[domain][slot]["MENTIONED"])
						# train value-specific tracker
						valid_pred = sess.run(request_slot_tracker[domain][slot]._correct_pred,
						                      feed_dict={
							                      request_slot_tracker[domain][slot]._x_user_nl: num_valid_request_slot[
								                      "user_nl"],
							                      request_slot_tracker[domain][slot]._x_stringmatch:
								                      num_valid_request_slot["stringmatch"],
							                      request_slot_tracker[domain][slot]._is_training: False,
							                      request_slot_tracker[domain][slot]._y: num_valid_request_slot["label"]
						                      }
						                      )
						del num_valid_request_slot
						valid_accuracy = np.mean(valid_pred)
						print(domain, slot, "batchnum", (batch_num + 1), "requestable_slot accuracy: + ",
						      np.mean(valid_pred[:len1]), ' - ',
						      np.mean(valid_pred[len1:]), ' total ', valid_accuracy)

						# whether stop
						if valid_accuracy > previous_valid_accuracy:
							previous_valid_accuracy = valid_accuracy
							patience_count = 0
							# save model
							if not os.path.exists('../../model/tmp_final/'):
								os.mkdir('../../model/tmp_final/')
							save_path = saver.save(sess, "../../model/tmp_final/model.ckpt")
						else:
							patience_count += 1
							if patience_count == patience:
								print(domain, slot, "best request_slot accuracy: ", previous_valid_accuracy)
								break
				saver.restore(sess, "../../model/tmp_final/model.ckpt")
				data = data_valid_request_slot[domain][slot]["MENTIONED"] + data_valid_request_slot[domain][slot][
					"NOT_MENTIONED"]
				num_valid_request_slot = data2num_request_slot(data, has_label=True)
				valid_pred, valid_pro = sess.run(
					(request_slot_tracker[domain][slot]._correct_pred, request_slot_tracker[domain][slot]._probability),
					feed_dict={
						request_slot_tracker[domain][slot]._x_user_nl: num_valid_request_slot["user_nl"],
						request_slot_tracker[domain][slot]._x_stringmatch: num_valid_request_slot["stringmatch"],
						request_slot_tracker[domain][slot]._is_training: False,
						request_slot_tracker[domain][slot]._y: num_valid_request_slot['label']
					}
				)
				valid_accuracy = np.mean(valid_pred)
				for index, correct_flag in enumerate(valid_pred):
					if correct_flag == 0:
						print(num_valid_request_slot['label'][index], "MENTIONED %.4f  NOT_MENTIONED %.4f"
						      % (valid_pro[index, 0], valid_pro[index, 1]))
						print(data[index])
				del num_valid_request_slot
		pass  # train informable slot tracker
		print('\n\n------train informable slot tracker------')
		domain = '电影'
		slot = '主演'
		data = [
			{'user_nl': ['有', '哪些', '演员', '？', '。'], 'sys_act': {"request": {"电影": "主演"}},
			 'domain': '电影', 'slot': '主演'},
			{'user_nl': ['有', '哪些', '演员', '？', '。'], 'sys_act': {"confirm": {"电影": {"导演": '陈凯歌'}}},
			 'domain': '电影', 'slot': '导演'},
			{'user_nl': ['有', '哪些', '演员', '？', '。'], 'sys_act': {"inform": {"电影": ["类型"]}},
			 'domain': '电影', 'slot': '类型'},
			{'user_nl': ['有', '哪些', '演员', '？', '。'], 'sys_act': {"request": {"电影": "主演"}},
			 'domain': '电影', 'slot': '地区'}
		]
		num_valid_inform_slot = data2num_inform_slot(data, has_label=False)
		print(num_valid_inform_slot['sys_act'])
		valid_pro = sess.run(inform_slot_tracker[domain][slot]._probability,
		                     feed_dict={
			                     inform_slot_tracker[domain][slot]._x_user_nl: num_valid_inform_slot['user_nl'],
			                     inform_slot_tracker[domain][slot]._x_stringmatch: num_valid_inform_slot['stringmatch'],
			                     inform_slot_tracker[domain][slot]._x_sys_act: num_valid_inform_slot['sys_act'],
			                     inform_slot_tracker[domain][slot]._is_training: False
		                     }
		                     )
		del num_valid_inform_slot
		print("---例子---\n", valid_pro)

		data_valid_inform_slot = datasets.valid._inform_slot_data
		for domain in ['电影', '音乐']:
			for slot in inform_slots[domain]:
				patience, patience_count = 5, 0
				previous_valid_accuracy = 0
				for batch_num in range(10000):
					# input the minibatch data
					next_batch_data = datasets.train.next_batch_inform_slot(batchsize_inform_slot, domain, slot)
					num = data2num_inform_slot(next_batch_data, has_label=True)  # x_usr, x_usr_len, x_slot
					sess.run(inform_slot_tracker[domain][slot].train_op(),
					         feed_dict={
						         inform_slot_tracker[domain][slot]._x_user_nl: num['user_nl'],
						         inform_slot_tracker[domain][slot]._x_stringmatch: num['stringmatch'],
						         inform_slot_tracker[domain][slot]._x_sys_act: num['sys_act'],
						         inform_slot_tracker[domain][slot]._is_training: True,
						         inform_slot_tracker[domain][slot]._y: num['label']
					         }
					         )
					if (batch_num + 1) % 100 == 0:
						data = []
						for label in inform_slot_label:
							data += data_valid_inform_slot[domain][slot][label]
						num_valid_inform_slot = data2num_inform_slot(data, has_label=True)
						len1 = len(data_valid_inform_slot[domain][slot]["MENTIONED"])
						len2 = len1 + len(data_valid_inform_slot[domain][slot]["NOT_MENTIONED"])
						feed_dict = {
							inform_slot_tracker[domain][slot]._x_user_nl: num_valid_inform_slot["user_nl"],
							inform_slot_tracker[domain][slot]._x_stringmatch: num_valid_inform_slot["stringmatch"],
							inform_slot_tracker[domain][slot]._x_sys_act: num_valid_inform_slot['sys_act'],
							inform_slot_tracker[domain][slot]._is_training: False,
							inform_slot_tracker[domain][slot]._y: num_valid_inform_slot["label"]
						}
						valid_pred = sess.run(inform_slot_tracker[domain][slot]._correct_pred, feed_dict=feed_dict)
						del num_valid_inform_slot
						valid_accuracy = np.mean(valid_pred)
						print(domain, slot, 'batchnum', (batch_num + 1), 'informable_slot accuracy:',
						      ' + ', np.mean(valid_pred[:len1]),
						      ' - ', np.mean(valid_pred[len1: len2]),
						      ' O ', np.mean(valid_pred[len2:]),
						      ' total ', valid_accuracy)

						# whether stop
						if valid_accuracy > previous_valid_accuracy:
							previous_valid_accuracy = valid_accuracy
							patience_count = 0
							# save model
							if not os.path.exists('../../model/tmp_final/'):
								os.mkdir('../../model/tmp_final/')
							save_path = saver.save(sess, "../../model/tmp_final/model.ckpt")
						else:
							patience_count += 1
							if patience_count == patience:
								print(domain, slot, "best inform_slot accuracy: ", previous_valid_accuracy)
								break
				saver.restore(sess, "../../model/tmp_final/model.ckpt")
				data = []
				for label in inform_slot_label:
					data += data_valid_inform_slot[domain][slot][label]
				num_valid_inform_slot = data2num_inform_slot(data, has_label=True)
				valid_pred, valid_pro = sess.run(
					(inform_slot_tracker[domain][slot]._correct_pred, inform_slot_tracker[domain][slot]._probability),
					feed_dict={
						inform_slot_tracker[domain][slot]._x_user_nl: num_valid_inform_slot["user_nl"],
						inform_slot_tracker[domain][slot]._x_stringmatch: num_valid_inform_slot["stringmatch"],
						inform_slot_tracker[domain][slot]._x_sys_act: num_valid_inform_slot["sys_act"],
						inform_slot_tracker[domain][slot]._is_training: False,
						inform_slot_tracker[domain][slot]._y: num_valid_inform_slot['label']
					}
				)
				valid_accuracy = np.mean(valid_pred)
				for index, correct_flag in enumerate(valid_pred):
					if correct_flag == 0:
						print(num_valid_inform_slot['label'][index],
						      "MENTIONED %.4f  NOT_MENTIONED %.4f  DONT_CARE %.4f"
						      % (valid_pro[index, 0], valid_pro[index, 1], valid_pro[index, 2]))
						print(data[index])
				del num_valid_inform_slot
		pass  # train value tracker
		print('\n\n------train value tracker------')
		domain = '电影'
		slot = '主演'
		data = [
			{'user_nl': ['我', '喜欢', '吴京', '。'], 'domain': '电影', 'slot': '主演', 'value': '吴京'},
			{'user_nl': ['我', '讨厌', '吴京', '。'], 'domain': '电影', 'slot': '主演', 'value': '吴京'},
			{'user_nl': ['我', '喜欢', '成龙', '。'], 'domain': '电影', 'slot': '主演', 'value': '吴京'},
			{'user_nl': ['我', '喜欢', '吴京', '。'], 'domain': '电影', 'slot': '导演', 'value': '吴京'},
		]
		num_valid_value = data2num_value(data, has_label=False)
		print(num_valid_value['stringmatch'])
		valid_pro = sess.run(value_tracker[domain][slot]._probability,
		                     feed_dict={
			                     value_tracker[domain][slot]._x_user_nl: num_valid_value['user_nl'],
			                     value_tracker[domain][slot]._x_stringmatch: num_valid_value['stringmatch'],
			                     value_tracker[domain][slot]._is_training: False
		                     }
		                     )
		del num_valid_value
		print("---例子---\n", valid_pro)

		data_valid_value = datasets.valid._value_data
		for domain in ['电影', '音乐']:
			for slot in inform_slots[domain]:
				patience, patience_count = 5, 0
				previous_valid_accuracy = 0
				for batch_num in range(10000):
					# input the minibatch data
					next_batch_data = datasets.train.next_batch_value(batchsize_value, domain, slot)
					num = data2num_value(next_batch_data, has_label=True)  # x_usr, x_usr_len, x_slot
					sess.run(value_tracker[domain][slot].train_op(),
					         feed_dict={
						         value_tracker[domain][slot]._x_user_nl: num['user_nl'],
						         value_tracker[domain][slot]._x_stringmatch: num['stringmatch'],
						         value_tracker[domain][slot]._is_training: True,
						         value_tracker[domain][slot]._y: num['label']
					         }
					         )
					if (batch_num + 1) % 100 == 0:
						data = []
						for label in value_label:
							data += data_valid_value[domain][slot][label]
						num_valid_value = data2num_value(data, has_label=True)
						len1 = len(data_valid_value[domain][slot]["LIKE"])
						feed_dict = {
							value_tracker[domain][slot]._x_user_nl: num_valid_value["user_nl"],
							value_tracker[domain][slot]._x_stringmatch: num_valid_value["stringmatch"],
							value_tracker[domain][slot]._is_training: False,
							value_tracker[domain][slot]._y: num_valid_value["label"]
						}
						valid_pred = sess.run(value_tracker[domain][slot]._correct_pred, feed_dict=feed_dict)
						del num_valid_value
						valid_accuracy = np.mean(valid_pred)
						print(domain, slot, 'batchnum', (batch_num + 1), 'informable_slot accuracy:',
						      ' + ', np.mean(valid_pred[:len1]),
						      ' - ', np.mean(valid_pred[len1:]),
						      ' total ', valid_accuracy)

						# whether stop
						if valid_accuracy > previous_valid_accuracy:
							previous_valid_accuracy = valid_accuracy
							patience_count = 0
							# save model
							if not os.path.exists('../../model/tmp_final/'):
								os.mkdir('../../model/tmp_final/')
							save_path = saver.save(sess, "../../model/tmp_final/model.ckpt")
						else:
							patience_count += 1
							if patience_count == patience:
								print(domain, slot, "best value accuracy: ", previous_valid_accuracy)
								break
				saver.restore(sess, "../../model/tmp_final/model.ckpt")
				data = []
				for label in value_label:
					data += data_valid_value[domain][slot][label]
				num_valid_value = data2num_value(data, has_label=True)
				valid_pred, valid_pro = sess.run(
					(value_tracker[domain][slot]._correct_pred, value_tracker[domain][slot]._probability),
					feed_dict={
						value_tracker[domain][slot]._x_user_nl: num_valid_value["user_nl"],
						value_tracker[domain][slot]._x_stringmatch: num_valid_value["stringmatch"],
						value_tracker[domain][slot]._is_training: False,
						value_tracker[domain][slot]._y: num_valid_value['label']
					}
				)
				valid_accuracy = np.mean(valid_pred)
				for index, correct_flag in enumerate(valid_pred):
					if correct_flag == 0:
						print(num_valid_value['label'][index], "LIKE %.4f  DISLIKE %.4f"
						      % (valid_pro[index, 0], valid_pro[index, 1]))
						print(data[index])
				del num_valid_value
		pass  # train confirm slot tracker
		# print('\n\n------train confirm slot tracker------')
		# data_valid_confirm_slot = datasets.valid._confirm_slot_data
		# domain = '电影'
		# slot = '主演'
		# data = [
		# 	{'user_nl': ['演员', '有', '葛优', '么', '？'],
		# 	 'domain': '电影', 'slot': '主演'},
		# 	{'user_nl': ['演员', '有', '葛优', '么', '？'],
		# 	 'domain': '电影', 'slot': '导演'},
		# 	{'user_nl': ['演员', '有', '葛优', '么', '？'],
		# 	 'domain': '电影', 'slot': '评分'}
		# ]
		# num_valid_confirm_slot = data2num_confirm_slot(data, has_label=False)
		# valid_pro = sess.run(confirm_slot_tracker[domain][slot]._probability,
		#                      feed_dict={
		# 	                     confirm_slot_tracker[domain][slot]._x_user_nl: num_valid_confirm_slot['user_nl'],
		# 	                     confirm_slot_tracker[domain][slot]._x_stringmatch: num_valid_confirm_slot[
		# 		                     'stringmatch'],
		# 	                     confirm_slot_tracker[domain][slot]._is_training: False
		#                      }
		#                      )
		# del num_valid_confirm_slot
		# print("---例子---\n", valid_pro)
		#
		# for domain in ['电影', '音乐']:
		# 	for slot in confirm_slots[domain]:
		# 		patience, patience_count = 5, 0
		# 		previous_valid_accuracy = 0
		# 		for batch_num in range(10000):
		# 			# input the minibatch data
		# 			next_batch_data = datasets.train.next_batch_confirm_slot(batchsize_confirm_slot, domain, slot)
		# 			num = data2num_confirm_slot(next_batch_data, has_label=True)  # x_usr, x_usr_len, x_slot
		# 			sess.run(confirm_slot_tracker[domain][slot].train_op(),
		# 			         feed_dict={
		# 				         confirm_slot_tracker[domain][slot]._x_user_nl: num['user_nl'],
		# 				         confirm_slot_tracker[domain][slot]._x_stringmatch: num['stringmatch'],
		# 				         confirm_slot_tracker[domain][slot]._is_training: True,
		# 				         confirm_slot_tracker[domain][slot]._y: num['label']
		# 			         }
		# 			         )
		# 			if (batch_num + 1) % 100 == 0:
		# 				data = data_valid_confirm_slot[domain][slot]["MENTIONED"] + \
		# 				       data_valid_confirm_slot[domain][slot]["NOT_MENTIONED"]
		# 				num_valid_confirm_slot = data2num_confirm_slot(data, has_label=True)
		# 				len1 = len(data_valid_confirm_slot[domain][slot]["MENTIONED"])
		# 				# train value-specific tracker
		# 				valid_pred = sess.run(confirm_slot_tracker[domain][slot]._correct_pred,
		# 				                      feed_dict={
		# 					                      confirm_slot_tracker[domain][slot]._x_user_nl: num_valid_confirm_slot[
		# 						                      "user_nl"],
		# 					                      confirm_slot_tracker[domain][slot]._x_stringmatch:
		# 						                      num_valid_confirm_slot["stringmatch"],
		# 					                      confirm_slot_tracker[domain][slot]._is_training: False,
		# 					                      confirm_slot_tracker[domain][slot]._y: num_valid_confirm_slot["label"]
		# 				                      }
		# 				                      )
		# 				del num_valid_confirm_slot
		# 				valid_accuracy = np.mean(valid_pred)
		# 				print(domain, slot, "batchnum", (batch_num + 1), "confirmable_slot accuracy: + ",
		# 				      np.mean(valid_pred[:len1]), ' - ',
		# 				      np.mean(valid_pred[len1:]), ' total ', valid_accuracy)
		#
		# 				# whether stop
		# 				if valid_accuracy > previous_valid_accuracy:
		# 					previous_valid_accuracy = valid_accuracy
		# 					patience_count = 0
		# 					# save model
		# 					if not os.path.exists('../../model/tmp_final/'):
		# 						os.mkdir('../../model/tmp_final/')
		# 					save_path = saver.save(sess, "../../model/tmp_final/model.ckpt")
		# 				else:
		# 					patience_count += 1
		# 					if patience_count == patience:
		# 						print(domain, slot, "best confirm_slot accuracy: ", previous_valid_accuracy)
		# 						break
		# 		saver.restore(sess, "../../model/tmp_final/model.ckpt")
		# 		data = data_valid_confirm_slot[domain][slot]["MENTIONED"] + data_valid_confirm_slot[domain][slot][
		# 			"NOT_MENTIONED"]
		# 		num_valid_confirm_slot = data2num_confirm_slot(data, has_label=True)
		# 		valid_pred, valid_pro = sess.run(
		# 			(confirm_slot_tracker[domain][slot]._correct_pred, confirm_slot_tracker[domain][slot]._probability),
		# 			feed_dict={
		# 				confirm_slot_tracker[domain][slot]._x_user_nl: num_valid_confirm_slot["user_nl"],
		# 				confirm_slot_tracker[domain][slot]._x_stringmatch: num_valid_confirm_slot["stringmatch"],
		# 				confirm_slot_tracker[domain][slot]._is_training: False,
		# 				confirm_slot_tracker[domain][slot]._y: num_valid_confirm_slot['label']
		# 			}
		# 		)
		# 		valid_accuracy = np.mean(valid_pred)
		# 		for index, correct_flag in enumerate(valid_pred):
		# 			if correct_flag == 0:
		# 				print(num_valid_confirm_slot['label'][index], "MENTIONED %.4f  NOT_MENTIONED %.4f"
		# 				      % (valid_pro[index, 0], valid_pro[index, 1]))
		# 				print(data[index])
		# 		del num_valid_confirm_slot
		pass  # train act tracker
		print('\n\n------train act tracker------')
		act = 'first'
		data = [
			{'user_nl': ['我', '想看', '第一', '个', '。'], 'user_act': 'first'},
			{'user_nl': ['我', '想看', '第一', '个', '。'], 'user_act': 'second'},
			{'user_nl': ['我', '想看', '第一', '个', '。'], 'user_act': 'third'},
			{'user_nl': ['我', '想看', '第二', '个', '。'], 'user_act': 'first'}
		]
		num_valid_act = data2num_act(data, has_label=False)
		valid_pro = sess.run(act_tracker[act]._probability,
		                     feed_dict={
			                     act_tracker[act]._x_user_nl: num_valid_act['user_nl'],
			                     act_tracker[act]._x_stringmatch: num_valid_act['stringmatch'],
			                     act_tracker[act]._is_training: False
		                     }
		                     )
		del num_valid_act
		print("---例子---\n", valid_pro)
		
		data_valid_act = datasets.valid._act_data
		for act in user_acts:
			patience, patience_count = 5, 0
			previous_valid_accuracy = 0
			for batch_num in range(10000):
				# input the minibatch data
				next_batch_data = datasets.train.next_batch_act(batchsize_act, act)
				num = data2num_act(next_batch_data, has_label=True)  # x_usr, x_usr_len, x_slot
				sess.run(act_tracker[act].train_op(),
				         feed_dict={
					         act_tracker[act]._x_user_nl: num['user_nl'],
					         act_tracker[act]._x_stringmatch: num['stringmatch'],
					         act_tracker[act]._is_training: True,
					         act_tracker[act]._y: num['label']
				         }
				         )
				if (batch_num + 1) % 100 == 0:
					data = data_valid_act[act]["MENTIONED"] + data_valid_act[act]["NOT_MENTIONED"]
					num_valid_act = data2num_act(data, has_label=True)
					len1 = len(data_valid_act[act]["MENTIONED"])
					# train value-specific tracker
					valid_pred = sess.run(act_tracker[act]._correct_pred,
					                      feed_dict={
						                      act_tracker[act]._x_user_nl: num_valid_act["user_nl"],
						                      act_tracker[act]._x_stringmatch: num_valid_act["stringmatch"],
						                      act_tracker[act]._is_training: False,
						                      act_tracker[act]._y: num_valid_act["label"]
					                      }
					                      )
					del num_valid_act
					valid_accuracy = np.mean(valid_pred)
					print(act, "batchnum", (batch_num + 1), "act accuracy: + ", np.mean(valid_pred[:len1]), ' - ',
					      np.mean(valid_pred[len1:]), ' total ', valid_accuracy)
					
					# whether stop
					if valid_accuracy > previous_valid_accuracy:
						previous_valid_accuracy = valid_accuracy
						patience_count = 0
						# save model
						if not os.path.exists('../../model/tmp_final/'):
							os.mkdir('../../model/tmp_final/')
						save_path = saver.save(sess, "../../model/tmp_final/model.ckpt")
					else:
						patience_count += 1
						if patience_count == patience:
							print(act, "best act accuracy: ", previous_valid_accuracy)
							break
			saver.restore(sess, "../../model/tmp_final/model.ckpt")
			data = data_valid_act[act]["MENTIONED"] + data_valid_act[act]["NOT_MENTIONED"]
			num_valid_act = data2num_act(data, has_label=True)
			valid_pred, valid_pro = sess.run(
				(act_tracker[act]._correct_pred, act_tracker[act]._probability),
				feed_dict={
					act_tracker[act]._x_user_nl: num_valid_act["user_nl"],
					act_tracker[act]._x_stringmatch: num_valid_act["stringmatch"],
					act_tracker[act]._is_training: False,
					act_tracker[act]._y: num_valid_act['label']
				}
			)
			valid_accuracy = np.mean(valid_pred)
			for index, correct_flag in enumerate(valid_pred):
				if correct_flag == 0:
					print(num_valid_act['label'][index], "MENTIONED %.4f  NOT_MENTIONED %.4f"
					      % (valid_pro[index, 0], valid_pro[index, 1]))
					print(data[index])
			del num_valid_act