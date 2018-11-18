#!/usr/bin/env python3
# coding=utf-8


import tensorflow as tf
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
from policy import rule_based_policy
from NLG import rule_based_NLG
from six.moves import xrange

# jieba
def jieba_init():
	def del_redundant_words(filename):
		with open(filename, 'r', encoding="utf-8") as f:
			words = json.load(f)
			for word in words:
				jieba.del_word(word)
	jieba.load_userdict("../../data/word2vec/OTLG_words.txt")
	jieba.load_userdict("../../data/word2vec/words_to_add.txt")
	del_redundant_words("../../data/word2vec/words_to_del.json")
jieba_init()

# vocabulary, word embedding
print('loading files ...')
with open('../../data/word2vec/OTLG_words.json', 'r', encoding='utf-8') as f:
	OTLG_words = json.load(f)
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
# synonym dict
with open('../../data/word2vec/synonym_dict.json', 'r', encoding='utf-8') as f:
	synonym_dict = json.load(f)
# ontology
with open('../../definition/OTLG.json', 'r', encoding='utf-8') as f:
	OTLG = json.load(f)
# domains, slots
domain_set = ["电影", "音乐", "天气", "时间"]
request_slots = {domain: OTLG[domain]['requestable'] for domain in domain_set}
inform_slots = {domain: set(OTLG[domain]['informable'].keys()) - set(OTLG[domain]['major_key']) for domain in ['电影', '音乐']}
confirm_slots = {domain: set(OTLG[domain]['informable'].keys()) - set(OTLG[domain]['major_key']) for domain in ['电影', '音乐']}
user_acts = ['first', 'second', 'third', 'last', 'other', 'affirm', 'deny']
# KB
def load_knowledge_base():
	kb = {'电影': {}, '音乐': {}}
	with open('../../data/knowledge_base/kb_movie_name.json', encoding='utf-8') as f:
		kb_movie_name = json.load(f)
	with open('../../data/knowledge_base/kb_movie_actor.json', encoding='utf-8') as f:
		kb_movie_actor = json.load(f)
	with open('../../data/knowledge_base/kb_movie_director.json', encoding='utf-8') as f:
		kb_movie_director = json.load(f)
	with open('../../data/knowledge_base/kb_movie_genre.json', encoding='utf-8') as f:
		kb_movie_genre = json.load(f)
	with open('../../data/knowledge_base/kb_movie_area.json', encoding='utf-8') as f:
		kb_movie_area = json.load(f)
	with open('../../data/knowledge_base/kb_movie_era.json', encoding='utf-8') as f:
		kb_movie_era = json.load(f)
	with open('../../data/knowledge_base/kb_movie_payment.json', encoding='utf-8') as f:
		kb_movie_payment = json.load(f)
	kb['电影'].update({'片名': kb_movie_name})
	kb['电影'].update({'主演': kb_movie_actor})
	kb['电影'].update({'导演': kb_movie_director})
	kb['电影'].update({'类型': kb_movie_genre})
	kb['电影'].update({'地区': kb_movie_area})
	kb['电影'].update({'年代': kb_movie_era})
	kb['电影'].update({'资费': kb_movie_payment})
	with open('../../data/knowledge_base/kb_music_name.json', encoding='utf-8') as f:
		kb_music_name = json.load(f)
	with open('../../data/knowledge_base/kb_music_artist.json', encoding='utf-8') as f:
		kb_music_artist = json.load(f)
	with open('../../data/knowledge_base/kb_music_genre.json', encoding='utf-8') as f:
		kb_music_genre = json.load(f)
	with open('../../data/knowledge_base/kb_music_era.json', encoding='utf-8') as f:
		kb_music_era = json.load(f)
	with open('../../data/knowledge_base/kb_music_album.json', encoding='utf-8') as f:
		kb_music_album = json.load(f)
	kb['音乐'].update({'歌名': kb_music_name})
	kb['音乐'].update({'歌手': kb_music_artist})
	kb['音乐'].update({'曲风': kb_music_genre})
	kb['音乐'].update({'年代': kb_music_era})
	kb['音乐'].update({'专辑': kb_music_album})
	return kb
knowledge_base = load_knowledge_base()


# global variable
embedding_dim = 25
max_length = 55

round_num = 0
last_belief_states = {
	'major_key': {
		'电影': {
			'片名': []
		},
		'音乐': {
			'歌名': []
		},
		'天气': {
			'城市': [],
			'时间': []
		}
	},
	'informed': {
		'电影': {
			'主演': 'NOT_MENTIONED',
			'导演': 'NOT_MENTIONED',
			'类型': 'NOT_MENTIONED',
			'地区': 'NOT_MENTIONED',
			'年代': 'NOT_MENTIONED',
			'资费': 'NOT_MENTIONED',
		},
		'音乐': {
			'歌手': 'NOT_MENTIONED',
			'曲风': 'NOT_MENTIONED',
			'专辑': 'NOT_MENTIONED',
			'年代': 'NOT_MENTIONED',
		}
	},
	'requested': {
		'电影': [],
		'音乐': [],
		'天气': [],
		'时间': [],
	}
}
sys_act = {"hello": {}}
last_domain = None
KB_pointer = {
	'电影': {
		'片名': None
	},
	'音乐': {
		'歌名': None
	},
	'天气': {
		'城市': '北京市',
		'时间': '今天'
	}
}


# global function
def sentence2words(sentence):  # 分词
	def entity_filter(sentence, entity_set):
		for word in entity_set:
			if len(word) > 1:
				if word in sentence:
					sub_sentences = sentence.split(word)
					sentence = (' ' + word + ' ').join([entity_filter(sub_sentence, entity_set) for sub_sentence in sub_sentences])
					break
		return sentence
	words = ' '.join(jieba.cut(entity_filter(sentence, OTLG_words) + '。', HMM=False)).split()
	return words


def system_init():
	global round_num, last_belief_states, sys_act, last_domain, KB_pointer
	round_num = 0
	last_belief_states = {
		'major_key': {
			'电影': {
				'片名': []
			},
			'音乐': {
				'歌名': []
			},
			'天气': {
				'城市': [],
				'时间': []
			}
		},
		'informed': {
			'电影': {
				'主演': 'NOT_MENTIONED',
				'导演': 'NOT_MENTIONED',
				'类型': 'NOT_MENTIONED',
				'地区': 'NOT_MENTIONED',
				'年代': 'NOT_MENTIONED',
				'资费': 'NOT_MENTIONED',
			},
			'音乐': {
				'歌手': 'NOT_MENTIONED',
				'曲风': 'NOT_MENTIONED',
				'专辑': 'NOT_MENTIONED',
				'年代': 'NOT_MENTIONED',
			}
		},
		'requested': {
			'电影': [],
			'音乐': [],
			'天气': [],
			'时间': [],
		}
	}
	sys_act = {"hello": {}}
	last_domain = None
	KB_pointer = {
		'电影': {
			'片名': None
		},
		'音乐': {
			'歌名': None
		},
		'天气': {
			'城市': '北京市',
			'时间': '今天'
		}
	}


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


# class ConfirmSlotTracker(object):
# 	def __init__(self):
# 		# tf Graph input
# 		self._x_user_nl = tf.placeholder(dtype=tf.float32, shape=(None, max_length, embedding_dim))
# 		self._x_stringmatch = tf.placeholder(dtype=tf.float32, shape=(None, max_length, 1))  # whether MENTIONED
# 		self._is_training = tf.placeholder(dtype=tf.bool)
# 		self._y = tf.placeholder(dtype=tf.float32, shape=(None, 2))
# 		self._lr = tf.Variable(0.0003, trainable=False)
#
# 		self._input_cnn = tf.concat([self._x_user_nl, self._x_stringmatch], 2)  # -> (?, max_length, embedding_size + 1)
# 		kernel_sizes = [1, 2, 3]
# 		self._pools = []
# 		for kernel_size in kernel_sizes:
# 			self._conv = tf.layers.conv1d(
# 				inputs=self._input_cnn,
# 				filters=100,
# 				kernel_size=kernel_size,
# 				strides=1,
# 				padding='same',
# 				activation=tf.nn.relu,
# 				kernel_initializer=tf.random_normal_initializer(0, 0.01))
# 			self._pool = tf.layers.max_pooling1d(self._conv, pool_size=max_length, strides=1)
# 			self._pool = tf.squeeze(self._pool, [1])
# 			self._pools.append(self._pool)
# 		self._output_cnn = tf.concat(values=self._pools, axis=1)
#
# 		self._dnn_hiddenlayer = tf.layers.dense(inputs=self._output_cnn, units=500, activation=tf.nn.relu,
# 		                                        kernel_initializer=tf.random_normal_initializer(0, 0.1))
# 		self._dnn_hiddenlayer = tf.layers.dropout(self._dnn_hiddenlayer, rate=0.5, training=self._is_training)
# 		self._pred = tf.layers.dense(inputs=self._dnn_hiddenlayer, units=2,
# 		                             kernel_initializer=tf.random_normal_initializer(0, 0.1))
# 		self._loss = tf.losses.softmax_cross_entropy(onehot_labels=self._y, logits=self._pred)  # compute cost
#
# 		# gradient clipping
# 		self._tvars = tf.trainable_variables()
# 		self._grads, _ = tf.clip_by_global_norm(tf.gradients(self._loss, self._tvars), clip_norm=5)
# 		self._optimizer = tf.train.AdamOptimizer(self._lr)
# 		self._train_op = self._optimizer.apply_gradients(
# 			grads_and_vars=zip(self._grads, self._tvars),
# 			global_step=tf.train.get_or_create_global_step())
# 		# learning rate update
# 		self._new_lr = tf.placeholder(dtype=tf.float32, shape=[], name="new_learning_rate")
# 		self._lr_update = tf.assign(self._lr, self._new_lr)
#
# 		# Evaluate model
# 		self._results = tf.argmax(self._pred, 1)
# 		self._probability = tf.nn.softmax(self._pred)
# 		self._correct_pred = tf.equal(self._results, tf.argmax(self._y, 1))
# 		self._accuracy = tf.reduce_mean(tf.cast(self._correct_pred, tf.float32))
#
# 	def assign_lr(self, session, lr_value):
# 		session.run(self._lr_update, feed_dict={self._new_lr: lr_value})
#
# 	def accuracy(self):
# 		return self._accuracy
#
# 	def train_op(self):
# 		return self._train_op
#
# 	def loss(self):
# 		return self._loss


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
	# data['user_nl'], data["sys_act"], data["domain"], data["label"]
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
	# data['user_nl'], data["domain"], data["slot"], data["label"]
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
	# data['user_nl'], data['sys_act'], data['domain'], data['slot'], data['label']
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
	# data['user_nl'], data['value'], data['label']

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


# def data2num_confirm_slot(batch_data, has_label=False):
# 	# data['user_nl'], data['domain'], data['slot'], data['label']
# 	def is_zh(content):
# 		zhPattern = re.compile(u'^[\u4e00-\u9fa5]+$')
# 		match = zhPattern.search(content)
# 		return True if match else False
#
# 	batchsize = len(batch_data)
# 	x_user_nl = np.zeros((batchsize, max_length, embedding_dim))
# 	x_stringmatch = np.zeros((batchsize, max_length, 1))
# 	x_user_nl_len = np.zeros(batchsize, dtype='int32')
# 	y = np.zeros((batchsize, 2))
#
# 	for batch_id, data in enumerate(batch_data):
# 		# x_user_nl, x_stringmatch, x_user_nl_len
# 		words = filter(lambda x: is_zh(x), data['user_nl'])
# 		for word_id, word in enumerate(words):
# 			if word_id == max_length:
# 				break
# 			if word in vocab_dict:
# 				x_user_nl[batch_id, word_id, :] = embedding_table[word]
# 			else:
# 				x_user_nl[batch_id, word_id, :] = embedding_table['unk']
# 			if word in semantic_dict_request_slot[data['domain']][data['slot']]:
# 				x_stringmatch[batch_id, word_id, 0] = 1
# 		x_user_nl_len[batch_id] = len(data['user_nl'])
#
# 		# y
# 		if has_label:
# 			label_requestable_slot = {'MENTIONED': 0, 'NOT_MENTIONED': 1}
# 			y[batch_id, label_requestable_slot[data['label']]] = 1
# 	if has_label:
# 		return {'user_nl': x_user_nl, 'stringmatch': x_stringmatch, 'user_nl_len': x_user_nl_len, 'label': y}
# 	else:
# 		return {'user_nl': x_user_nl, 'stringmatch': x_stringmatch, 'user_nl_len': x_user_nl_len}


def data2num_act(batch_data, has_label=False):
	# data['user_nl'], data['user_act'], data['label']

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
	print('load the model ...')
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
	# with tf.name_scope('ConfirmSlotTracker'):
	# 	with tf.name_scope('movie'):
	# 		with tf.variable_scope('c_s_actor'):
	# 			confirm_slot_tracker_actor = ConfirmSlotTracker()
	# 		with tf.variable_scope('c_s_director'):
	# 			confirm_slot_tracker_director = ConfirmSlotTracker()
	# 		with tf.variable_scope('c_s_genre'):
	# 			confirm_slot_tracker_genre = ConfirmSlotTracker()
	# 		with tf.variable_scope('c_s_area'):
	# 			confirm_slot_tracker_area = ConfirmSlotTracker()
	# 		with tf.variable_scope('c_s_era'):
	# 			confirm_slot_tracker_era = ConfirmSlotTracker()
	# 		with tf.variable_scope('c_s_payment'):
	# 			confirm_slot_tracker_payment = ConfirmSlotTracker()
	# 	with tf.name_scope('music'):
	# 		with tf.variable_scope('c_s_artist'):
	# 			confirm_slot_tracker_artist = ConfirmSlotTracker()
	# 		with tf.variable_scope('c_s_album'):
	# 			confirm_slot_tracker_album = ConfirmSlotTracker()
	# 		with tf.variable_scope('c_s_style'):
	# 			confirm_slot_tracker_style = ConfirmSlotTracker()
	# 		with tf.variable_scope('c_s_musicera'):
	# 			confirm_slot_tracker_music_era = ConfirmSlotTracker()
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
		saver.restore(sess, "../../model/tmp_final/model.ckpt")
		
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
		# confirm_slot_tracker = {
		# 	'电影': {
		# 		'主演': confirm_slot_tracker_actor,
		# 		'导演': confirm_slot_tracker_director,
		# 		'类型': confirm_slot_tracker_genre,
		# 		'地区': confirm_slot_tracker_area,
		# 		'年代': confirm_slot_tracker_era,
		# 		'资费': confirm_slot_tracker_payment
		# 	},
		# 	'音乐': {
		# 		'歌手': confirm_slot_tracker_artist,
		# 		'专辑': confirm_slot_tracker_album,
		# 		'曲风': confirm_slot_tracker_style,
		# 		'年代': confirm_slot_tracker_music_era
		# 	}
		# }
		act_tracker = {
			'first': act_tracker_first,
			'second': act_tracker_second,
			'third': act_tracker_third,
			'last': act_tracker_last,
			'other': act_tracker_other,
			'affirm': act_tracker_affirm,
			'deny': act_tracker_deny
		}
		
		def slu(current_round, last_domain):
			def evaluate_user_act(current_round, detected_user_acts):
				for act in user_acts:
					current_round['user_act'] = act
					num_act = data2num_act([current_round])
					prediction = sess.run(act_tracker[act]._probability,
					                      feed_dict={
						                      act_tracker[act]._x_user_nl: num_act['user_nl'],
						                      act_tracker[act]._x_stringmatch: num_act['stringmatch'],
						                      act_tracker[act]._is_training: False}
					                      )
					del num_act
					if prediction[0][0] > 0.5:
						detected_user_acts.append(act)
				return detected_user_acts
			def evaluate_domain(current_round, last_domain):
				domain_detected = {}
				for domain in domain_set:
					current_round['domain'] = domain
					num_domain = data2num_domain([current_round])
					prediction = sess.run(domain_tracker[domain]._probability,
					                      feed_dict={
						                      domain_tracker[domain]._x_user_nl: num_domain['user_nl'],
						                      domain_tracker[domain]._x_stringmatch: num_domain['stringmatch'],
						                      domain_tracker[domain]._x_sys_act: num_domain['sys_act'],
						                      domain_tracker[domain]._is_training: False}
					                      )
					del num_domain
					domain_detected.update({domain: prediction[0][0]})
				if not last_domain:
					domain = sorted(domain_detected.keys(), key=lambda x: domain_detected[x], reverse=True)[0]
				else:
					domain_detected = {key: value for (key, value) in domain_detected.items() if value > 0.5}
					if domain_detected:
						domain = sorted(domain_detected.keys(), key=lambda x: domain_detected[x], reverse=True)[0]
					else:
						domain = last_domain
				return domain
			def evaluate_request_slot(current_round, domain, requested_slots):
				for slot in request_slots[domain]:
					current_round['domain'] = domain
					current_round['slot'] = slot
					num_request_slot = data2num_request_slot([current_round])
					prediction = sess.run(request_slot_tracker[domain][slot]._probability,
					                      feed_dict={
						                      request_slot_tracker[domain][slot]._x_user_nl: num_request_slot['user_nl'],
						                      request_slot_tracker[domain][slot]._x_stringmatch: num_request_slot['stringmatch'],
						                      request_slot_tracker[domain][slot]._is_training: False}
					                      )
					del num_request_slot
					if prediction[0][0] > 0.5:
						requested_slots[domain].append(slot)
				return requested_slots
			def evaluate_major_key(current_round, domain, major_key_mem):
				user_nl = current_round['user_nl']
				for slot in OTLG[domain]['major_key']:
					for value in OTLG[domain]['informable'][slot]:
						for word in synonym_dict[domain][slot][value]:
							if word in user_nl and len(word) > 1:
								major_key_mem[domain][slot].append(value)
								break
				return major_key_mem
			def evaluate_inform_slot(current_round, domain, informed_slots):
				for slot in inform_slots[domain]:
					current_round['domain'] = domain
					current_round['slot'] = slot
					num_inform_slot = data2num_inform_slot([current_round])
					prediction = sess.run(inform_slot_tracker[domain][slot]._probability,
					                      feed_dict={
						                      inform_slot_tracker[domain][slot]._x_user_nl: num_inform_slot['user_nl'],
						                      inform_slot_tracker[domain][slot]._x_stringmatch: num_inform_slot[
							                      'stringmatch'],
						                      inform_slot_tracker[domain][slot]._x_sys_act: num_inform_slot['sys_act'],
						                      inform_slot_tracker[domain][slot]._is_training: False})
					del num_inform_slot
					inform_slot_label = ['MENTIONED', 'NOT_MENTIONED', 'DONT_CARE']
					informed_slots[domain].update({slot: inform_slot_label[np.argmax(prediction, 1)[0]]})
					
					if informed_slots[domain][slot] == "MENTIONED":
						informed_slots[domain][slot] = {}
						data_value = []
						values = []
						for value in synonym_dict[domain][slot]:
							for word in synonym_dict[domain][slot][value]:
								if word in current_round['user_nl']:
									current_round['value'] = value
									data_value.append(copy.deepcopy(current_round))
									values.append(value)
									break
						num_value = data2num_value(data_value)
						prediction = sess.run(value_tracker[domain][slot]._probability,
						                      feed_dict={
							                      value_tracker[domain][slot]._x_user_nl: num_value['user_nl'],
							                      value_tracker[domain][slot]._x_stringmatch: num_value['stringmatch'],
							                      value_tracker[domain][slot]._is_training: False})
						del num_value
						value_label = ['LIKE', 'DISLIKE']
						# print(prediction)
						label_id = np.argmax(prediction, 1)
						for value_id, value in enumerate(values):
							informed_slots[domain][slot].update({value: value_label[label_id[value_id]]})
						if not informed_slots[domain][slot]:
							informed_slots[domain][slot] = "NOT_MENTIONED"
				return informed_slots
			# def evaluate_confirm_slot(current_round, domain, requested_slots):
			# 	for slot in confirm_slots[domain]:
			# 		current_round['domain'] = domain
			# 		current_round['slot'] = slot
			# 		num_confirm_slot = data2num_confirm_slot([current_round])
			# 		prediction = sess.run(confirm_slot_tracker[domain][slot]._probability,
			# 		                      feed_dict={
			# 			                      confirm_slot_tracker[domain][slot]._x_user_nl: num_confirm_slot[
			# 				                      'user_nl'],
			# 			                      confirm_slot_tracker[domain][slot]._x_stringmatch: num_confirm_slot[
			# 				                      'stringmatch'],
			# 			                      confirm_slot_tracker[domain][slot]._is_training: False}
			# 		                      )
			# 		del num_confirm_slot
			# 		if prediction[0][0] > 0.5:
			# 			requested_slots[domain].append(slot)
			# 	requested_slots[domain] = list(set(requested_slots[domain]))
			# 	return requested_slots
			sys_act = current_round['sys_act']
			domain = None
			detected_user_acts = []
			major_key_mem = {
				'电影': {
					'片名': []
				},
				'音乐': {
					'歌名': []
				},
				'天气': {
					'城市': [],
					'时间': []
				}
			}
			# first, second, third, last, other
			if 'inform_some_match' in sys_act:
				detected_user_acts = evaluate_user_act(current_round, detected_user_acts)
				detected_user_acts = list(filter(lambda x: x in ['first', 'second', 'third', 'last', 'other'], detected_user_acts))
				major_key = {'电影': '片名', '音乐': '歌名'}
				for act in detected_user_acts:
					if act in ['first', 'second', 'third', 'last', 'other']:
						domain = last_domain
					if act == 'first':
						major_key_mem[domain][major_key[domain]] = [sys_act['inform_some_match'][domain][major_key[domain]][0]]
					if act == 'second':
						major_key_mem[domain][major_key[domain]] = [sys_act['inform_some_match'][domain][major_key[domain]][1]]
					if act == 'third':
						index = min(len(sys_act['inform_some_match'][domain][major_key[domain]]), 3)
						major_key_mem[domain][major_key[domain]] = [sys_act['inform_some_match'][domain][major_key[domain]][index-1]]
					if act == 'last':
						major_key_mem[domain][major_key[domain]] = [sys_act['inform_some_match'][domain][major_key[domain]][-1]]
			# domain
			if domain is None:
				domain = evaluate_domain(current_round, last_domain)
			# requested_slots
			requested_slots = {domain: []}
			requested_slots = evaluate_request_slot(current_round, domain, requested_slots)
			# major_key
			if domain in ['电影', '音乐', '天气']:
				major_key_mem = evaluate_major_key(current_round, domain, major_key_mem)
			# informed_slots, confirmed_slots
			informed_slots = {domain: {}}
			if domain in ['电影', '音乐']:
				informed_slots = evaluate_inform_slot(current_round, domain, informed_slots)
				# requested_slots = evaluate_confirm_slot(current_round, domain, requested_slots)
			# affirm, deny
			if 'confirm' in sys_act:
				detected_user_acts = evaluate_user_act(current_round, detected_user_acts)
				detected_user_acts = list(filter(lambda x: x in ['affirm', 'deny'], detected_user_acts))
				for act in detected_user_acts:
					if act in ['affirm', 'deny']:
						domain = last_domain
						slot = list(sys_act['confirm'][domain].keys())[0]
						value = sys_act['confirm'][domain][slot]
						if act == 'affirm':
							if domain in informed_slots:
								if type(informed_slots[domain][slot]) == dict:
									informed_slots[domain][slot].update({value: 'LIKE'})
								else:
									informed_slots[domain][slot] = {value: 'LIKE'}
							else:
								informed_slots = {domain: {slot: {value: 'LIKE'}}}
						elif act == 'deny':
							if domain in informed_slots:
								if type(informed_slots[domain][slot]) == dict:
									informed_slots[domain][slot].update({value: 'DISLIKE'})
								else:
									informed_slots[domain][slot] = {value: 'DISLIKE'}
							else:
								informed_slots = {domain: {slot: {value: 'DISLIKE'}}}
			
			major_key_mem = {domain: major_key_mem[domain]} if domain in major_key_mem else {domain: {}}
			current_belief_states = {'major_key': major_key_mem, 'informed': informed_slots, 'requested': requested_slots}
			current_domain = domain
			return current_belief_states, current_domain, detected_user_acts

		def update_belief_states(current_round, current_belief_states, current_domain, last_belief_states):
			sys_act = current_round['sys_act']
			major_key_mem = current_belief_states['major_key']
			current_informed_slots = current_belief_states['informed']
			current_requested_slots = current_belief_states['requested']
			domain = current_domain
			belief_states = copy.deepcopy(last_belief_states)
			
			# major_key
			belief_states['major_key'] = major_key_mem
			# informed slots
			if domain in belief_states['informed']:
				if sys_act in ['inform', 'inform_no_match', 'inform_one_match', 'inform_some_match']:
					for slot in current_informed_slots[domain]:
						if current_informed_slots[domain][slot] != 'NOT_MENTIONED':
							belief_states['informed'][domain][slot] = current_informed_slots[domain][slot]
				else:
					for slot in current_informed_slots[domain]:
						if current_informed_slots[domain][slot] == 'DONT_CARE':
							belief_states['informed'][domain][slot] = 'DONT_CARE'
						elif type(current_informed_slots[domain][slot]) == dict:
							if type(belief_states['informed'][domain][slot]) == dict:
								belief_states['informed'][domain][slot].update(current_informed_slots[domain][slot])
							else:
								belief_states['informed'][domain][slot] = current_informed_slots[domain][slot]
			# requested slots
			belief_states['requested'] = {'电影': [], '音乐': [], '天气': [], '时间': []}
			if current_requested_slots[domain]:
				belief_states['requested'][domain] = current_requested_slots[domain]
			elif last_belief_states['requested'][domain]:
				for slot in major_key_mem[domain]:
					if major_key_mem[domain][slot]:
						belief_states['requested'][domain] = last_belief_states['requested'][domain]
						break
			return belief_states
		# start
		system_init()
		while True:
			# try:
			if round_num == 0:
				print("sys 0: 欢迎使用清华大学电子系多领域信息问询系统，您可以对电影、音乐、时间、天气信息进行查询。")
			
			# 用户输入语句
			input_usr = input("usr " + str(round_num) + ": ")
			if input_usr == "重来" or "谢谢" in input_usr:  # 用户重来
				print('\n\n')
				system_init() 
			elif input_usr == "结束":  # 用户结束
				break
			else:  # 用户正常对话
				round_num += 1
				current_round = {}
				current_round["user_nl"] = sentence2words(input_usr)
				current_round["sys_act"] = sys_act
			
				# 计算 belief states
				current_belief_states, current_domain, detected_user_acts = slu(current_round, last_domain)
				# print(current_belief_states, current_domain, detected_user_acts)
				belief_states = update_belief_states(current_round, current_belief_states, current_domain, last_belief_states)
				last_belief_states = copy.deepcopy(belief_states)
				last_domain = copy.deepcopy(current_domain)
			
				# 输出中间结果
				print("  belief_states: ")
				for domain in ['电影', '音乐']:
					print(domain + ':', end=' ')
					for slot in belief_states['informed'][domain]:
						if belief_states['informed'][domain][slot] != "NOT_MENTIONED":
							print(slot + ': ', belief_states['informed'][domain][slot], end='；')
					print()
				print("  requested_slots: ", belief_states['requested'])
				
				# 生成 system_acts
				sys_act, KB_pointer = rule_based_policy(belief_states, current_domain, KB_pointer, knowledge_base)
				# 生成回复
				sys_nl = rule_based_NLG(sys_act, KB_pointer, knowledge_base)
				
				print("sys " + str(round_num) + ": " + sys_nl)
			# except Exception as e:
			# 	print("=== 出现异常重新开始 ===")
			# 	system_init()
