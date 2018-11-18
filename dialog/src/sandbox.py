#!/usr/bin/env python3
# coding=utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# import collections
# import math
# import pickle
# import copy
import random
# import numpy as np
# import tensorflow as tf
import jieba
import json
# import matplotlib as mpl
# from sklearn.manifold import TSNE
# import matplotlib.pyplot as plt
import time


pass    # round['user_nl_words']
def jieba_init():
	def del_redundant_words(filename):
		with open(filename, 'r', encoding="utf-8") as f:
			words = json.load(f)
			for word in words:
				jieba.del_word(word)
	jieba.load_userdict("../../data/word2vec/OTLG_words.txt")  # 添加自定义词典，用来分词
	jieba.load_userdict("../../data/word2vec/words_to_add.txt")  # 添加自定义词典，用来分词
	del_redundant_words("../../data/word2vec/words_to_del.json")
jieba_init()

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

with open('../../data/word2vec/OTLG_words.json', 'r', encoding='utf-8') as f:
	OTLG_words = json.load(f)
with open('../../data/rounds_dialog_20000.json', 'r', encoding='utf-8') as f:
	rounds_dialog = json.load(f)
for round_dialog in rounds_dialog:
	round_dialog['user_nl_words'] = sentence2words(round_dialog['user_nl'])

with open('../../data/rounds_dialog_20000.json', 'w', encoding='utf-8') as f:
	json.dump(rounds_dialog, f, ensure_ascii=False, indent=2)


pass    # change weather requestable slots
# def change_weather_request_slots(filedir):
# 	with open(filedir, 'r', encoding='utf-8') as f:
# 		rounds_dialog = json.load(f)
# 	for round_dialog in rounds_dialog:
# 		if 'request' in round_dialog['user_act']:
# 			if '天气' in round_dialog['user_act']['request']:
# 				round_dialog['user_act']['request']['天气'] = ['天气']
# 	with open(filedir, 'w', encoding='utf-8') as f:
# 		json.dump(rounds_dialog, f, ensure_ascii=False, indent=2)
# change_weather_request_slots('../data/rounds_dialog_20000.json')


pass  # 检测'user_act'中存在'first/second/last+request/confirm'的错误
# with open('../data/rounds_dialog_20000.json', 'r', encoding='utf-8') as f:
# 	rounds_dialog = json.load(f)
# num = 0
# for round_dialog in rounds_dialog:
# 	if '两' in round_dialog['sys_nl']:
# 		user_acts = ['first+request', 'second+request', 'last+request',
# 		             'first+confirm', 'second+confirm', 'last+confirm']
# 		for user_act in user_acts:
# 			if user_act in round_dialog['user_act']:
# 				print(round_dialog['user_nl'])
# 				print(round_dialog['user_act'])
# 				print(round_dialog['id'])
# 				print(num)
# 				print()
# 				num += 1

pass  # 纠正'user_act'中存在'first/second/last+request/confirm'的错误
# def correct_user_act(filedir):
# 	with open(filedir, 'r', encoding='utf-8') as f:
# 		rounds_dialog = json.load(f)
# 	num = 0
# 	for round_dialog in rounds_dialog:
# 		if '两' in round_dialog['sys_nl']:
# 			user_acts = ['first+request', 'second+request', 'last+request',
# 			             'first+confirm', 'second+confirm', 'last+confirm']
# 			for user_act in user_acts:
# 				if user_act in round_dialog['user_act']:
# 					round_dialog['user_act'].pop(user_act)
# 					round_dialog['user_act'].update({user_act.split('+')[0]: {}})
# 	with open(filedir, 'w', encoding='utf-8') as f:
# 		json.dump(rounds_dialog, f, ensure_ascii=False, indent=2)
# correct_user_act('../data/rounds_dialog_20000.json')







