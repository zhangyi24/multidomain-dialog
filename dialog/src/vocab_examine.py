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
import chardet

with open('../data/word2vec/vocab_dict.json', 'r', encoding='utf-8') as f:
	vocab_dict = json.load(f)
with open("../data/word2vec/OTLG_words.txt", 'r', encoding='utf-8') as f:
	new_words = f.read().split()
for word in vocab_dict:
	if vocab_dict[word] in range(0, 5000) and word not in new_words:
		print(word, vocab_dict[word])
