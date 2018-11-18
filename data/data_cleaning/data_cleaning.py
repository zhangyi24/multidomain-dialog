#!/usr/bin/env python3
# coding=utf-8

import csv
import json
import copy

with open('rounds_dialog_10000.json', 'r', encoding='utf-8') as f:
	rounds_dialog = json.load(f)
with open('人机对话语料采集-0--阿里众包处理结果详情下载-utf-8.csv', 'r', encoding='utf-8') as f:
	data = csv.reader(f)
	data = list(data)
for row in data[1:]:
	index = int(json.loads(row[2])["id"])
	if index >= 10000:
		continue
	reply = row[5]
	key = "回复%s" % row[3]
	rounds_dialog[index].update({key: reply})
	

	
rounds_dialog_20000 = []
for round_id, round in enumerate(rounds_dialog):
	for reply_id in [1, 2]:
		reply = round['回复%d' % reply_id]
		print(2 * round_id + reply_id - 1)
		rounds_dialog_20000.append(copy.deepcopy(round))
		rounds_dialog_20000[2 * round_id + reply_id - 1].update({'user_nl': reply})
		rounds_dialog_20000[2 * round_id + reply_id - 1].update({'reply_id': reply_id})
		rounds_dialog_20000[2 * round_id + reply_id - 1].update({'reply_example': round['user_nl']})
		rounds_dialog_20000[2 * round_id + reply_id - 1].pop('回复1')
		rounds_dialog_20000[2 * round_id + reply_id - 1].pop('回复2')
with open('rounds_dialog_20000.json', 'w', encoding='utf-8') as f:
	json.dump(rounds_dialog_20000, f, ensure_ascii=False, indent=2)


for round_id, round in enumerate(rounds_dialog_20000):
	if "user_nl" in round:
		print(round['id'])
		print(round['sys_nl'])
		print(round['reply_example'])
		print(round['user_nl'])
		print(round_id)
		print()

