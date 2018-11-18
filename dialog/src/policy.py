#!/usr/bin/env python3
# coding=utf-8

import json
import random


def search_kb(domain, informed_slots, kb):
	LIKED = set()
	if domain == '电影':
		LIKED = set(kb['片名'].keys())
	elif domain == '音乐':
		LIKED = set(kb['歌名'].keys())
	DISLIKED = set()
	
	for slot in informed_slots:
		if type(informed_slots[slot]) == dict:
			constraint = {'LIKE': set(), 'DISLIKE': set()}
			for value in informed_slots[slot]:
				if informed_slots[slot][value] == "LIKE":
					constraint["LIKE"] = constraint["LIKE"] | find_items(domain, slot, value, kb)
				elif informed_slots[slot][value] == "DISLIKE":
					constraint["DISLIKE"] = constraint["DISLIKE"] | find_items(domain, slot, value, kb)
			if constraint["LIKE"]:
				LIKED = LIKED & constraint["LIKE"]
			DISLIKED = DISLIKED | constraint["DISLIKE"]
	return list(LIKED - DISLIKED)


def find_items(domain, slot, value, kb):
	items = set()
	for item in kb[slot][value]:
		if domain == '电影':
			items.add(item['片名'])
		elif domain == '音乐':
			items.add(item['歌名'])
	return items


def rule_based_policy(belief_states, domain, KB_pointer, knowledge_base):
	sys_act = {}
	if domain == '电影':
		informed_slots = belief_states['informed'][domain]
		requested_slots = belief_states['requested'][domain]
		major_key = belief_states['major_key'][domain]['片名']
		slots2request = ['类型', '主演', '导演', '地区', '资费', '年代']
		if major_key:
			if len(major_key) == 1:
				KB_pointer['电影']['片名'] = major_key[0]
				sys_act = {'inform_one_match': {'电影': {'片名': major_key}}}
			else:
				sys_act = {'inform_some_match': {'电影': {'片名': major_key}}}
		else:
			search_results = search_kb(domain, informed_slots, knowledge_base[domain])
			if len(search_results) == 1:
				KB_pointer['电影']['片名'] = search_results[0]
				sys_act = {'inform_one_match': {'电影': {'片名': search_results}}}
			elif len(search_results) == 2:
				sys_act = {'inform_some_match': {'电影': {'片名': search_results}}}
			elif len(search_results) == 0:
				sys_act = {'inform_no_match': {'电影': {'片名': []}}}
			if len(sys_act) == 0:
				# 处理 informable slots
				for slot in informed_slots:
					if informed_slots[slot] != "NOT_MENTIONED":
						slots2request.remove(slot)
				if len(slots2request) > 0:
					sys_act = {'request': {domain: slots2request[0]}}
				else:
					random.shuffle(search_results)
					sys_act = {'inform_some_match': {'电影': {'片名': search_results[0:3]}}}
		# 处理 requestable slots
		if len(requested_slots) > 0 and KB_pointer['电影']['片名']:
			sys_act = {'inform': {domain: requested_slots}}
	elif domain == '音乐':
		informed_slots = belief_states['informed'][domain]
		requested_slots = belief_states['requested'][domain]
		major_key = belief_states['major_key'][domain]['歌名']
		slots2request = ['歌手', '曲风', '专辑', '年代']
		if major_key:
			if len(major_key) == 1:
				KB_pointer['音乐']['歌名'] = major_key[0]
				sys_act = {'inform_one_match': {'音乐': {'歌名': major_key}}}
			else:
				sys_act = {'inform_some_match': {'音乐': {'歌名': major_key}}}
		else:
			search_results = search_kb(domain, informed_slots, knowledge_base[domain])
			if len(search_results) == 1:
				KB_pointer['音乐']['歌名'] = search_results[0]
				sys_act = {'inform_one_match': {'音乐': {'歌名': search_results}}}
			elif len(search_results) == 2:
				sys_act = {'inform_some_match': {'音乐': {'歌名': search_results}}}
			elif len(search_results) == 0:
				sys_act = {'inform_no_match': {'音乐': {'歌名': []}}}
			if len(sys_act) == 0:
				# 处理 informable slots
				for slot in informed_slots:
					if informed_slots[slot] != "NOT_MENTIONED":
						slots2request.remove(slot)
				if len(slots2request) > 0:
					sys_act = {'request': {domain: slots2request[0]}}
				else:
					random.shuffle(search_results)
					sys_act = {'inform_some_match': {'音乐': {'歌名': search_results[0:3]}}}
		# 处理 requestable slots
		if len(requested_slots) > 0 and KB_pointer['音乐']['歌名']:
			sys_act = {'inform': {domain: requested_slots}}
	elif domain == '天气':
		requested_slots = belief_states['requested'][domain]
		major_key = belief_states['major_key'][domain]
		for slot in major_key:
			if major_key[slot]:
				KB_pointer[domain][slot] = major_key[slot][0]
		# 处理 requestable slots
		if len(requested_slots) > 0:
			sys_act = {'inform': {domain: requested_slots}}
	elif domain == '时间':
		requested_slots = belief_states['requested'][domain]
		# 处理 requestable slots
		if len(requested_slots) > 0:
			sys_act = {'inform': {domain: requested_slots}}
	if not sys_act:
		sys_act = {'repeat': {}}
	return sys_act, KB_pointer



if __name__ == '__main__':
	def load_knowledge_base():
		kb = {'电影': {}, '音乐': {}}
		with open('../data/knowledge_base/kb_movie_name.json', encoding='utf-8') as f:
			kb_movie_name = json.load(f)
		with open('../data/knowledge_base/kb_movie_actor.json', encoding='utf-8') as f:
			kb_movie_actor = json.load(f)
		with open('../data/knowledge_base/kb_movie_director.json', encoding='utf-8') as f:
			kb_movie_director = json.load(f)
		with open('../data/knowledge_base/kb_movie_genre.json', encoding='utf-8') as f:
			kb_movie_genre = json.load(f)
		with open('../data/knowledge_base/kb_movie_area.json', encoding='utf-8') as f:
			kb_movie_area = json.load(f)
		with open('../data/knowledge_base/kb_movie_era.json', encoding='utf-8') as f:
			kb_movie_era = json.load(f)
		with open('../data/knowledge_base/kb_movie_payment.json', encoding='utf-8') as f:
			kb_movie_payment = json.load(f)
		kb['电影'].update({'片名': kb_movie_name})
		kb['电影'].update({'主演': kb_movie_actor})
		kb['电影'].update({'导演': kb_movie_director})
		kb['电影'].update({'类型': kb_movie_genre})
		kb['电影'].update({'地区': kb_movie_area})
		kb['电影'].update({'年代': kb_movie_era})
		kb['电影'].update({'资费': kb_movie_payment})
		with open('../data/knowledge_base/kb_music_name.json', encoding='utf-8') as f:
			kb_music_name = json.load(f)
		with open('../data/knowledge_base/kb_music_artist.json', encoding='utf-8') as f:
			kb_music_artist = json.load(f)
		with open('../data/knowledge_base/kb_music_genre.json', encoding='utf-8') as f:
			kb_music_genre = json.load(f)
		with open('../data/knowledge_base/kb_music_era.json', encoding='utf-8') as f:
			kb_music_era = json.load(f)
		with open('../data/knowledge_base/kb_music_album.json', encoding='utf-8') as f:
			kb_music_album = json.load(f)
		kb['音乐'].update({'歌名': kb_music_name})
		kb['音乐'].update({'歌手': kb_music_artist})
		kb['音乐'].update({'曲风': kb_music_genre})
		kb['音乐'].update({'年代': kb_music_era})
		kb['音乐'].update({'专辑': kb_music_album})
		return kb
	kb = load_knowledge_base()
	informed_slots = {'歌手': 'DONT_CARE', '曲风': {'流行': 'LIKE'}, '年代': {'零零': 'LIKE'}, '专辑': {'我很忙': 'LIKE'}}
	results = search_kb('音乐', informed_slots, kb['音乐'])
	print(results, len(results))
