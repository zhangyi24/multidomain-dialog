#!/usr/bin/env python3
# coding=utf-8

import json
import copy
import random

# load files
with open('../../data/word2vec/OTLG_words.json', 'r', encoding='utf-8') as f:
	OTLG_words = json.load(f)
with open('../../definition/OTLG.json', 'r', encoding='utf-8') as f:
	OTLG = json.load(f)
with open('../../data/rounds_dialog_20000.json', 'r', encoding='utf-8') as f:
	rounds_dialog = json.load(f)

domain_set = ["电影", "音乐", "天气", "时间"]
request_slots = {domain: OTLG[domain]['requestable'] for domain in domain_set}
inform_slots = {domain: set(OTLG[domain]['informable'].keys()) - set(OTLG[domain]['major_key']) for domain in ['电影', '音乐']}
confirm_slots = {domain: set(OTLG[domain]['informable'].keys()) - set(OTLG[domain]['major_key']) for domain in ['电影', '音乐']}
user_acts = ['first', 'second', 'third', 'last', 'other', 'affirm', 'deny']

domain_label = ["MENTIONED", "NOT_MENTIONED"]
request_slot_label = ["MENTIONED", "NOT_MENTIONED"]
inform_slot_label = ["MENTIONED", "NOT_MENTIONED", "DONT_CARE"]
value_label = ["LIKE", "DISLIKE"]
confirm_slot_label = ["MENTIONED", "NOT_MENTIONED"]
act_label = ["MENTIONED", "NOT_MENTIONED"]
train_size = 20000


class DataSet(object):
	def __init__(self, rounds_dialog):
		self._rounds_dialog = rounds_dialog
		self._OTLG = OTLG
		
		self._domain_data = self.build_domain_dataset()
		self._request_slot_data = self.build_request_slot_dataset()
		self._inform_slot_data = self.build_inform_slot_dataset()
		self._value_data = self.build_value_dataset()
		# self._confirm_slot_data = self.build_confirm_slot_dataset()
		self._act_data = self.build_act_dataset()
		
		# index for generate next batch
		self._domain_data_index = {}
		self._request_slot_data_index = {}
		self._inform_slot_data_index = {}
		self._value_data_index = {}
		# self._confirm_slot_data_index = {}
		self._act_data_index = {}
		self.data_index_init()
	
	def data_index_init(self):
		# self._domain_data_index
		for domain in domain_set:
			self._domain_data_index[domain] = {}
			for label in domain_label:
				self._domain_data_index[domain][label] = 0

		# self._request_slot_data_index
		for domain in domain_set:
			self._request_slot_data_index[domain] = {}
			for slot in request_slots[domain]:
				self._request_slot_data_index[domain][slot] = {}
				for label in request_slot_label:
					self._request_slot_data_index[domain][slot][label] = 0

		# self._inform_slot_data_index
		for domain in ['电影', '音乐']:
			self._inform_slot_data_index[domain] = {}
			for slot in inform_slots[domain]:
				self._inform_slot_data_index[domain][slot] = {}
				for label in inform_slot_label:
					self._inform_slot_data_index[domain][slot][label] = 0
					
		# self._value_data_index
		for domain in ['电影', '音乐']:
			self._value_data_index[domain] = {}
			for slot in inform_slots[domain]:
				self._value_data_index[domain][slot] = {}
				for label in value_label:
					self._value_data_index[domain][slot][label] = 0
				
		# self._confirm_slot_data_index
		# for domain in ['电影', '音乐']:
		# 	self._confirm_slot_data_index[domain] = {}
		# 	for slot in confirm_slots[domain]:
		# 		self._confirm_slot_data_index[domain][slot] = {}
		# 		for label in confirm_slot_label:
		# 			self._confirm_slot_data_index[domain][slot][label] = 0
				
		pass    # self._act_data_index
		for act in user_acts:
			self._act_data_index[act] = {}
			for label in act_label:
				self._act_data_index[act][label] = 0
					
	def build_domain_dataset(self):
		def extract_one_round(one_round, domain_data):
			user_nl = one_round["user_nl_words"]
			sys_act = one_round["sys_act"]
			one_round['domain_label'] = None
			for domain in domain_set:
				mentioned_flag = False
				user_act_set = ["inform", "request", "confirm"]
				for user_act_type in user_act_set:
					if user_act_type in one_round["user_act"]:
						if domain in one_round["user_act"][user_act_type]:
							data = [{"user_nl": user_nl, "sys_act": sys_act, "domain": domain, "label": "MENTIONED"}]
							domain_data[domain]["MENTIONED"].extend(copy.deepcopy(data))
							mentioned_flag = True
							one_round['domain_label'] = domain
							break
				if not mentioned_flag:
					user_act_set = ["first", "second", "third", "last", "other", "affirm", "deny"]
					for user_act_type in user_act_set:
						if user_act_type in one_round["user_act"]:
							for sys_act_type in ['inform_some_match', 'confirm']:
								if sys_act_type in one_round["sys_act"]:
									if domain in one_round["sys_act"][sys_act_type]:
										data = [{"user_nl": user_nl, "sys_act": sys_act, "domain": domain, "label": "MENTIONED"}]
										domain_data[domain]["MENTIONED"].extend(copy.deepcopy(data))
										mentioned_flag = True
										break
				if not mentioned_flag:
					data = [{"user_nl": user_nl, "sys_act": sys_act, "domain": domain, "label": "NOT_MENTIONED"}]
					domain_data[domain]["NOT_MENTIONED"].extend(copy.deepcopy(data))
			return domain_data
			
		domain_data = {}
		# init
		for domain in domain_set:
			domain_data[domain] = {}
			for label in domain_label:
				domain_data[domain][label] = []
		# build dataset
		for one_round in self._rounds_dialog:
			domain_data = extract_one_round(one_round, domain_data)
		return domain_data
	
	def next_batch_domain(self, batchsize, domain):
		domain_data = []
		start = copy.deepcopy(self._domain_data_index)
		end = copy.deepcopy(self._domain_data_index)
		for label in domain_label:
			num_total_data = len(self._domain_data[domain][label])
			if start[domain][label] + batchsize[label] >= num_total_data:
				# Get the rest examples in this epoch
				num_rest_data = num_total_data - start[domain][label]
				data_rest_part = copy.deepcopy(self._domain_data[domain][label][
				                               start[domain][label]:
				                               num_total_data])
				# Shuffle the data
				random.shuffle(self._domain_data[domain][label])
				# Start next epoch
				start[domain][label] = 0
				self._domain_data_index[domain][label] = batchsize[label] - num_rest_data
				end[domain][label] = self._domain_data_index[domain][label]
				data_new_part = copy.deepcopy(self._domain_data[domain][label][
				                              start[domain][label]:
				                              end[domain][label]])
				domain_data.extend(data_rest_part + data_new_part)
			else:
				self._domain_data_index[domain][label] += batchsize[label]
				end[domain][label] = self._domain_data_index[domain][label]
				domain_data.extend(copy.deepcopy(self._domain_data[domain][label][
				                                                  start[domain][label]:
				                                                  end[domain][label]]))
		return domain_data
	
	def build_request_slot_dataset(self):
		def extract_one_round(one_round, request_slot_data):
			user_nl = one_round["user_nl_words"]
			sys_act = one_round["sys_act"]
			for domain in domain_set:
				if one_round['domain_label'] == domain:
					for slot in request_slots[domain]:
						mentioned_flag = False
						for user_act in ['request', 'confirm']:
							if user_act in one_round['user_act']:
								if domain in one_round['user_act'][user_act]:
									if slot in one_round['user_act'][user_act][domain]:
										data = [{'user_nl': user_nl, 'sys_act': sys_act, 'domain': domain, 'slot': slot,
										         'label': "MENTIONED"}]
										request_slot_data[domain][slot]["MENTIONED"].extend(copy.deepcopy(data))
										mentioned_flag = True
										break
						if not mentioned_flag:
							data = [{'user_nl': user_nl, 'sys_act': sys_act, 'domain': domain, 'slot': slot,
							         'label': "NOT_MENTIONED"}]
							request_slot_data[domain][slot]["NOT_MENTIONED"].extend(copy.deepcopy(data))
			return request_slot_data
			
		request_slot_data = {}  # 按照["MENTIONED", "NOT_MENTIONED"] label的不同来对x进行分类
		for domain in domain_set:
			request_slot_data[domain] = {}
			for slot in request_slots[domain]:
				request_slot_data[domain][slot] = {}
				for label in request_slot_label:
					request_slot_data[domain][slot][label] = []
		for one_round in self._rounds_dialog:
			request_slot_data = extract_one_round(one_round, request_slot_data)
		return request_slot_data
	
	def next_batch_request_slot(self, batchsize, domain, slot):
		request_slot_data = []
		start = copy.deepcopy(self._request_slot_data_index)
		end = copy.deepcopy(self._request_slot_data_index)
		for label in request_slot_label:
			num_total_data = len(self._request_slot_data[domain][slot][label])
			if start[domain][slot][label] + batchsize[label] >= num_total_data:
				# Get the rest examples in this epoch
				num_rest_data = num_total_data - start[domain][slot][label]
				data_rest_part = copy.deepcopy(self._request_slot_data[domain][slot][label][
				                               start[domain][slot][label]:
				                               num_total_data])
				# Shuffle the data
				random.shuffle(self._request_slot_data[domain][slot][label])
				# Start next epoch
				start[domain][slot][label] = 0
				self._request_slot_data_index[domain][slot][label] = batchsize[label] - num_rest_data
				end[domain][slot][label] = self._request_slot_data_index[domain][slot][label]
				data_new_part = copy.deepcopy(self._request_slot_data[domain][slot][label][
				                              start[domain][slot][label]:
				                              end[domain][slot][label]])
				request_slot_data.extend(data_rest_part + data_new_part)
			else:
				self._request_slot_data_index[domain][slot][label] += batchsize[label]
				end[domain][slot][label] = self._request_slot_data_index[domain][slot][label]
				request_slot_data.extend(
					copy.deepcopy(self._request_slot_data[domain][slot][label][
					              start[domain][slot][label]:
					              end[domain][slot][label]]))
		return request_slot_data
	
	def build_inform_slot_dataset(self):
		def extract_one_round(one_round, inform_slot_data):
			user_nl = one_round["user_nl_words"]
			sys_act = one_round["sys_act"]
			one_round['inform_slot_label'] = []
			for domain in ['电影', '音乐']:
				if one_round['domain_label'] == domain:
					for slot in inform_slots[domain]:
						mentioned_flag = False
						if 'inform' in one_round['user_act']:
							if domain in one_round['user_act']['inform']:
								if slot in one_round['user_act']['inform'][domain]:
									if one_round['user_act']['inform'][domain][slot]:
										data = [{'user_nl': user_nl, 'sys_act': sys_act, 'domain': domain, 'slot': slot,
										         'label': "MENTIONED"}]
										one_round['inform_slot_label'].append(slot)
										inform_slot_data[domain][slot]["MENTIONED"].extend(copy.deepcopy(data))
									else:
										data = [{'user_nl': user_nl, 'sys_act': sys_act, 'domain': domain, 'slot': slot,
										         'label': "DONT_CARE"}]
										inform_slot_data[domain][slot]["DONT_CARE"].extend(copy.deepcopy(data))
									mentioned_flag = True
						if not mentioned_flag:
							data = [{'user_nl': user_nl, 'sys_act': sys_act, 'domain': domain, 'slot': slot,
							         'label': "NOT_MENTIONED"}]
							inform_slot_data[domain][slot]["NOT_MENTIONED"].extend(copy.deepcopy(data))
			return inform_slot_data
		
		inform_slot_data = {}  # 按照["MENTIONED", "NOT_MENTIONED", "DONT_CARE"] label的不同来对x进行分类
		for domain in ['电影', '音乐']:
			inform_slot_data[domain] = {}
			for slot in inform_slots[domain]:
				inform_slot_data[domain][slot] = {}
				for label in inform_slot_label:
					inform_slot_data[domain][slot][label] = []
		for one_round in self._rounds_dialog:
			inform_slot_data = extract_one_round(one_round, inform_slot_data)
		return inform_slot_data
	
	def next_batch_inform_slot(self, batchsize, domain, slot):
		inform_slot_data = []
		start = copy.deepcopy(self._inform_slot_data_index)
		end = copy.deepcopy(self._inform_slot_data_index)
		for label in inform_slot_label:
			num_total_data = len(self._inform_slot_data[domain][slot][label])
			if start[domain][slot][label] + batchsize[label] >= num_total_data:
				# Get the rest examples in this epoch
				num_rest_data = num_total_data - start[domain][slot][label]
				data_rest_part = copy.deepcopy(self._inform_slot_data[domain][slot][label][
				                               start[domain][slot][label]:
				                               num_total_data])
				# Shuffle the data
				random.shuffle(self._inform_slot_data[domain][slot][label])
				# Start next epoch
				start[domain][slot][label] = 0
				self._inform_slot_data_index[domain][slot][label] = batchsize[label] - num_rest_data
				end[domain][slot][label] = self._inform_slot_data_index[domain][slot][label]
				data_new_part = copy.deepcopy(self._inform_slot_data[domain][slot][label][
				                              start[domain][slot][label]:
				                              end[domain][slot][label]])
				inform_slot_data.extend(data_rest_part + data_new_part)
			else:
				self._inform_slot_data_index[domain][slot][label] += batchsize[label]
				end[domain][slot][label] = self._inform_slot_data_index[domain][slot][label]
				inform_slot_data.extend(
					copy.deepcopy(self._inform_slot_data[domain][slot][label][
					              start[domain][slot][label]:
					              end[domain][slot][label]]))
		return inform_slot_data

	def build_value_dataset(self):
		def extract_one_round(one_round, value_data):
			user_nl = one_round["user_nl_words"]
			sys_act = one_round["sys_act"]
			for domain in ['电影', '音乐']:
				if one_round['domain_label'] == domain:
					for slot in one_round['inform_slot_label']:
						for value in one_round['user_act']['inform'][domain][slot]:
							if one_round['user_act']['inform'][domain][slot][value] == 'LIKE':
								data = [{'user_nl': user_nl, 'domain': domain, 'slot': slot, 'value': value,
								         'label': "LIKE"}]
								value_data[domain][slot]["LIKE"].extend(copy.deepcopy(data))
							elif one_round['user_act']['inform'][domain][slot][value] == 'DISLIKE':
								data = [{'user_nl': user_nl, 'domain': domain, 'slot': slot, 'value': value,
								         'label': "DISLIKE"}]
								value_data[domain][slot]["DISLIKE"].extend(copy.deepcopy(data))
			return value_data
		
		value_data = {}  # 按照["MENTIONED", "NOT_MENTIONED", "DONT_CARE"] label的不同来对x进行分类
		for domain in ['电影', '音乐']:
			value_data[domain] = {}
			for slot in inform_slots[domain]:
				value_data[domain][slot] = {}
				for label in value_label:
					value_data[domain][slot][label] = []
		for one_round in self._rounds_dialog:
			value_data = extract_one_round(one_round, value_data)
		return value_data
	
	def next_batch_value(self, batchsize, domain, slot):
		value_data = []
		start = copy.deepcopy(self._value_data_index)
		end = copy.deepcopy(self._value_data_index)
		for label in value_label:
			num_total_data = len(self._value_data[domain][slot][label])
			if start[domain][slot][label] + batchsize[label] >= num_total_data:
				# Get the rest examples in this epoch
				num_rest_data = num_total_data - start[domain][slot][label]
				data_rest_part = copy.deepcopy(self._value_data[domain][slot][label][
				                               start[domain][slot][label]:
				                               num_total_data])
				# Shuffle the data
				random.shuffle(self._value_data[domain][slot][label])
				# Start next epoch
				start[domain][slot][label] = 0
				self._value_data_index[domain][slot][label] = batchsize[label] - num_rest_data
				end[domain][slot][label] = self._value_data_index[domain][slot][label]
				data_new_part = copy.deepcopy(self._value_data[domain][slot][label][
				                              start[domain][slot][label]:
				                              end[domain][slot][label]])
				value_data.extend(data_rest_part + data_new_part)
			else:
				self._value_data_index[domain][slot][label] += batchsize[label]
				end[domain][slot][label] = self._value_data_index[domain][slot][label]
				value_data.extend(
					copy.deepcopy(self._value_data[domain][slot][label][
					              start[domain][slot][label]:
					              end[domain][slot][label]]))
		return value_data
	
	# def build_confirm_slot_dataset(self):
	# 	def extract_one_round(one_round, confirm_slot_data):
	# 		user_nl = one_round["user_nl_words"]
	# 		sys_act = one_round["sys_act"]
	# 		for domain in ['电影', '音乐']:
	# 			if one_round['domain_label'] == domain:
	# 				for slot in confirm_slots[domain]:
	# 					mentioned_flag = False
	# 					if 'confirm' in one_round['user_act']:
	# 						if domain in one_round['user_act']['confirm']:
	# 							if slot in one_round['user_act']['confirm'][domain]:
	# 								data = [{'user_nl': user_nl, 'sys_act': sys_act, 'domain': domain, 'slot': slot,
	# 								         'label': "MENTIONED"}]
	# 								confirm_slot_data[domain][slot]["MENTIONED"].extend(copy.deepcopy(data))
	# 								mentioned_flag = True
	# 					if not mentioned_flag:
	# 						data = [{'user_nl': user_nl, 'sys_act': sys_act, 'domain': domain, 'slot': slot,
	# 						         'label': "NOT_MENTIONED"}]
	# 						confirm_slot_data[domain][slot]["NOT_MENTIONED"].extend(copy.deepcopy(data))
	# 		return confirm_slot_data
	#
	# 	confirm_slot_data = {}  # 按照["MENTIONED", "NOT_MENTIONED"] label的不同来对x进行分类
	# 	for domain in ['电影', '音乐']:
	# 		confirm_slot_data[domain] = {}
	# 		for slot in confirm_slots[domain]:
	# 			confirm_slot_data[domain][slot] = {}
	# 			for label in confirm_slot_label:
	# 				confirm_slot_data[domain][slot][label] = []
	# 	for one_round in self._rounds_dialog:
	# 		confirm_slot_data = extract_one_round(one_round, confirm_slot_data)
	# 	return confirm_slot_data
	#
	# def next_batch_confirm_slot(self, batchsize, domain, slot):
	# 	confirm_slot_data = []
	# 	start = copy.deepcopy(self._confirm_slot_data_index)
	# 	end = copy.deepcopy(self._confirm_slot_data_index)
	# 	for label in confirm_slot_label:
	# 		num_total_data = len(self._confirm_slot_data[domain][slot][label])
	# 		if start[domain][slot][label] + batchsize[label] >= num_total_data:
	# 			# Get the rest examples in this epoch
	# 			num_rest_data = num_total_data - start[domain][slot][label]
	# 			data_rest_part = copy.deepcopy(self._confirm_slot_data[domain][slot][label][
	# 			                               start[domain][slot][label]:
	# 			                               num_total_data])
	# 			# Shuffle the data
	# 			random.shuffle(self._confirm_slot_data[domain][slot][label])
	# 			# Start next epoch
	# 			start[domain][slot][label] = 0
	# 			self._confirm_slot_data_index[domain][slot][label] = batchsize[label] - num_rest_data
	# 			end[domain][slot][label] = self._confirm_slot_data_index[domain][slot][label]
	# 			data_new_part = copy.deepcopy(self._confirm_slot_data[domain][slot][label][
	# 			                              start[domain][slot][label]:
	# 			                              end[domain][slot][label]])
	# 			confirm_slot_data.extend(data_rest_part + data_new_part)
	# 		else:
	# 			self._confirm_slot_data_index[domain][slot][label] += batchsize[label]
	# 			end[domain][slot][label] = self._confirm_slot_data_index[domain][slot][label]
	# 			confirm_slot_data.extend(
	# 				copy.deepcopy(self._confirm_slot_data[domain][slot][label][
	# 				              start[domain][slot][label]:
	# 				              end[domain][slot][label]]))
	# 	return confirm_slot_data
	
	def build_act_dataset(self):
		def extract_one_round(one_round, act_data):
			user_nl = one_round["user_nl_words"]
			sys_act = one_round["sys_act"]
			for act in user_acts:
				if act in one_round["user_act"]:
					data = [{"user_nl": user_nl, "sys_act": sys_act, "user_act": act, "label": "MENTIONED"}]
					act_data[act]["MENTIONED"].extend(copy.deepcopy(data))
				else:
					data = [{"user_nl": user_nl, "sys_act": sys_act, "user_act": act, "label": "NOT_MENTIONED"}]
					act_data[act]["NOT_MENTIONED"].extend(copy.deepcopy(data))
			return act_data
			
		act_data = {}
		# init
		for act in user_acts:
			act_data[act] = {}
			for label in act_label:
				act_data[act][label] = []
		# build dataset
		for one_round in self._rounds_dialog:
			act_data = extract_one_round(one_round, act_data)
		return act_data
	
	def next_batch_act(self, batchsize,act):
		act_data = []
		start = copy.deepcopy(self._act_data_index)
		end = copy.deepcopy(self._act_data_index)
		for label in act_label:
			num_total_data = len(self._act_data[act][label])
			if start[act][label] + batchsize[label] >= num_total_data:
				# Get the rest examples in this epoch
				num_rest_data = num_total_data - start[act][label]
				data_rest_part = copy.deepcopy(self._act_data[act][label][
				                               start[act][label]:
				                               num_total_data])
				# Shuffle the data
				random.shuffle(self._act_data[act][label])
				# Start next epoch
				start[act][label] = 0
				self._act_data_index[act][label] = batchsize[label] - num_rest_data
				end[act][label] = self._act_data_index[act][label]
				data_new_part = copy.deepcopy(self._act_data[act][label][
				                              start[act][label]:
				                              end[act][label]])
				act_data.extend(data_rest_part + data_new_part)
			else:
				self._act_data_index[act][label] += batchsize[label]
				end[act][label] = self._act_data_index[act][label]
				act_data.extend(copy.deepcopy(self._act_data[act][label][
				                                 start[act][label]:
				                                 end[act][label]]))
		return act_data
	
class DataSets(object):
	pass


def read_data_sets():
	data_sets = DataSets()
	random.shuffle(rounds_dialog)

	with open('../../data/train_valid_set.json', 'w', encoding='utf-8') as f:
		json.dump(rounds_dialog[:train_size], f, ensure_ascii=False, indent=2)
	with open('../../data/test_set.json', 'w', encoding='utf-8') as f:
		json.dump(rounds_dialog[train_size:], f, ensure_ascii=False, indent=2)
	
	with open('../../data/train_valid_set.json', 'r', encoding='utf-8') as f:
		train_valid_set = json.load(f)
	data_sets.train = DataSet(train_valid_set)
	data_sets.valid = DataSet(train_valid_set)
	with open('../../data/test_set.json', 'r', encoding='utf-8') as f:
		test_set = json.load(f)
	
	return data_sets


if __name__ == '__main__':
	pass
