#!/usr/bin/env python3
# coding=utf-8
"""
给定系统动作，输出自然语言
"""
import random
import time
import requests

def list2str(items):
	for forbidden in ['今年', '去年', '前年']:
		if forbidden in items:
			items.remove(forbidden)
	return '、'.join(items)


class Weather(object):
	def __init__(self, city='北京', time='今天'):
		self._city = city
		self._time = time
		self._extensions = 'all' if time else 'base'
		self._params = {'key': '307728c37b380a75a9b96c1848eb1179', 'city': self._city,
		                'extensions': self._extensions}
		self._url = 'http://restapi.amap.com/v3/weather/weatherInfo'
		self._r = requests.get(self._url, params=self._params)
		self._r.encoding = 'utf-8'
		self._weather_info = self._r.json()
	
	def show(self):
		nl = ''
		if self._time == '今天':
			# 今天
			extensions = 'all'
			params = {'key': '307728c37b380a75a9b96c1848eb1179', 'city': self._city,
			          'extensions': extensions}
			url = 'http://restapi.amap.com/v3/weather/weatherInfo'
			r = requests.get(url, params=params)
			r.encoding = 'utf-8'
			weather_info = self._r.json()
			self._city = weather_info['forecasts'][0]['city']
			casts = self._weather_info['forecasts'][0]['casts']
			cast = casts[0]
			nl += (self._city + '今天天气状况：')
			weather = []
			weather.append(cast["date"])
			if cast["dayweather"] == cast["nightweather"]:
				weather.append(cast["dayweather"])
			else:
				weather.append(cast["dayweather"] + '转' + cast["nightweather"])
			weather.append('%s℃~%s℃' % (cast["nighttemp"], cast["daytemp"]))
			nl += ' '.join(weather)
			nl += '\n'
			# 实时
			extensions = 'base'
			params = {'key': '307728c37b380a75a9b96c1848eb1179', 'city': self._city,
			          'extensions': extensions}
			url = 'http://restapi.amap.com/v3/weather/weatherInfo'
			r = requests.get(url, params=params)
			r.encoding = 'utf-8'
			weather_info = r.json()
			self._city = weather_info['lives'][0]['city']
			live = weather_info['lives'][0]
			nl += (self._city + '实时天气：%s\n' % live["weather"])
			nl += ('温度：%s℃\t湿度：%s\n' % (live["temperature"], live["humidity"]))
			nl += ('风向：%s\t风力：%s级\n' % (live["winddirection"], live["windpower"]))
			nl += ('天气更新时间：%s\n' % live["reporttime"])
		elif self._time == '明天':
			extensions = 'all'
			params = {'key': '307728c37b380a75a9b96c1848eb1179', 'city': self._city,
			          'extensions': extensions}
			url = 'http://restapi.amap.com/v3/weather/weatherInfo'
			r = requests.get(url, params=params)
			r.encoding = 'utf-8'
			weather_info = self._r.json()
			self._city = weather_info['forecasts'][0]['city']
			casts = self._weather_info['forecasts'][0]['casts']
			cast = casts[1]
			nl += (self._city + '明天天气状况：')
			weather = []
			weather.append(cast["date"])
			if cast["dayweather"] == cast["nightweather"]:
				weather.append(cast["dayweather"])
			else:
				weather.append(cast["dayweather"] + '转' + cast["nightweather"])
			weather.append('%s℃~%s℃' % (cast["nighttemp"], cast["daytemp"]))
			nl += (' '.join(weather))
		elif self._time == '未来几天':
			extensions = 'all'
			params = {'key': '307728c37b380a75a9b96c1848eb1179', 'city': self._city,
			          'extensions': extensions}
			url = 'http://restapi.amap.com/v3/weather/weatherInfo'
			r = requests.get(url, params=params)
			r.encoding = 'utf-8'
			weather_info = self._r.json()
			self._city = weather_info['forecasts'][0]['city']
			casts = self._weather_info['forecasts'][0]['casts']
			nl += (self._city + '未来几天天气状况：\n')
			casts = casts[1:]
			for cast in casts:
				weather = []
				weather.append(cast["date"])
				if cast["dayweather"] == cast["nightweather"]:
					weather.append(cast["dayweather"])
				else:
					weather.append(cast["dayweather"] + '转' + cast["nightweather"])
				weather.append('%s℃~%s℃' % (cast["nighttemp"], cast["daytemp"]))
				nl += (' '.join(weather) + '\n')
		return nl
		


def rule_based_NLG(sys_act, KB_pointer, knowledge_base):
	def nlg_movie(act_type, content):
		nl = ""
		if act_type == "request":
			slot = content
			if slot == "主演":
				nl += "您喜欢看由谁主演的电影？" if random.random() < 0.5 else "您想看由谁主演的电影？"
			elif slot == "地区":
				nl += "您喜欢看哪国（地区）的电影？" if random.random() < 0.5 else "您想看哪国（地区）的电影？"
			elif slot == "导演":
				nl += "您喜欢看由谁导演的电影？" if random.random() < 0.5 else "您想看由谁导演的电影？"
			elif slot == "年代":
				nl += "您喜欢看哪个年代的电影？" if random.random() < 0.5 else "您想看哪个年代的电影？"
			elif slot == "类型":
				nl += "您喜欢看什么类型的电影？" if random.random() < 0.5 else "您想看什么类型的电影？"
			elif slot == "资费":
				nl += "您想看免费还是付费的电影？"
		elif act_type == "inform":
			movie = knowledge_base['电影']['片名'][KB_pointer['电影']['片名']][0]
			nl += "《%s》" % movie['片名']
			for slot in content:
				if slot == "主演":
					nl += "主演有%s，" % list2str(movie[slot])
				elif slot == "导演":
					nl += "导演是%s，" % movie[slot]
				elif slot == "类型":
					nl += "类型是%s，" % list2str(movie[slot])
				elif slot == "地区":
					nl += "是%s电影，" % list2str(movie[slot])
				elif slot == "评分":
					nl += "评分为%s，" % movie[slot]
				elif slot == "年代":
					nl += "是%s年代的，" % list2str(movie[slot])
				elif slot == "上映日期":
					nl += "是%s年上映的，" % movie[slot]
				elif slot == "资费":
					nl += "是%s电影，" % movie[slot]
				elif slot == "片长":
					nl += "片长为%s分钟，" % movie[slot]
				elif slot == "简介":
					nl += "简介为：%s" % movie[slot]
			nl = nl.rstrip('，').rstrip('。')
		elif act_type == "confirm":
			for slot in content:
				if slot == "主演":
					if random.random() < 0.5:
						nl += "您" + random.choice(["喜欢", "想"]) + "看%s主演的电影么？" % content[slot]
					else:
						nl += "来部%s主演的电影怎么样？" % content[slot]
				elif slot == "地区":
					if random.random() < 0.5:
						nl += "您" + random.choice(["喜欢", "想"]) + "看%s的电影么？" % content[slot]
					else:
						nl += "来部%s的电影怎么样？" % content[slot]
				elif slot == "导演":
					if random.random() < 0.5:
						nl += "您" + random.choice(["喜欢", "想"]) + "看%s导演的电影么？" % content[slot]
					else:
						nl += "来部%s导演的电影怎么样？" % content[slot]
				elif slot == "类型":
					if random.random() < 0.5:
						nl += "您" + random.choice(["喜欢", "想"]) + "看%s这种类型的电影么？" % content[slot]
					else:
						nl += "来部%s这种类型的电影怎么样？" % content[slot]
		elif act_type == "inform_no_match":
			movies = content['片名']
			num_movies = len(movies)
			if num_movies == 0:
				nl += "对不起没有找到符合您要求的电影。"
		elif act_type == "inform_one_match":
			movies = content['片名']
			num_movies = len(movies)
			if num_movies == 1:
				nl += "为您找到一部影片：%s。准备为您播放。请问您还想知道关于该电影的哪些信息？" % list2str(movies)
		elif act_type == "inform_some_match":
			movies = content["片名"]
			num_movies = len(movies)
			if num_movies == 2:
				nl += "为您找到两部影片：%s。您想先看哪一部呢？" % list2str(movies)
			elif num_movies >= 3:
				nl += "找到多部影片，为您推荐其中三部：%s。您想先看哪一部呢？" % list2str(movies[0:3])
		return nl
	
	def nlg_music(act_type, content):
		nl = ""
		if act_type == "request":
			slot = content
			if slot == "歌手":
				nl += "您喜欢听谁唱的歌？" if random.random() < 0.5 else "您想听谁唱的歌？"
			elif slot == "曲风":
				nl += "您喜欢听什么类型或风格的歌曲？" if random.random() < 0.5 else "您想听什么类型或风格的歌曲？"
			elif slot == "年代":
				nl += "您喜欢听什么年代的歌曲？" if random.random() < 0.5 else "您想听什么年代的歌曲？"
			elif slot == "专辑":
				nl += "您喜欢哪张专辑？" if random.random() < 0.5 else "您想听哪张专辑？"
		elif act_type == "inform":
			music = knowledge_base['音乐']['歌名'][KB_pointer['音乐']['歌名']][0]
			nl += "《%s》" % music['歌名']
			for slot in content:
				if slot == "歌手":
					nl += "歌手是%s，" % music[slot]
				elif slot == "专辑":
					nl += "专辑是%s，" % "《%s》" % music[slot]
				elif slot == "曲风":
					nl += "类型是%s，" % list2str(music[slot])
				elif slot == "年代":
					nl += "是%s年代的，" % list2str(music[slot])
				elif slot == "发行日期":
					nl += "发行日期是%s，" % music[slot]
				elif slot == "时长":
					nl += "歌曲时长为%s，" % music[slot]
			nl = nl.rstrip('，')
			nl += '。'
		elif act_type == "confirm":
			for slot in content:
				if slot == "歌手":
					if random.random() < 0.5:
						nl += "您" + random.choice(["喜欢", "想"]) + "听%s唱的歌么？" % content[slot]
					else:
						nl += "来首%s的歌怎么样？" % content[slot]
				elif slot == "曲风":
					if content[slot] in ["旅行", "夜晚", "运动", "学习", "放松", "快乐", "感动", "孤独"]:
						nl += "来首%s的时候听的歌怎么样？" % content[slot]
					elif content[slot] in ["安静", "怀旧", "治愈", "粤语", "浪漫", "伤感"]:
						nl += "您想听%s的歌么？" % content[slot]
					elif content[slot] in ["说唱", "民谣", "摇滚", "古风"]:
						nl += "您喜不喜欢听%s？" % content[slot]
					elif content[slot] in ["流行", "经典"]:
						nl += "放一首%s的歌好不好？" % content[slot]
		elif act_type == "inform_no_match":
			songs = content["歌名"]
			num_songs = len(songs)
			if num_songs == 0:
				nl += "对不起没有找到符合您要求的歌曲。"
		elif act_type == "inform_one_match":
			songs = content["歌名"]
			num_songs = len(songs)
			if num_songs == 1:
				nl += "为您找到一首歌曲：%s。准备为您播放。请问您还想知道关于该歌曲的哪些信息？" % list2str(songs)
		elif act_type == "inform_some_match":
			songs = content["歌名"]
			num_songs = len(songs)
			if num_songs == 2:
				nl += "为您找到两首歌曲：%s。您想先听哪一首呢？" % list2str(songs)
			elif num_songs >= 3:
				nl += "找到多首歌曲，为您推荐其中三首：%s。您想先听哪一首呢？" % list2str(songs)
		return nl
	
	def nlg_weather(act_type, content):
		nl = ""
		if act_type == "inform":
			for slot in content:
				if slot == "天气":
					city = KB_pointer['天气']['城市']
					time = KB_pointer['天气']['时间']
					try:
						weather = Weather(city, time)
						nl += weather.show()
					except:
						nl += "当前网络不可用，请检查网络设置"
		return nl
	
	def nlg_time(act_type, content):
		nl = ""
		if act_type == "inform":
			struct_time = time.localtime(time.time())
			year, month, day = struct_time[0], struct_time[1], struct_time[2]
			for slot in content:
				if slot == "日期":
					nl += "今天是%d年%d月%d日。" % (year, month, day)
				elif slot == "星期":
					weekdays = ["一", "二", "三", "四", "五", "六", "日"]
					weekday = weekdays[struct_time[6]]
					nl += "今天是星期%s。" % weekday
				elif slot == "时刻":
					nl += "现在是北京时间" + time.strftime('%X', time.localtime(time.time())) + "。"
		return nl
	
	nl = ""
	nlg_fun = {"电影": nlg_movie, "音乐": nlg_music, "天气": nlg_weather, "时间": nlg_time}
	sys_act_type = list(sys_act.keys())[0]
	content = sys_act[sys_act_type]
	if sys_act_type == "hello":
		nl += "欢迎使用清华大学电子系多领域信息问询系统，您可以对电影、音乐、时间、天气信息进行查询。"
	elif sys_act_type == "repeat":
		nl += "抱歉，能再重述一遍么？"
	else:
		for domain in content:
			nl += nlg_fun[domain](sys_act_type, content[domain])
	return nl


if __name__ == '__main__':
	pass
