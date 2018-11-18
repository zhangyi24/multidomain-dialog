#!/usr/bin/env python3
# coding=utf-8

import json
import random
from itertools import combinations

# Load files
with open("../../definition/OTLG.json", "r", encoding="utf-8") as f:
	OTLG = json.load(f)
with open("kb_movie.json", "r", encoding="utf-8") as f:
	kb_movie = json.load(f)
with open("kb_music.json", "r", encoding="utf-8") as f:
	kb_music = json.load(f)
knowledgebase = {"电影": kb_movie, "音乐": kb_music}

filename = {
	"电影": {
		"片名": "kb_movie_name.json",
		"主演": "kb_movie_actor.json",
		"导演": "kb_movie_director.json",
		"类型": "kb_movie_genre.json",
		"地区": "kb_movie_area.json",
		"年代": "kb_movie_era.json",
		"资费": "kb_movie_payment.json"
	},
	"音乐": {
		"歌名": "kb_music_name.json",
		"歌手": "kb_music_artist.json",
		"专辑": "kb_music_album.json",
		"曲风": "kb_music_genre.json",
		"年代": "kb_music_era.json",
	}
}

if __name__ == "__main__":
	for domain in ["电影", "音乐"]:
		for slot in OTLG[domain]["informable"]:
			kb = {}
			for value in OTLG[domain]["informable"][slot]:
				kb[value] = []
				for item in knowledgebase[domain]:
					if type(item[slot]) == list:
						if value in item[slot]:
							kb[value].append(item)
					else:
						if value == item[slot]:
							kb[value].append(item)
			with open(filename[domain][slot], 'w', encoding="utf-8") as f:
				json.dump(kb, f, ensure_ascii=False, indent=2)

