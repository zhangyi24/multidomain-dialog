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
	with open("kb_movie_actor.json", 'r', encoding="utf-8") as f:
		kb_movie_actor = json.load(f)
	actors = []
	for actor in kb_movie_actor:
		if len(kb_movie_actor[actor]) >= 2:
			actors.append(actor)
			print(actor)

	with open("actors.json", 'w', encoding="utf-8") as f:
		json.dump(actors, f, ensure_ascii=False, indent=2)
	print(len(actors))
