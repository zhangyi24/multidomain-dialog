#!/usr/bin/env python3
# coding=utf-8

import json
import random
import xlwt

NUM_ROUNDS = 50000

# Load files
with open("../definition/OTLG.json", "r", encoding="utf-8") as f:
	OTLG = json.load(f)
with open("knowledge_base/kb_movie.json", "r", encoding="utf-8") as f:
	kb_movie = json.load(f)
with open("knowledge_base/kb_music.json", "r", encoding="utf-8") as f:
	kb_music = json.load(f)
database = {"电影": kb_movie, "音乐": kb_music}

# Act definition
# System
sys_act_set = [
	"hello",
	"request",
	"inform_no_match",
	"inform_one_match",
	"inform_some_match",
	"inform",
	"confirm"
	# "play"    后面跟的用户回答和"hello"后面跟的类似
]
sys_act_prop = {
	"hello": 0.1,
	"request": 0.35,
	"inform_no_match": 0.05,
	"inform_one_match": 0.05,
	"inform_some_match": 0.05,
	"inform": 0.35,
	"confirm": 0.05
}
sys_domain_prop = {
	"hello": None,
	"request": {"电影": 0.6, "音乐": 0.4},
	"inform_no_match": {"电影": 0.6, "音乐": 0.4},
	"inform_one_match": {"电影": 0.6, "音乐": 0.4},
	"inform_some_match": {"电影": 0.6, "音乐": 0.4},
	"inform": {"电影": 0.4, "音乐": 0.3, "天气": 0.2, "时间": 0.1},
	"confirm": {"电影": 0.6, "音乐": 0.4}
}
sys_slot_prop = {
	"hello": None,
	"request": {
		"电影": {"主演": 0.35, "导演": 0.25, "类型": 0.15, "地区": 0.15, "年代": 0.05, "资费": 0.05},
		"音乐": {"歌手": 0.4, "曲风": 0.25, "年代": 0.2, "专辑": 0.15}
	},
	"inform_no_match": {"电影": {"片名": 1.0}, "音乐": {"歌名": 1.0}},
	"inform_one_match": {"电影": {"片名": 1.0}, "音乐": {"歌名": 1.0}},
	"inform_some_match": {"电影": {"片名": 1.0}, "音乐": {"歌名": 1.0}},
	"inform": {
		"电影": {"主演": 0.3, "导演": 0.2, "类型": 0.1, "评分": 0.1, "简介": 0.1, "地区": 0.05,
		       "资费": 0.05, "上映日期": 0.05, "片长": 0.03, "年代": 0.02},
		"音乐": {"歌手": 0.35, "专辑": 0.25, "曲风": 0.25, "年代": 0.1, "发行日期": 0.03, "时长": 0.02},
		"天气": {"天气": 0.6, "气温": 0.3, "雨": 0.1},
		"时间": {"时刻": 0.5, "日期": 0.25, "星期": 0.25}
	},
	"confirm": {
		"电影": {"主演": 0.4, "导演": 0.25, "类型": 0.2, "地区": 0.15},
		"音乐": {"歌手": 0.65, "曲风": 0.35}
	}
}
sys_kb_pointer_movie = 0
sys_kb_pointer_music = 0
# User
user_act_set = [
	"inform_slot",
	"inform_major_key",
	"request",
	"confirm",
	"implicit_request",
	"first",
	"second",
	"third",
	"last",
	"other",
	"affirm",
	"deny"
]
user_policy_set = [
	"inform_slot",
	"inform_major_key",
	"inform_major_key+request",
	"inform_major_key+confirm",
	"request",
	"confirm",
	"implicit_request",
	"first",
	"first+request",
	"first+confirm",
	"second",
	"second+request",
	"second+confirm",
	"third",
	"third+request",
	"third+confirm",
	"last",
	"last+request",
	"last+confirm",
	"other",
	"affirm",
	"deny"
]
user_mode_prop = {
	"hello": {
		"usual": 1.0
	},
	"request": {
		"usual": 0.85,
		"casual": 0.03,
		"switch_domain": 0.12
	},
	"inform_no_match": {
		"usual": 0.7,
		"switch_domain": 0.3
	},
	"inform_one_match": {
		"usual": 0.85,
		"casual": 0.03,
		"switch_domain": 0.12
	},
	"inform_some_match": {
		"usual": 0.85,
		"casual": 0.03,
		"switch_domain": 0.12
	},
	"inform": {
		"usual": 0.65,
		"casual": 0.02,
		"switch_domain": 0.33
	},
	"confirm": {
		"usual": 0.8,
		"casual": 0.05,
		"switch_domain": 0.15
	}
}
user_domain_prop = {
	"电影": 0.5, "音乐": 0.3, "天气": 0.12, "时间": 0.08
}
user_policy_prop = {
	"hello": {
		"usual": {
			"电影": {
				"inform_slot": 0.5,
				"inform_major_key": 0.2,
				"inform_major_key+request": 0.25,
				"inform_major_key+confirm": 0.05,
			},
			"音乐": {
				"inform_slot": 0.5,
				"inform_major_key": 0.2,
				"inform_major_key+request": 0.25,
				"inform_major_key+confirm": 0.05,
			},
			"天气": {
				"inform_major_key+request": 1.0,
			},
			"时间": {
				"request": 1.0
			}
		}
	},
	"request": {
		"usual": {
			"电影": {
				"inform_slot": 1.0
			},
			"音乐": {
				"inform_slot": 1.0
			}
		},
		"casual": {
			"电影": {
				"inform_slot": 0.55,
				"inform_major_key": 0.1,
				"inform_major_key+request": 0.2,
				"inform_major_key+confirm": 0.15
			},
			"音乐": {
				"inform_slot": 0.55,
				"inform_major_key": 0.1,
				"inform_major_key+request": 0.2,
				"inform_major_key+confirm": 0.15
			}
		},
		"switch_domain": {
			"电影": {
				"inform_slot": 0.5,
				"inform_major_key": 0.2,
				"inform_major_key+request": 0.25,
				"inform_major_key+confirm": 0.05,
			},
			"音乐": {
				"inform_slot": 0.5,
				"inform_major_key": 0.2,
				"inform_major_key+request": 0.25,
				"inform_major_key+confirm": 0.05,
			},
			"天气": {
				"inform_major_key+request": 1.0,
			},
			"时间": {
				"request": 1.0
			}
		}
	},
	"inform_no_match": {
		"usual": {
			"电影": {
				"inform_slot": 0.55,
				"inform_major_key": 0.1,
				"inform_major_key+request": 0.2,
				"inform_major_key+confirm": 0.15
			},
			"音乐": {
				"inform_slot": 0.55,
				"inform_major_key": 0.1,
				"inform_major_key+request": 0.2,
				"inform_major_key+confirm": 0.15
			}
		},
		"switch_domain": {
			"电影": {
				"inform_slot": 0.5,
				"inform_major_key": 0.2,
				"inform_major_key+request": 0.25,
				"inform_major_key+confirm": 0.05,
			},
			"音乐": {
				"inform_slot": 0.5,
				"inform_major_key": 0.2,
				"inform_major_key+request": 0.25,
				"inform_major_key+confirm": 0.05,
			},
			"天气": {
				"inform_major_key+request": 1.0,
			},
			"时间": {
				"request": 1.0
			}
		}
	},
	"inform_one_match": {
		"usual": {
			"电影": {
				"request": 0.6,
				"inform_major_key+request": 0.3,
				"inform_major_key+confirm": 0.1
			},
			"音乐": {
				"request": 0.6,
				"inform_major_key+request": 0.3,
				"inform_major_key+confirm": 0.1
			}
		},
		"casual": {
			"电影": {
				"inform_slot": 0.55,
				"inform_major_key": 0.1,
				"inform_major_key+request": 0.2,
				"inform_major_key+confirm": 0.15
			},
			"音乐": {
				"inform_slot": 0.55,
				"inform_major_key": 0.1,
				"inform_major_key+request": 0.2,
				"inform_major_key+confirm": 0.15
			}
		},
		"switch_domain": {
			"电影": {
				"inform_slot": 0.5,
				"inform_major_key": 0.2,
				"inform_major_key+request": 0.25,
				"inform_major_key+confirm": 0.05,
			},
			"音乐": {
				"inform_slot": 0.5,
				"inform_major_key": 0.2,
				"inform_major_key+request": 0.25,
				"inform_major_key+confirm": 0.05,
			},
			"天气": {
				"inform_major_key+request": 1.0,
			},
			"时间": {
				"request": 1.0
			}
		}
	},
	"inform_some_match": {
		"usual": {
			"电影": {
				"inform_major_key": 0.2,
				"inform_major_key+request": 0.08,
				"inform_major_key+confirm": 0.02,
				"first": 0.05,
				"first+request": 0.06,
				"first+confirm": 0.04,
				"second": 0.05,
				"second+request": 0.06,
				"second+confirm": 0.04,
				"third": 0.05,
				"third+request": 0.06,
				"third+confirm": 0.04,
				"last": 0.05,
				"last+request": 0.06,
				"last+confirm": 0.04,
				"other": 0.1
			},
			"音乐": {
				"inform_major_key": 0.2,
				"inform_major_key+request": 0.08,
				"inform_major_key+confirm": 0.02,
				"first": 0.05,
				"first+request": 0.06,
				"first+confirm": 0.04,
				"second": 0.05,
				"second+request": 0.06,
				"second+confirm": 0.04,
				"third": 0.05,
				"third+request": 0.06,
				"third+confirm": 0.04,
				"last": 0.05,
				"last+request": 0.06,
				"last+confirm": 0.04,
				"other": 0.1
			}
		},
		"casual": {
			"电影": {
				"inform_slot": 0.55,
				"inform_major_key": 0.1,
				"inform_major_key+request": 0.2,
				"inform_major_key+confirm": 0.15
			},
			"音乐": {
				"inform_slot": 0.55,
				"inform_major_key": 0.1,
				"inform_major_key+request": 0.2,
				"inform_major_key+confirm": 0.15
			}
		},
		"switch_domain": {
			"电影": {
				"inform_slot": 0.5,
				"inform_major_key": 0.2,
				"inform_major_key+request": 0.25,
				"inform_major_key+confirm": 0.05,
			},
			"音乐": {
				"inform_slot": 0.5,
				"inform_major_key": 0.2,
				"inform_major_key+request": 0.25,
				"inform_major_key+confirm": 0.05,
			},
			"天气": {
				"inform_major_key+request": 1.0,
			},
			"时间": {
				"request": 1.0
			}
		}
	},
	"inform": {
		"usual": {
			"电影": {
				"request": 0.5,
				"confirm": 0.05,
				"implicit_request": 0.1,
				"inform_major_key+request": 0.25,
				"inform_major_key+confirm": 0.1
			},
			"音乐": {
				"request": 0.5,
				"confirm": 0.05,
				"implicit_request": 0.1,
				"inform_major_key+request": 0.25,
				"inform_major_key+confirm": 0.1
			},
			"天气": {
				"implicit_request": 0.2,
				"inform_major_key+request": 0.8
			},
			"时间": {
				"request": 1.0
			}
		},
		"casual": {
			"电影": {
				"inform_slot": 1.0
			},
			"音乐": {
				"inform_slot": 1.0
			},
			"天气": {
				"inform_major_key+request": 1.0
			},
			"时间": {
				"request": 1.0
			}
		},
		"switch_domain": {
			"电影": {
				"inform_slot": 0.5,
				"inform_major_key": 0.2,
				"inform_major_key+request": 0.25,
				"inform_major_key+confirm": 0.05,
			},
			"音乐": {
				"inform_slot": 0.5,
				"inform_major_key": 0.2,
				"inform_major_key+request": 0.25,
				"inform_major_key+confirm": 0.05,
			},
			"天气": {
				"inform_major_key+request": 1.0,
			},
			"时间": {
				"request": 1.0
			}
		}
	},
	"confirm": {
		"usual": {
			"电影": {
				"affirm": 0.5,
				"deny": 0.5
			},
			"音乐": {
				"affirm": 0.5,
				"deny": 0.5
			}
		},
		"casual": {
			"电影": {
				"inform_slot": 0.55,
				"inform_major_key": 0.1,
				"inform_major_key+request": 0.2,
				"inform_major_key+confirm": 0.15
			},
			"音乐": {
				"inform_slot": 0.55,
				"inform_major_key": 0.1,
				"inform_major_key+request": 0.2,
				"inform_major_key+confirm": 0.15
			}
		},
		"switch_domain": {
			"电影": {
				"inform_slot": 0.5,
				"inform_major_key": 0.2,
				"inform_major_key+request": 0.25,
				"inform_major_key+confirm": 0.05,
			},
			"音乐": {
				"inform_slot": 0.5,
				"inform_major_key": 0.2,
				"inform_major_key+request": 0.25,
				"inform_major_key+confirm": 0.05,
			},
			"天气": {
				"inform_major_key+request": 1.0,
			},
			"时间": {
				"request": 1.0
			}
		}
	}
}
user_slot_prop = {
	"inform_slot": {
		"电影": {"主演": 0.35, "导演": 0.25, "类型": 0.15, "地区": 0.12, "年代": 0.08, "资费": 0.05},
		"音乐": {"歌手": 0.4, "曲风": 0.25, "年代": 0.2, "专辑": 0.15}
	},
	"inform_major_key": {
		"电影": {"片名": 1.0},
		"音乐": {"歌名": 1.0},
		"天气": {"城市": 0.7, "时间": 0.3}
	},
	"request": {
		"电影": {"主演": 0.3, "导演": 0.2, "类型": 0.1, "评分": 0.1, "简介": 0.1, "地区": 0.05,
		       "资费": 0.05, "上映日期": 0.05, "片长": 0.03, "年代": 0.02},
		"音乐": {"歌手": 0.35, "专辑": 0.25, "曲风": 0.25, "年代": 0.1, "发行日期": 0.03, "时长": 0.02},
		"天气": {"天气": 0.6, "气温": 0.3, "雨": 0.1},
		"时间": {"时刻": 0.5, "日期": 0.25, "星期": 0.25}
	},
	"confirm": {
		"电影": {"主演": 0.35, "导演": 0.25, "类型": 0.15, "地区": 0.12, "年代": 0.08, "资费": 0.05},
		"音乐": {"歌手": 0.4, "曲风": 0.25, "年代": 0.2, "专辑": 0.15}
	},
	"implicit_request": {
		"电影": {"片名": 1.0},
		"音乐": {"歌名": 1.0},
		"天气": {"城市": 0.7, "时间": 0.3}
	}
}
num_slots_prop = {1: 0.7, 2: 0.22, 3: 0.08}
num_values_prop = {0: 0.2, 1: 0.5, 2: 0.2, 3: 0.1}
sentiment_prop = {"LIKE": 0.7, "DISLIKE": 0.3}
user_nl_candidate = {
	"request": {
		"电影": {
			"主演": [
				"主演是谁？",
				"是谁演的？",
				"主演有哪些？",
				"演员阵容是？",
				"有哪些演员？"
			],
			"导演": [
				"是谁导演的？",
				"是谁拍的？",
				"导演是谁？"
			],
			"类型": [
				"是什么类型的电影？",
				"类型是什么？",
				"属于什么类型？"
			],
			"地区": [
				"是什么地区的电影？",
				"是哪个国家的电影？",
				"是哪里拍的？"
			],
			"评分": [
				"评分是多少？",
				"评分有多少？"
			],
			"年代": [
				"是什么年代的？",
				"是哪个年代的？",
				"属于什么年代？"
			],
			"上映日期": [
				"是哪一年拍的？",
				"是什么时候的电影？",
				"是哪一年的电影？"
			],
			"资费": [
				"免费么？",
				"免不免费？",
				"是免费电影还是付费电影？"
			],
			"片长": [
				"有多长？",
				"有多少分钟？",
				"时长是多少？"
			],
			"简介": [
				"简介是什么？",
				"讲了个什么故事？",
				"故事梗概是什么？"
			],
		},
		"音乐": {
			"歌手": [
				"歌手是谁？",
				"是谁唱的？",
				"是谁的歌？"
			],
			"专辑": [
				"是哪张专辑的？",
				"属于哪张专辑？",
				"是哪张专辑里的歌？"
			],
			"曲风": [
				"属于什么风格？",
				"属于什么流派？",
				"属于什么曲风？",
				"是什么曲风的歌？"
			],
			"年代": [
				"是什么年代的歌？",
				"是哪个年代的？",
				"属于哪个年代？",
				"是什么年代的歌？"
			],
			"发行日期": [
				"是什么时候发行的？",
				"是哪一年发行的？",
				"是哪一年的歌？",
				"是哪一年出的？"
			],
			"时长": [
				"有多长？",
				"有几分钟？",
				"时长是多少？"
			]
		},
		"天气": {
			"天气": [
				"天气怎么样？",
				"是什么天气？"
			],
			"气温": [
				"多少度？",
				"冷不冷？",
				"热不热?",
				"气温是多少？"
			],
			"雨": [
				"有雨么？",
				"有没有雨？",
				"下雨么？"
			]
			
		},
		"时间": {
			"日期": [
				"今天几号？",
				"今天多少号？",
				"今天几月几号？"
			],
			"星期": [
				"今天是星期几？",
				"今天星期几了？"
			],
			"时刻": [
				"几点了？",
				"现在是几点钟？",
				"现在几点了？"
			]
		}
		
	},
	"confirm": {
		"电影": {
			"主演": [
				"的主演有%s么？",
				"是%s演的么？",
				"是不是%s主演的？"
			],
			"导演": [
				"的导演是%s么？",
				"是%s导演的么？",
				"是不是%s导演的？"
			],
			"年代": [
				"是%s的电影么？",
				"是%s的影片么？",
				"是不是%s的电影？"
			],
			"类型": [
				"是%s类型的电影么？",
				"是不是%s类型的电影？",
				"是%s类型的么？",
				"是不是%s类型的？"
			],
			"地区": [
				"是%s的电影么？",
				"是不是%s的电影？"
			],
			"资费": [
				"是%s电影么？",
				"是%s的么？",
				"是不是%s电影？"
			]
		},
		"音乐": {
			"歌手": [
				"是%s的歌么？",
				"是%s唱的么？",
				"是不是%s唱的？"
			],
			"专辑": [
				"的专辑是%s么？",
				"是专辑%s里的歌么？"
			],
			"年代": [
				"是%s的歌么？",
				"是%s的音乐么？",
				"是不是%s的歌？"
			],
			"曲风": [
				"的曲风是%s么？",
				"是不是%s曲风的歌？",
				"属不属于%s曲风的歌？",
				"是不是%s曲风？",
				"属不属于%s曲风？"
			]
		}
	},
	"first": {
		"电影": [
			"第一部",
			"第一个电影",
			"第一部电影",
			"第一个影片",
			"第一部影片"
		],
		"音乐": [
			"第一首",
			"第一个音乐",
			"第一首音乐",
			"第一首歌"
		]
	},
	"second": {
		"电影": [
			"第二部",
			"第二个电影",
			"第二部电影",
			"第二个影片",
			"第二部影片"
		],
		"音乐": [
			"第二首",
			"第二个音乐",
			"第二首音乐",
			"第二首歌"
		]
	},
	"third": {
		"电影": [
			"第三部",
			"第三个电影",
			"第三部电影",
			"第三个影片",
			"第三部影片"
		],
		"音乐": [
			"第三首",
			"第三个音乐",
			"第三首音乐",
			"第三首歌"
		]
	},
	"last": {
		"电影": [
			"最后一部",
			"最后一个电影",
			"最后一部电影",
			"最后一个影片",
			"最后一部影片"
		],
		"音乐": [
			"最后一首",
			"最后一个音乐",
			"最后一首音乐",
			"最后一首歌"
		]
	}
}


def rounds_sys_gen():
	def random_pick(prop_dict):
		x = random.uniform(0, 1)
		cumulative_prop = 0.0
		for item, item_prop in prop_dict.items():
			cumulative_prop += item_prop
			if x <= cumulative_prop: break
		return item
	
	def domain_gen(sys_act_type):
		domain_prop = sys_domain_prop[sys_act_type]
		domain = random_pick(domain_prop)
		return domain
	
	def slot_gen(sys_act_type, domain):
		if sys_slot_prop[sys_act_type] is not None:
			slot_prop = sys_slot_prop[sys_act_type]
			if slot_prop[domain] is not None:
				slot_prop = slot_prop[domain]
				slot = random_pick(slot_prop)
		return slot
	
	def act_type_gen():
		rounds_sys = []
		index = 0
		for sys_act_type in sys_act_set:
			num_rounds_per_type = int(NUM_ROUNDS * sys_act_prop[sys_act_type])
			for id_per_type in range(num_rounds_per_type):
				rounds_sys.append({"sys_act": sys_act_type, "id": index})
				index += 1
		return rounds_sys
	
	rounds_sys = act_type_gen()
	
	def content_gen(round_sys):
		sys_act_type = round_sys["sys_act"]
		content = {}
		if sys_act_type != "hello":
			domain = domain_gen(sys_act_type)
			if sys_act_type == "request":
				slot = slot_gen(sys_act_type, domain)
				content = {domain: slot}
			elif sys_act_type == "inform_no_match":
				slot = slot_gen(sys_act_type, domain)
				content = {domain: {slot: []}}
			elif sys_act_type == "inform_one_match":
				slot = slot_gen(sys_act_type, domain)
				value = random.sample(OTLG[domain]["informable"][slot], 1)
				content = {domain: {slot: value}}
			elif sys_act_type == "inform_some_match":
				slot = slot_gen(sys_act_type, domain)
				num = random.choice([2, 3, 4])
				values = random.sample(OTLG[domain]["informable"][slot], num)
				content = {domain: {slot: values}}
			elif sys_act_type == "inform":
				slot = slot_gen(sys_act_type, domain)
				content = {domain: [slot]}
			elif sys_act_type == "confirm":
				slot = slot_gen(sys_act_type, domain)
				value = random.choice(OTLG[domain]["informable"][slot])
				content = {domain: {slot: value}}
		round_sys["sys_act"] = {sys_act_type: content}
		return round_sys
	
	rounds_sys = map(content_gen, rounds_sys)
	
	def sys_nlg(round_sys):
		def list2str(items):
			for forbidden in ['今年', '去年', '前年']:
				if forbidden in items:
					items.remove(forbidden)
			return '、'.join(items)
		
		def nlg_movie(act_type, content):
			global sys_kb_pointer_movie
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
				movie = kb_movie[sys_kb_pointer_movie]
				sys_kb_pointer_movie = (sys_kb_pointer_movie + 1) % len(kb_movie)
				nl += "《%s》" % movie['片名']
				for slot in content:
					if slot == "主演":
						nl += "的主演有%s，" % list2str(movie[slot])
					elif slot == "导演":
						nl += "的导演是%s，" % movie[slot]
					elif slot == "类型":
						nl += "的类型是%s，" % list2str(movie[slot])
					elif slot == "地区":
						nl += "是%s电影，" % list2str(movie[slot])
					elif slot == "评分":
						nl += "的评分为%s，" % movie[slot]
					elif slot == "年代":
						nl += "是%s年代的，" % list2str(movie[slot])
					elif slot == "上映日期":
						nl += "是%s年上映的，" % movie[slot]
					elif slot == "资费":
						nl += "是%s电影，" % movie[slot]
					elif slot == "片长":
						nl += "的片长为%s分钟，" % movie[slot]
					elif slot == "简介":
						nl += "的简介为：..."
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
				movies = content["片名"]
				num_movies = len(movies)
				if num_movies == 0:
					nl += "对不起没有找到符合您要求的电影。"
			elif act_type == "inform_one_match":
				movies = content["片名"]
				num_movies = len(movies)
				if num_movies == 1:
					nl += "为您找到一部影片：%s。请问您还想知道关于该电影的哪些信息？" % list2str(movies)
			elif act_type == "inform_some_match":
				movies = content["片名"]
				num_movies = len(movies)
				if num_movies == 2:
					nl += "为您找到两部影片：%s。您想先看哪一部呢？" % list2str(movies)
				elif num_movies == 3:
					nl += "为您找到三部影片：%s。您想先看哪一部呢？" % list2str(movies)
				elif num_movies > 3:
					nl += "找到多部影片，已为您选择评分最高的三部：%s。您想先看哪一部呢？" % list2str(movies[0:3])
			return nl
		
		def nlg_music(act_type, content):
			global sys_kb_pointer_music
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
				music = kb_music[sys_kb_pointer_music]
				sys_kb_pointer_music = (sys_kb_pointer_music + 1) % len(kb_music)
				nl += "《%s》" % music['歌名']
				for slot in content:
					if slot == "歌手":
						nl += "的歌手是%s，" % music[slot]
					elif slot == "专辑":
						nl += "的专辑是%s，" % "《%s》" % music[slot]
					elif slot == "曲风":
						nl += "的类型是%s，" % list2str(music[slot])
					elif slot == "年代":
						nl += "是%s年代的，" % list2str(music[slot])
					elif slot == "发行日期":
						nl += "的发行日期是%s，" % music[slot]
					elif slot == "时长":
						nl += "的歌曲时长为%s，" % music[slot]
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
					nl += "为您找到一首歌曲：%s。请问您还想知道关于该歌曲的哪些信息？" % list2str(songs)
			elif act_type == "inform_some_match":
				songs = content["歌名"]
				num_songs = len(songs)
				if num_songs == 2:
					nl += "为您找到两首歌曲：%s。您想先听哪一首呢？" % list2str(songs)
				elif num_songs == 3:
					nl += "为您找到三首歌曲：%s。您想先听哪一首呢？" % list2str(songs)
				elif num_songs > 3:
					random.shuffle(songs)
					nl += "找到多首歌曲，已为您随机选择三首：%s。您想先听哪一首呢？" % list2str(songs[0:3])
			return nl
		
		def nlg_weather(act_type, content):
			nl = ""
			if act_type == "inform":
				city = random.choice(OTLG["天气"]["informable"]["城市"])
				nl += city + random.choice(["今天", "明天"])
				for slot in content:
					if slot == "天气":
						nl += "天气状况：" + random.choice(["晴", "阴", "多云", "雨"])
					elif slot == "气温":
						nl += "气温：17℃~28℃，"
					elif slot == "雨":
						nl += "有雨" if random.random() < 0.5 else "没有雨"
				nl = nl.rstrip('，')
				nl += "。 "
			return nl
		
		def nlg_time(act_type, content):
			nl = ""
			if act_type == "inform":
				for slot in content:
					if slot == "日期":
						nl += "今天是2018年4月10日。"
					elif slot == "星期":
						nl += "今天是星期" + random.choice(["一", "二", "三", "四", "五", "六", "日"]) + "。"
					elif slot == "时刻":
						nl += "现在是北京时间17点18分。"
			return nl
		
		nl = ""
		nlg_fun = {"电影": nlg_movie, "音乐": nlg_music, "天气": nlg_weather, "时间": nlg_time}
		sys_act_type = list(round_sys["sys_act"].keys())[0]
		content = round_sys["sys_act"][sys_act_type]
		if sys_act_type == "hello":
			nl += "欢迎使用清华大学电子系多领域信息问询系统，您可以对电影、音乐、时间、天气信息进行查询。"
		else:
			for domain in content:
				nl += nlg_fun[domain](sys_act_type, content[domain])
		round_sys["sys_nl"] = nl
		return round_sys
	
	rounds_sys = list(map(sys_nlg, rounds_sys))
	
	return rounds_sys


def user_response_gen(rounds_sys):
	def random_pick(prop_dict):
		x = random.uniform(0, 1)
		cumulative_prop = 0.0
		for item, item_prop in prop_dict.items():
			cumulative_prop += item_prop
			if x <= cumulative_prop: break
		return item
	
	def random_sample(prop_dict, num=1):
		result = set()
		while len(result) < num:
			x = random.uniform(0, 1)
			cumulative_prop = 0.0
			for item, item_prop in prop_dict.items():
				cumulative_prop += item_prop
				if x <= cumulative_prop:
					result.add(item)
					break
		return list(result)
	
	def act_gen(round, sys_act_type, response_mode, user_domain, response_policy):
		if sys_act_type == "hello":
			if response_mode == "usual":
				if response_policy == "inform_slot":
					content = {user_domain: {}}
					num_slots = random_pick(num_slots_prop)
					user_slots = random_sample(user_slot_prop[response_policy][user_domain], num_slots)
					for slot in user_slots:
						content[user_domain].update({slot: {}})
						num_values = random_pick(num_values_prop)
						user_values = random.sample(OTLG[user_domain]["informable"][slot],
						                            min(num_values, len(OTLG[user_domain]["informable"][slot]) - 1))
						for user_value in user_values:
							content[user_domain][slot].update({user_value: random_pick(sentiment_prop)})
					user_act = {"inform": content}
				elif response_policy == "inform_major_key":
					content = {user_domain: {}}
					user_slot = random_pick(user_slot_prop[response_policy][user_domain])
					user_value = random.choice(OTLG[user_domain]["informable"][user_slot])
					content[user_domain].update({user_slot: {user_value: "LIKE"}})
					user_act = {"inform": content}
				elif response_policy == "inform_major_key+request":
					user_act = {"inform": {}, "request": {}}
					# inform_major_key
					content_inform = {user_domain: {}}
					user_slot = random_pick(user_slot_prop["inform_major_key"][user_domain])
					user_value = random.choice(OTLG[user_domain]["informable"][user_slot])
					content_inform[user_domain] = {user_slot: {user_value: "LIKE"}}
					user_act["inform"] = content_inform
					# request
					content_request = {user_domain: {}}
					num_slots = random_pick(num_slots_prop)
					user_slots = random_sample(user_slot_prop["request"][user_domain],
					                           min(num_slots, len(OTLG[user_domain]["requestable"])))
					content_request[user_domain] = user_slots
					user_act["request"] = content_request
				elif response_policy == "inform_major_key+confirm":
					user_act = {"inform": {}, "confirm": {}}
					# inform_major_key
					content_inform = {user_domain: {}}
					user_slot = random_pick(user_slot_prop["inform_major_key"][user_domain])
					user_value = random.choice(OTLG[user_domain]["informable"][user_slot])
					content_inform[user_domain].update({user_slot: {user_value: "LIKE"}})
					user_act["inform"] = content_inform
					# confirm
					content_confirm = {user_domain: {}}
					user_slot = random_pick(user_slot_prop["confirm"][user_domain])
					user_value = random.choice(OTLG[user_domain]["informable"][user_slot])
					content_confirm[user_domain].update({user_slot: user_value})
					user_act["confirm"] = content_confirm
				elif response_policy == "request":
					content = {user_domain: {}}
					num_slots = random_pick(num_slots_prop)
					user_slots = random_sample(user_slot_prop[response_policy][user_domain],
					                           min(num_slots, len(OTLG[user_domain]["requestable"])))
					content[user_domain] = user_slots
					user_act = {"request": content}
		elif sys_act_type == "request":
			sys_domain = list(round["sys_act"][sys_act_type].keys())[0]
			sys_slot = round["sys_act"][sys_act_type][sys_domain]
			if response_mode == "usual":
				if response_policy == "inform_slot":
					content = {user_domain: {}}
					user_slot = sys_slot
					content[user_domain].update({user_slot: {}})
					num_values = random_pick(num_values_prop)
					user_values = random.sample(OTLG[user_domain]["informable"][user_slot],
					                            min(num_values, len(OTLG[user_domain]["informable"][user_slot]) - 1))
					for user_value in user_values:
						content[user_domain][user_slot].update({user_value: random_pick(sentiment_prop)})
					user_act = {"inform": content}
			elif response_mode == "casual":
				if response_policy == "inform_slot":
					content = {user_domain: {}}
					while True:
						num_slots = random_pick(num_slots_prop)
						user_slots = random_sample(user_slot_prop[response_policy][user_domain],
						                           min(num_slots, len(OTLG[user_domain]["requestable"])))
						if sys_slot not in user_slots:
							break
					for slot in user_slots:
						content[user_domain].update({slot: {}})
						num_values = random_pick(num_values_prop)
						user_values = random.sample(OTLG[user_domain]["informable"][slot],
						                            min(num_values, len(OTLG[user_domain]["informable"][slot]) - 1))
						for user_value in user_values:
							content[user_domain][slot].update({user_value: random_pick(sentiment_prop)})
					user_act = {"inform": content}
				elif response_policy == "inform_major_key":
					content = {user_domain: {}}
					user_slot = random_pick(user_slot_prop[response_policy][user_domain])
					user_value = random.choice(OTLG[user_domain]["informable"][user_slot])
					content[user_domain].update({user_slot: {user_value: "LIKE"}})
					user_act = {"inform": content}
				elif response_policy == "inform_major_key+request":
					user_act = {"inform": {}, "request": {}}
					# inform_major_key
					content_inform = {user_domain: {}}
					user_slot = random_pick(user_slot_prop["inform_major_key"][user_domain])
					user_value = random.choice(OTLG[user_domain]["informable"][user_slot])
					content_inform[user_domain] = {user_slot: {user_value: "LIKE"}}
					user_act["inform"] = content_inform
					# request
					content_request = {user_domain: {}}
					num_slots = random_pick(num_slots_prop)
					user_slots = random_sample(user_slot_prop["request"][user_domain],
					                           min(num_slots, len(OTLG[user_domain]["requestable"])))
					content_request[user_domain] = user_slots
					user_act["request"] = content_request
				elif response_policy == "inform_major_key+confirm":
					user_act = {"inform": {}, "confirm": {}}
					# inform_major_key
					content_inform = {user_domain: {}}
					user_slot = random_pick(user_slot_prop["inform_major_key"][user_domain])
					user_value = random.choice(OTLG[user_domain]["informable"][user_slot])
					content_inform[user_domain].update({user_slot: {user_value: "LIKE"}})
					user_act["inform"] = content_inform
					# confirm
					content_confirm = {user_domain: {}}
					user_slot = random_pick(user_slot_prop["confirm"][user_domain])
					user_value = random.choice(OTLG[user_domain]["informable"][user_slot])
					content_confirm[user_domain].update({user_slot: user_value})
					user_act["confirm"] = content_confirm
			elif response_mode == "switch_domain":
				if response_policy == "inform_slot":
					content = {user_domain: {}}
					num_slots = random_pick(num_slots_prop)
					user_slots = random_sample(user_slot_prop[response_policy][user_domain], num_slots)
					for slot in user_slots:
						content[user_domain].update({slot: {}})
						num_values = random_pick(num_values_prop)
						user_values = random.sample(OTLG[user_domain]["informable"][slot],
						                            min(num_values, len(OTLG[user_domain]["informable"][slot]) - 1))
						for user_value in user_values:
							content[user_domain][slot].update({user_value: random_pick(sentiment_prop)})
					user_act = {"inform": content}
				elif response_policy == "inform_major_key":
					content = {user_domain: {}}
					user_slot = random_pick(user_slot_prop[response_policy][user_domain])
					user_value = random.choice(OTLG[user_domain]["informable"][user_slot])
					content[user_domain].update({user_slot: {user_value: "LIKE"}})
					user_act = {"inform": content}
				elif response_policy == "inform_major_key+request":
					user_act = {"inform": {}, "request": {}}
					# inform_major_key
					content_inform = {user_domain: {}}
					user_slot = random_pick(user_slot_prop["inform_major_key"][user_domain])
					user_value = random.choice(OTLG[user_domain]["informable"][user_slot])
					content_inform[user_domain] = {user_slot: {user_value: "LIKE"}}
					user_act["inform"] = content_inform
					# request
					content_request = {user_domain: {}}
					num_slots = random_pick(num_slots_prop)
					user_slots = random_sample(user_slot_prop["request"][user_domain],
					                           min(num_slots, len(OTLG[user_domain]["requestable"])))
					content_request[user_domain] = user_slots
					user_act["request"] = content_request
				elif response_policy == "inform_major_key+confirm":
					user_act = {"inform": {}, "confirm": {}}
					# inform_major_key
					content_inform = {user_domain: {}}
					user_slot = random_pick(user_slot_prop["inform_major_key"][user_domain])
					user_value = random.choice(OTLG[user_domain]["informable"][user_slot])
					content_inform[user_domain].update({user_slot: {user_value: "LIKE"}})
					user_act["inform"] = content_inform
					# confirm
					content_confirm = {user_domain: {}}
					user_slot = random_pick(user_slot_prop["confirm"][user_domain])
					user_value = random.choice(OTLG[user_domain]["informable"][user_slot])
					content_confirm[user_domain].update({user_slot: user_value})
					user_act["confirm"] = content_confirm
				elif response_policy == "request":
					content = {user_domain: {}}
					num_slots = random_pick(num_slots_prop)
					user_slots = random_sample(user_slot_prop[response_policy][user_domain],
					                           min(num_slots, len(OTLG[user_domain]["requestable"])))
					content[user_domain] = user_slots
					user_act = {"request": content}
		elif sys_act_type == "inform_no_match":
			if response_policy == "inform_slot":
				content = {user_domain: {}}
				num_slots = random_pick(num_slots_prop)
				user_slots = random_sample(user_slot_prop[response_policy][user_domain], num_slots)
				for slot in user_slots:
					content[user_domain].update({slot: {}})
					num_values = random_pick(num_values_prop)
					user_values = random.sample(OTLG[user_domain]["informable"][slot],
					                            min(num_values, len(OTLG[user_domain]["informable"][slot]) - 1))
					for user_value in user_values:
						content[user_domain][slot].update({user_value: random_pick(sentiment_prop)})
				user_act = {"inform": content}
			elif response_policy == "inform_major_key":
				content = {user_domain: {}}
				user_slot = random_pick(user_slot_prop[response_policy][user_domain])
				user_value = random.choice(OTLG[user_domain]["informable"][user_slot])
				content[user_domain].update({user_slot: {user_value: "LIKE"}})
				user_act = {"inform": content}
			elif response_policy == "inform_major_key+request":
				user_act = {"inform": {}, "request": {}}
				# inform_major_key
				content_inform = {user_domain: {}}
				user_slot = random_pick(user_slot_prop["inform_major_key"][user_domain])
				user_value = random.choice(OTLG[user_domain]["informable"][user_slot])
				content_inform[user_domain] = {user_slot: {user_value: "LIKE"}}
				user_act["inform"] = content_inform
				# request
				content_request = {user_domain: {}}
				num_slots = random_pick(num_slots_prop)
				user_slots = random_sample(user_slot_prop["request"][user_domain],
				                           min(num_slots, len(OTLG[user_domain]["requestable"])))
				content_request[user_domain] = user_slots
				user_act["request"] = content_request
			elif response_policy == "inform_major_key+confirm":
				user_act = {"inform": {}, "confirm": {}}
				# inform_major_key
				content_inform = {user_domain: {}}
				user_slot = random_pick(user_slot_prop["inform_major_key"][user_domain])
				user_value = random.choice(OTLG[user_domain]["informable"][user_slot])
				content_inform[user_domain].update({user_slot: {user_value: "LIKE"}})
				user_act["inform"] = content_inform
				# confirm
				content_confirm = {user_domain: {}}
				user_slot = random_pick(user_slot_prop["confirm"][user_domain])
				user_value = random.choice(OTLG[user_domain]["informable"][user_slot])
				content_confirm[user_domain].update({user_slot: user_value})
				user_act["confirm"] = content_confirm
			elif response_policy == "request":
				content = {user_domain: {}}
				num_slots = random_pick(num_slots_prop)
				user_slots = random_sample(user_slot_prop[response_policy][user_domain],
				                           min(num_slots, len(OTLG[user_domain]["requestable"])))
				content[user_domain] = user_slots
				user_act = {"request": content}
		elif sys_act_type == "inform_one_match":
			if response_mode == "usual":
				sys_domain = list(round["sys_act"][sys_act_type].keys())[0]
				sys_slot = list(round["sys_act"][sys_act_type][sys_domain].keys())[0]
				sys_value = round["sys_act"][sys_act_type][sys_domain][sys_slot][0]
				if response_policy == "request":
					content = {user_domain: {}}
					num_slots = random_pick(num_slots_prop)
					user_slots = random_sample(user_slot_prop[response_policy][user_domain],
					                           min(num_slots, len(OTLG[user_domain]["requestable"])))
					content[user_domain] = user_slots
					user_act = {"request": content}
				elif response_policy == "inform_major_key+request":
					user_act = {"inform": {}, "request": {}}
					# inform_major_key
					content_inform = {user_domain: {}}
					user_slot = sys_slot
					user_value = sys_value
					content_inform[user_domain] = {user_slot: {user_value: "LIKE"}}
					user_act["inform"] = content_inform
					# request
					content_request = {user_domain: {}}
					num_slots = random_pick(num_slots_prop)
					user_slots = random_sample(user_slot_prop["request"][user_domain],
					                           min(num_slots, len(OTLG[user_domain]["requestable"])))
					content_request[user_domain] = user_slots
					user_act["request"] = content_request
				elif response_policy == "inform_major_key+confirm":
					user_act = {"inform": {}, "confirm": {}}
					# inform_major_key
					content_inform = {user_domain: {}}
					user_slot = sys_slot
					user_value = sys_value
					content_inform[user_domain].update({user_slot: {user_value: "LIKE"}})
					user_act["inform"] = content_inform
					# confirm
					content_confirm = {user_domain: {}}
					user_slot = random_pick(user_slot_prop["confirm"][user_domain])
					user_value = random.choice(OTLG[user_domain]["informable"][user_slot])
					content_confirm[user_domain].update({user_slot: user_value})
					user_act["confirm"] = content_confirm
			elif response_mode == "casual" or response_mode == "switch_domain":
				if response_policy == "inform_slot":
					content = {user_domain: {}}
					num_slots = random_pick(num_slots_prop)
					user_slots = random_sample(user_slot_prop[response_policy][user_domain], num_slots)
					for slot in user_slots:
						content[user_domain].update({slot: {}})
						num_values = random_pick(num_values_prop)
						user_values = random.sample(OTLG[user_domain]["informable"][slot],
						                            min(num_values, len(OTLG[user_domain]["informable"][slot]) - 1))
						for user_value in user_values:
							content[user_domain][slot].update({user_value: random_pick(sentiment_prop)})
					user_act = {"inform": content}
				elif response_policy == "inform_major_key":
					content = {user_domain: {}}
					user_slot = random_pick(user_slot_prop[response_policy][user_domain])
					user_value = random.choice(OTLG[user_domain]["informable"][user_slot])
					content[user_domain].update({user_slot: {user_value: "LIKE"}})
					user_act = {"inform": content}
				elif response_policy == "inform_major_key+request":
					user_act = {"inform": {}, "request": {}}
					# inform_major_key
					content_inform = {user_domain: {}}
					user_slot = random_pick(user_slot_prop["inform_major_key"][user_domain])
					user_value = random.choice(OTLG[user_domain]["informable"][user_slot])
					content_inform[user_domain] = {user_slot: {user_value: "LIKE"}}
					user_act["inform"] = content_inform
					# request
					content_request = {user_domain: {}}
					num_slots = random_pick(num_slots_prop)
					user_slots = random_sample(user_slot_prop["request"][user_domain],
					                           min(num_slots, len(OTLG[user_domain]["requestable"])))
					content_request[user_domain] = user_slots
					user_act["request"] = content_request
				elif response_policy == "inform_major_key+confirm":
					user_act = {"inform": {}, "confirm": {}}
					# inform_major_key
					content_inform = {user_domain: {}}
					user_slot = random_pick(user_slot_prop["inform_major_key"][user_domain])
					user_value = random.choice(OTLG[user_domain]["informable"][user_slot])
					content_inform[user_domain].update({user_slot: {user_value: "LIKE"}})
					user_act["inform"] = content_inform
					# confirm
					content_confirm = {user_domain: {}}
					user_slot = random_pick(user_slot_prop["confirm"][user_domain])
					user_value = random.choice(OTLG[user_domain]["informable"][user_slot])
					content_confirm[user_domain].update({user_slot: user_value})
					user_act["confirm"] = content_confirm
				elif response_policy == "request":
					content = {user_domain: {}}
					num_slots = random_pick(num_slots_prop)
					user_slots = random_sample(user_slot_prop[response_policy][user_domain],
					                           min(num_slots, len(OTLG[user_domain]["requestable"])))
					content[user_domain] = user_slots
					user_act = {"request": content}
		elif sys_act_type == "inform_some_match":
			if response_mode == "usual":
				sys_domain = list(round["sys_act"][sys_act_type].keys())[0]
				sys_slot = list(round["sys_act"][sys_act_type][sys_domain].keys())[0]
				sys_values = round["sys_act"][sys_act_type][sys_domain][sys_slot]
				if response_policy == "inform_major_key":
					content = {user_domain: {}}
					user_slot = sys_slot
					while True:
						user_value = random.choice(sys_values)
						if user_value in round["sys_nl"]:
							break
					content[user_domain].update({user_slot: {user_value: "LIKE"}})
					user_act = {"inform": content}
				elif response_policy == "inform_major_key+request":
					user_act = {"inform": {}, "request": {}}
					# inform_major_key
					content_inform = {user_domain: {}}
					user_slot = sys_slot
					user_value = random.choice(sys_values)
					content_inform[user_domain] = {user_slot: {user_value: "LIKE"}}
					user_act["inform"] = content_inform
					# request
					content_request = {user_domain: {}}
					num_slots = random_pick(num_slots_prop)
					user_slots = random_sample(user_slot_prop["request"][user_domain],
					                           min(num_slots, len(OTLG[user_domain]["requestable"])))
					content_request[user_domain] = user_slots
					user_act["request"] = content_request
				elif response_policy == "inform_major_key+confirm":
					user_act = {"inform": {}, "confirm": {}}
					# inform_major_key
					content_inform = {user_domain: {}}
					user_slot = sys_slot
					user_value = random.choice(sys_values)
					content_inform[user_domain].update({user_slot: {user_value: "LIKE"}})
					user_act["inform"] = content_inform
					# confirm
					content_confirm = {user_domain: {}}
					user_slot = random_pick(user_slot_prop["confirm"][user_domain])
					user_value = random.choice(OTLG[user_domain]["informable"][user_slot])
					content_confirm[user_domain].update({user_slot: user_value})
					user_act["confirm"] = content_confirm
				elif response_policy == "first":
					content = {}
					user_act = {"first": content}
				elif response_policy == "first+request":
					user_act = {"first": {}, "request": {}}
					# first
					content_first = {}
					user_act["first"] = content_first
					# request
					content_request = {user_domain: {}}
					num_slots = random_pick(num_slots_prop)
					user_slots = random_sample(user_slot_prop["request"][user_domain],
					                           min(num_slots, len(OTLG[user_domain]["requestable"])))
					content_request[user_domain] = user_slots
					user_act["request"] = content_request
				elif response_policy == "first+confirm":
					user_act = {"first": {}, "confirm": {}}
					# first
					content_first = {}
					user_act["first"] = content_first
					# confirm
					content_confirm = {user_domain: {}}
					user_slot = random_pick(user_slot_prop["confirm"][user_domain])
					user_value = random.choice(OTLG[user_domain]["informable"][user_slot])
					content_confirm[user_domain].update({user_slot: user_value})
					user_act["confirm"] = content_confirm
				elif response_policy == "second":
					content = {}
					user_act = {"second": content}
				elif response_policy == "second+request":
					user_act = {"second": {}, "request": {}}
					# second
					content_second = {}
					user_act["second"] = content_second
					# request
					content_request = {user_domain: {}}
					num_slots = random_pick(num_slots_prop)
					user_slots = random_sample(user_slot_prop["request"][user_domain],
					                           min(num_slots, len(OTLG[user_domain]["requestable"])))
					content_request[user_domain] = user_slots
					user_act["request"] = content_request
				elif response_policy == "second+confirm":
					user_act = {"second": {}, "confirm": {}}
					# second
					content_second = {}
					user_act["second"] = content_second
					# confirm
					content_confirm = {user_domain: {}}
					user_slot = random_pick(user_slot_prop["confirm"][user_domain])
					user_value = random.choice(OTLG[user_domain]["informable"][user_slot])
					content_confirm[user_domain].update({user_slot: user_value})
					user_act["confirm"] = content_confirm
				elif response_policy == "third":
					if len(sys_values) < 3:
						round["response_policy"] = random.choice(["first", "second", "last"])
						response_policy = round["response_policy"]
						content = {}
						user_act = {response_policy: content}
					else:
						content = {}
						user_act = {"third": content}
				elif response_policy == "third+request":
					# third
					if len(sys_values) < 3:
						round["response_policy"] = random.choice(["first+request", "second+request", "last+request"])
						response_policy = round["response_policy"]
						content = {}
						user_act = {response_policy.split('+')[0]: content, "request": {}}
					else:
						content = {}
						user_act = {"third": content, "request": {}}
					# request
					content_request = {user_domain: {}}
					num_slots = random_pick(num_slots_prop)
					user_slots = random_sample(user_slot_prop["request"][user_domain],
					                           min(num_slots, len(OTLG[user_domain]["requestable"])))
					content_request[user_domain] = user_slots
					user_act["request"] = content_request
				elif response_policy == "third+confirm":
					# third
					if len(sys_values) < 3:
						round["response_policy"] = random.choice(["first+confirm", "second+confirm", "last+confirm"])
						response_policy = round["response_policy"]
						content = {}
						user_act = {response_policy.split('+')[0]: content, "confirm": {}}
					else:
						content = {}
						user_act = {"third": content, "confirm": {}}
					# confirm
					content_confirm = {user_domain: {}}
					user_slot = random_pick(user_slot_prop["confirm"][user_domain])
					user_value = random.choice(OTLG[user_domain]["informable"][user_slot])
					content_confirm[user_domain].update({user_slot: user_value})
					user_act["confirm"] = content_confirm
				elif response_policy == "last":
					content = {}
					user_act = {"last": content}
				elif response_policy == "last+request":
					user_act = {"last": {}, "request": {}}
					# last
					content_last = {}
					user_act["last"] = content_last
					# request
					content_request = {user_domain: {}}
					num_slots = random_pick(num_slots_prop)
					user_slots = random_sample(user_slot_prop["request"][user_domain],
					                           min(num_slots, len(OTLG[user_domain]["requestable"])))
					content_request[user_domain] = user_slots
					user_act["request"] = content_request
				elif response_policy == "last+confirm":
					user_act = {"last": {}, "confirm": {}}
					# last
					content_last = {}
					user_act["last"] = content_last
					# confirm
					content_confirm = {user_domain: {}}
					user_slot = random_pick(user_slot_prop["confirm"][user_domain])
					user_value = random.choice(OTLG[user_domain]["informable"][user_slot])
					content_confirm[user_domain].update({user_slot: user_value})
					user_act["confirm"] = content_confirm
				elif response_policy == "other":
					content = {}
					user_act = {"other": content}
			elif response_mode == "casual" or response_mode == "switch_domain":
				if response_policy == "inform_slot":
					content = {user_domain: {}}
					num_slots = random_pick(num_slots_prop)
					user_slots = random_sample(user_slot_prop[response_policy][user_domain], num_slots)
					for slot in user_slots:
						content[user_domain].update({slot: {}})
						num_values = random_pick(num_values_prop)
						user_values = random.sample(OTLG[user_domain]["informable"][slot],
						                            min(num_values, len(OTLG[user_domain]["informable"][slot]) - 1))
						for user_value in user_values:
							content[user_domain][slot].update({user_value: random_pick(sentiment_prop)})
					user_act = {"inform": content}
				elif response_policy == "inform_major_key":
					content = {user_domain: {}}
					user_slot = random_pick(user_slot_prop[response_policy][user_domain])
					user_value = random.choice(OTLG[user_domain]["informable"][user_slot])
					content[user_domain].update({user_slot: {user_value: "LIKE"}})
					user_act = {"inform": content}
				elif response_policy == "inform_major_key+request":
					user_act = {"inform": {}, "request": {}}
					# inform_major_key
					content_inform = {user_domain: {}}
					user_slot = random_pick(user_slot_prop["inform_major_key"][user_domain])
					user_value = random.choice(OTLG[user_domain]["informable"][user_slot])
					content_inform[user_domain] = {user_slot: {user_value: "LIKE"}}
					user_act["inform"] = content_inform
					# request
					content_request = {user_domain: {}}
					num_slots = random_pick(num_slots_prop)
					user_slots = random_sample(user_slot_prop["request"][user_domain],
					                           min(num_slots, len(OTLG[user_domain]["requestable"])))
					content_request[user_domain] = user_slots
					user_act["request"] = content_request
				elif response_policy == "inform_major_key+confirm":
					user_act = {"inform": {}, "confirm": {}}
					# inform_major_key
					content_inform = {user_domain: {}}
					user_slot = random_pick(user_slot_prop["inform_major_key"][user_domain])
					user_value = random.choice(OTLG[user_domain]["informable"][user_slot])
					content_inform[user_domain].update({user_slot: {user_value: "LIKE"}})
					user_act["inform"] = content_inform
					# confirm
					content_confirm = {user_domain: {}}
					user_slot = random_pick(user_slot_prop["confirm"][user_domain])
					user_value = random.choice(OTLG[user_domain]["informable"][user_slot])
					content_confirm[user_domain].update({user_slot: user_value})
					user_act["confirm"] = content_confirm
				elif response_policy == "request":
					content = {user_domain: {}}
					num_slots = random_pick(num_slots_prop)
					user_slots = random_sample(user_slot_prop[response_policy][user_domain],
					                           min(num_slots, len(OTLG[user_domain]["requestable"])))
					content[user_domain] = user_slots
					user_act = {"request": content}
		elif sys_act_type == "inform":
			sys_domain = list(round["sys_act"][sys_act_type].keys())[0]
			sys_slot = round["sys_act"][sys_act_type][sys_domain][0]
			if response_mode == "usual":
				if response_policy == "request":
					content = {user_domain: {}}
					while True:
						num_slots = random_pick(num_slots_prop)
						user_slots = random_sample(user_slot_prop[response_policy][user_domain],
						                           min(num_slots, len(OTLG[user_domain]["requestable"])))
						if sys_slot not in user_slots:
							break
					content[user_domain] = user_slots
					user_act = {"request": content}
				elif response_policy == "confirm":
					content = {user_domain: {}}
					while True:
						user_slot = random_pick(user_slot_prop["confirm"][user_domain])
						if sys_slot != user_slot:
							break
					user_value = random.choice(OTLG[user_domain]["informable"][user_slot])
					content[user_domain] = {user_slot: user_value}
					user_act = {"confirm": content}
				elif response_policy == "implicit_request":
					content = {user_domain: {}}
					user_slot = random_pick(user_slot_prop[response_policy][user_domain])
					if user_domain in ["电影", "音乐"]:
						sys_value = round["sys_nl"].split("》")[0].split("《")[1]
					elif user_domain in ["天气"]:
						for value in OTLG[user_domain]["informable"][user_slot]:
							if value in round["sys_nl"]:
								sys_value = value
								break
					while True:
						user_value = random.choice(OTLG[user_domain]["informable"][user_slot])
						if user_value != sys_value:
							break
					content[user_domain] = {user_slot: user_value}
					user_act = {"inform": content}
				elif response_policy == "inform_major_key+request":
					user_act = {"inform": {}, "request": {}}
					# inform_major_key
					content_inform = {user_domain: {}}
					user_slot = random_pick(user_slot_prop["inform_major_key"][user_domain])
					if user_domain in ["电影", "音乐"]:
						user_value = round["sys_nl"].split("》")[0].split("《")[1]
					elif user_domain in ["天气"]:
						for value in OTLG[user_domain]["informable"][user_slot]:
							if value in round["sys_nl"]:
								user_value = value
								break
					content_inform[user_domain] = {user_slot: {user_value: "LIKE"}}
					user_act["inform"] = content_inform
					# request
					content_request = {user_domain: {}}
					while True:
						num_slots = random_pick(num_slots_prop)
						user_slots = random_sample(user_slot_prop["request"][user_domain],
						                           min(num_slots, len(OTLG[user_domain]["requestable"])))
						if sys_slot not in user_slots:
							break
					content_request[user_domain] = user_slots
					user_act["request"] = content_request
				elif response_policy == "inform_major_key+confirm":
					user_act = {"inform": {}, "confirm": {}}
					# inform_major_key
					content_inform = {user_domain: {}}
					user_slot = random_pick(user_slot_prop["inform_major_key"][user_domain])
					if user_domain in ["电影", "音乐"]:
						user_value = round["sys_nl"].split("》")[0].split("《")[1]
					elif user_domain in ["天气"]:
						for value in OTLG[user_domain]["informable"][user_slot]:
							if value in round["sys_nl"]:
								user_value = value
								break
					content_inform[user_domain].update({user_slot: {user_value: "LIKE"}})
					user_act["inform"] = content_inform
					# confirm
					content_confirm = {user_domain: {}}
					while True:
						user_slot = random_pick(user_slot_prop["confirm"][user_domain])
						if sys_slot != user_slot:
							break
					user_value = random.choice(OTLG[user_domain]["informable"][user_slot])
					content_confirm[user_domain].update({user_slot: user_value})
					user_act["confirm"] = content_confirm
			elif response_mode == "casual":
				if response_policy == "inform_slot":
					content = {user_domain: {}}
					num_slots = random_pick(num_slots_prop)
					user_slots = random_sample(user_slot_prop[response_policy][user_domain], num_slots)
					for slot in user_slots:
						content[user_domain].update({slot: {}})
						num_values = random_pick(num_values_prop)
						user_values = random.sample(OTLG[user_domain]["informable"][slot],
						                            min(num_values, len(OTLG[user_domain]["informable"][slot]) - 1))
						for user_value in user_values:
							content[user_domain][slot].update({user_value: random_pick(sentiment_prop)})
					user_act = {"inform": content}
				elif response_policy == "inform_major_key+request":
					user_act = {"inform": {}, "request": {}}
					# inform_major_key
					content_inform = {user_domain: {}}
					user_slot = random_pick(user_slot_prop["inform_major_key"][user_domain])
					user_value = random.choice(OTLG[user_domain]["informable"][user_slot])
					content_inform[user_domain] = {user_slot: {user_value: "LIKE"}}
					user_act["inform"] = content_inform
					# request
					content_request = {user_domain: {}}
					num_slots = random_pick(num_slots_prop)
					user_slots = random_sample(user_slot_prop["request"][user_domain],
					                           min(num_slots, len(OTLG[user_domain]["requestable"])))
					content_request[user_domain] = user_slots
					user_act["request"] = content_request
				elif response_policy == "request":
					content = {user_domain: {}}
					num_slots = random_pick(num_slots_prop)
					user_slots = random_sample(user_slot_prop[response_policy][user_domain],
					                           min(num_slots, len(OTLG[user_domain]["requestable"])))
					content[user_domain] = user_slots
					user_act = {"request": content}
			elif response_mode == "switch_domain":
				if response_policy == "inform_slot":
					content = {user_domain: {}}
					num_slots = random_pick(num_slots_prop)
					user_slots = random_sample(user_slot_prop[response_policy][user_domain], num_slots)
					for slot in user_slots:
						content[user_domain].update({slot: {}})
						num_values = random_pick(num_values_prop)
						user_values = random.sample(OTLG[user_domain]["informable"][slot],
						                            min(num_values, len(OTLG[user_domain]["informable"][slot]) - 1))
						for user_value in user_values:
							content[user_domain][slot].update({user_value: random_pick(sentiment_prop)})
					user_act = {"inform": content}
				elif response_policy == "inform_major_key":
					content = {user_domain: {}}
					user_slot = random_pick(user_slot_prop[response_policy][user_domain])
					user_value = random.choice(OTLG[user_domain]["informable"][user_slot])
					content[user_domain].update({user_slot: {user_value: "LIKE"}})
					user_act = {"inform": content}
				elif response_policy == "inform_major_key+request":
					user_act = {"inform": {}, "request": {}}
					# inform_major_key
					content_inform = {user_domain: {}}
					user_slot = random_pick(user_slot_prop["inform_major_key"][user_domain])
					user_value = random.choice(OTLG[user_domain]["informable"][user_slot])
					content_inform[user_domain] = {user_slot: {user_value: "LIKE"}}
					user_act["inform"] = content_inform
					# request
					content_request = {user_domain: {}}
					num_slots = random_pick(num_slots_prop)
					user_slots = random_sample(user_slot_prop["request"][user_domain],
					                           min(num_slots, len(OTLG[user_domain]["requestable"])))
					content_request[user_domain] = user_slots
					user_act["request"] = content_request
				elif response_policy == "inform_major_key+confirm":
					user_act = {"inform": {}, "confirm": {}}
					# inform_major_key
					content_inform = {user_domain: {}}
					user_slot = random_pick(user_slot_prop["inform_major_key"][user_domain])
					user_value = random.choice(OTLG[user_domain]["informable"][user_slot])
					content_inform[user_domain].update({user_slot: {user_value: "LIKE"}})
					user_act["inform"] = content_inform
					# confirm
					content_confirm = {user_domain: {}}
					user_slot = random_pick(user_slot_prop["confirm"][user_domain])
					user_value = random.choice(OTLG[user_domain]["informable"][user_slot])
					content_confirm[user_domain].update({user_slot: user_value})
					user_act["confirm"] = content_confirm
				elif response_policy == "request":
					content = {user_domain: {}}
					num_slots = random_pick(num_slots_prop)
					user_slots = random_sample(user_slot_prop[response_policy][user_domain],
					                           min(num_slots, len(OTLG[user_domain]["requestable"])))
					content[user_domain] = user_slots
					user_act = {"request": content}
		elif sys_act_type == "confirm":
			sys_domain = list(round["sys_act"][sys_act_type].keys())[0]
			sys_slot = list(round["sys_act"][sys_act_type][sys_domain].keys())[0]
			sys_value = round["sys_act"][sys_act_type][sys_domain][sys_slot]
			if response_mode == "usual":
				if response_policy == "affirm":
					content = {}
					user_act = {"affirm": content}
				elif response_policy == "deny":
					content = {}
					user_act = {"deny": content}
			elif response_mode == "casual":
				if response_policy == "inform_slot":
					content = {user_domain: {}}
					while True:
						num_slots = random_pick(num_slots_prop)
						user_slots = random_sample(user_slot_prop[response_policy][user_domain], num_slots)
						if sys_slot not in user_slots:
							break
					for slot in user_slots:
						content[user_domain].update({slot: {}})
						num_values = random_pick(num_values_prop)
						user_values = random.sample(OTLG[user_domain]["informable"][slot],
						                            min(num_values, len(OTLG[user_domain]["informable"][slot]) - 1))
						for user_value in user_values:
							content[user_domain][slot].update({user_value: random_pick(sentiment_prop)})
					user_act = {"inform": content}
				elif response_policy == "inform_major_key":
					content = {user_domain: {}}
					user_slot = random_pick(user_slot_prop[response_policy][user_domain])
					user_value = random.choice(OTLG[user_domain]["informable"][user_slot])
					content[user_domain].update({user_slot: {user_value: "LIKE"}})
					user_act = {"inform": content}
				elif response_policy == "inform_major_key+request":
					user_act = {"inform": {}, "request": {}}
					# inform_major_key
					content_inform = {user_domain: {}}
					user_slot = random_pick(user_slot_prop["inform_major_key"][user_domain])
					user_value = random.choice(OTLG[user_domain]["informable"][user_slot])
					content_inform[user_domain] = {user_slot: {user_value: "LIKE"}}
					user_act["inform"] = content_inform
					# request
					content_request = {user_domain: {}}
					num_slots = random_pick(num_slots_prop)
					user_slots = random_sample(user_slot_prop["request"][user_domain],
					                           min(num_slots, len(OTLG[user_domain]["requestable"])))
					content_request[user_domain] = user_slots
					user_act["request"] = content_request
				elif response_policy == "inform_major_key+confirm":
					user_act = {"inform": {}, "confirm": {}}
					# inform_major_key
					content_inform = {user_domain: {}}
					user_slot = random_pick(user_slot_prop["inform_major_key"][user_domain])
					user_value = random.choice(OTLG[user_domain]["informable"][user_slot])
					content_inform[user_domain].update({user_slot: {user_value: "LIKE"}})
					user_act["inform"] = content_inform
					# confirm
					content_confirm = {user_domain: {}}
					user_slot = random_pick(user_slot_prop["confirm"][user_domain])
					user_value = random.choice(OTLG[user_domain]["informable"][user_slot])
					content_confirm[user_domain].update({user_slot: user_value})
					user_act["confirm"] = content_confirm
			elif response_mode == "switch_domain":
				if response_policy == "inform_slot":
					content = {user_domain: {}}
					num_slots = random_pick(num_slots_prop)
					user_slots = random_sample(user_slot_prop[response_policy][user_domain], num_slots)
					for slot in user_slots:
						content[user_domain].update({slot: {}})
						num_values = random_pick(num_values_prop)
						user_values = random.sample(OTLG[user_domain]["informable"][slot],
						                            min(num_values, len(OTLG[user_domain]["informable"][slot]) - 1))
						for user_value in user_values:
							content[user_domain][slot].update({user_value: random_pick(sentiment_prop)})
					user_act = {"inform": content}
				elif response_policy == "inform_major_key":
					content = {user_domain: {}}
					user_slot = random_pick(user_slot_prop[response_policy][user_domain])
					user_value = random.choice(OTLG[user_domain]["informable"][user_slot])
					content[user_domain].update({user_slot: {user_value: "LIKE"}})
					user_act = {"inform": content}
				elif response_policy == "inform_major_key+request":
					user_act = {"inform": {}, "request": {}}
					# inform_major_key
					content_inform = {user_domain: {}}
					user_slot = random_pick(user_slot_prop["inform_major_key"][user_domain])
					user_value = random.choice(OTLG[user_domain]["informable"][user_slot])
					content_inform[user_domain] = {user_slot: {user_value: "LIKE"}}
					user_act["inform"] = content_inform
					# request
					content_request = {user_domain: {}}
					num_slots = random_pick(num_slots_prop)
					user_slots = random_sample(user_slot_prop["request"][user_domain],
					                           min(num_slots, len(OTLG[user_domain]["requestable"])))
					content_request[user_domain] = user_slots
					user_act["request"] = content_request
				elif response_policy == "inform_major_key+confirm":
					user_act = {"inform": {}, "confirm": {}}
					# inform_major_key
					content_inform = {user_domain: {}}
					user_slot = random_pick(user_slot_prop["inform_major_key"][user_domain])
					user_value = random.choice(OTLG[user_domain]["informable"][user_slot])
					content_inform[user_domain].update({user_slot: {user_value: "LIKE"}})
					user_act["inform"] = content_inform
					# confirm
					content_confirm = {user_domain: {}}
					user_slot = random_pick(user_slot_prop["confirm"][user_domain])
					user_value = random.choice(OTLG[user_domain]["informable"][user_slot])
					content_confirm[user_domain].update({user_slot: user_value})
					user_act["confirm"] = content_confirm
				elif response_policy == "request":
					content = {user_domain: {}}
					num_slots = random_pick(num_slots_prop)
					user_slots = random_sample(user_slot_prop[response_policy][user_domain],
					                           min(num_slots, len(OTLG[user_domain]["requestable"])))
					content[user_domain] = user_slots
					user_act = {"request": content}
		return user_act
	
	def user_nlg(sys_act_type, sys_nl, response_mode, response_policy, user_act, user_domain):
		def list2str(items):
			return '、'.join(items)
		
		def nlg_movie(sys_act_type, sys_nl, response_mode, response_policy, user_act, user_domain):
			nl = ""
			if response_policy == "inform_slot":
				if sys_act_type == "request" and response_mode == "usual":
					content = user_act["inform"][user_domain]
					user_slot = list(content.keys())[0]
					values = {"LIKE": [], "DISLIKE": []}
					for value, sentiment in content[user_slot].items():
						values[sentiment].append(value)
					if user_slot == "主演":
						if values["LIKE"] == [] and values["DISLIKE"] == []:
							nl_candidate = ["随便。", "都可以。", "谁演的都行。", "无所谓。", "都行。"]
							nl += random.choice(nl_candidate)
						elif values["DISLIKE"] == []:
							nl_candidate = ["%s。", "喜欢%s。", "喜欢看%s演的电影。", "想看%s演的电影。", "来几部%s的电影。", "有%s演的电影吗？"]
							nl += random.choice(nl_candidate) % list2str(values["LIKE"])
						else:
							for sentiment in values:
								if values[sentiment]:
									if sentiment == "LIKE":
										nl_candidate = ["喜欢%s。", "喜欢看%s演的电影。", "想看%s演的电影。", "来几部%s的电影。", "有%s演的电影吗？"]
										nl += random.choice(nl_candidate) % list2str(values[sentiment])
									elif sentiment == "DISLIKE":
										nl_candidate = ["不喜欢看%s演的电影。", "不想看%s演的电影。", "不要%s演的。", "不想看%s演的。"]
										nl += random.choice(nl_candidate) % list2str(values[sentiment])
					elif user_slot == "导演":
						if values["LIKE"] == [] and values["DISLIKE"] == []:
							nl_candidate = ["随便。", "都可以。", "谁导演的都行。", "谁拍的都行。", "无所谓。", "都行。"]
							nl += random.choice(nl_candidate)
						elif values["DISLIKE"] == []:
							nl_candidate = ["%s。", "喜欢%s。", "喜欢看%s导演的电影。", "想看%s拍的电影。", "来几部%s的电影。", "喜欢%s执导的影片"]
							nl += random.choice(nl_candidate) % list2str(values["LIKE"])
						else:
							for sentiment in values:
								if values[sentiment]:
									if sentiment == "LIKE":
										nl_candidate = ["喜欢%s。", "喜欢看%s导演的电影。", "想看%s拍的电影。", "来几部%s的电影。", "喜欢%s执导的影片"]
										nl += random.choice(nl_candidate) % list2str(values[sentiment])
									elif sentiment == "DISLIKE":
										nl_candidate = ["不喜欢看%s导演的电影。", "不想看%s拍的电影。", "不要%s拍的。", "不想看%s导演的。"]
										nl += random.choice(nl_candidate) % list2str(values[sentiment])
					elif user_slot == "年代":
						if values["LIKE"] == [] and values["DISLIKE"] == []:
							nl_candidate = ["随便。", "都可以。", "啥时候的都行。", "什么年代的都行。", "无所谓。", "都行。"]
							nl += random.choice(nl_candidate)
						elif values["DISLIKE"] == []:
							values_recent = []
							for value in ["今年", "去年", "前年"]:
								if value in values[sentiment]:
									values[sentiment].remove(value)
									values_recent.append(value)
							if values_recent:
								nl_candidate = ["%s的。", "想看%s的电影。", "来几部%s的电影。"]
								nl += random.choice(nl_candidate) % list2str(values_recent)
							if values["LIKE"]:
								nl_candidate = ["%s年代的。", "喜欢%s年代的。", "喜欢看%s年代的电影。", "想看%s年代的电影。", "来几部%s年代的电影。", "有%s年代的电影吗？"]
								nl += random.choice(nl_candidate) % list2str(values["LIKE"])
						else:
							for sentiment in values:
								if values[sentiment]:
									if sentiment == "LIKE":
										values_recent = []
										for value in ["今年", "去年", "前年"]:
											if value in values[sentiment]:
												values[sentiment].remove(value)
												values_recent.append(value)
										if values_recent:
											nl_candidate = ["%s的。", "想看%s的电影。", "来几部%s的电影。"]
											nl += random.choice(nl_candidate) % list2str(values_recent)
										if values[sentiment]:
											nl_candidate = ["%s年代的。", "喜欢%s年代的。", "喜欢看%s年代的电影。", "想看%s年代的电影。",
											                "来几部%s年代的电影。", "有%s年代的电影吗？"]
											nl += random.choice(nl_candidate) % list2str(values[sentiment])
									elif sentiment == "DISLIKE":
										values_recent = []
										for value in ["今年", "去年", "前年"]:
											if value in values[sentiment]:
												values[sentiment].remove(value)
												values_recent.append(value)
										if values_recent:
											nl_candidate = ["不要%s的。", "不想看%s的电影。", "不喜欢%s的电影。"]
											nl += random.choice(nl_candidate) % list2str(values_recent)
										if values[sentiment]:
											nl_candidate = ["不要%s年代的。", "不喜欢%s年代的。", "不喜欢看%s年代的电影。", "不想看%s年代的电影。"]
											nl += random.choice(nl_candidate) % list2str(values[sentiment])
					elif user_slot == "类型":
						if values["LIKE"] == [] and values["DISLIKE"] == []:
							nl_candidate = ["随便。", "都可以。", "啥类型的都行。", "什么题材的都行。", "无所谓。", "都行。"]
							nl += random.choice(nl_candidate)
						elif values["DISLIKE"] == []:
							nl_candidate = ["%s。", "%s类型的。", "想看%s电影。", "来几部%s电影。"]
							nl += random.choice(nl_candidate) % list2str(values["LIKE"])
						else:
							for sentiment in values:
								if values[sentiment]:
									if sentiment == "LIKE":
										nl_candidate = ["想看%s电影。", "来几部%s电影。", "喜欢看%s类型的电影。", "爱看%s电影。", "有没有%s片？"]
										nl += random.choice(nl_candidate) % list2str(values[sentiment])
									elif sentiment == "DISLIKE":
										nl_candidate = ["不想看%s电影。", "不要%s电影。", "不喜欢%s电影", "不爱看%s类型的电影。"]
										nl += random.choice(nl_candidate) % list2str(values[sentiment])
					elif user_slot == "地区":
						if values["LIKE"] == [] and values["DISLIKE"] == []:
							nl_candidate = ["随便。", "都可以。", "哪里的都行。", "哪国的都行。", "哪国的电影都行。", "无所谓。", "都行。"]
							nl += random.choice(nl_candidate)
						elif values["DISLIKE"] == []:
							nl_candidate = ["%s的。", "喜欢%s电影。", "喜欢看%s的电影。", "想看%s电影。", "来几部%s的电影。"]
							nl += random.choice(nl_candidate) % list2str(values["LIKE"])
						else:
							for sentiment in values:
								if values[sentiment]:
									if sentiment == "LIKE":
										nl_candidate = ["喜欢%s电影。", "喜欢看%s的电影。", "想看%s电影。", "来几部%s的电影。"]
										nl += random.choice(nl_candidate) % list2str(values[sentiment])
									elif sentiment == "DISLIKE":
										nl_candidate = ["不喜欢%s电影。", "不喜欢看%s的电影。", "不想看%s电影。", "不要%s的电影。"]
										nl += random.choice(nl_candidate) % list2str(values[sentiment])
					elif user_slot == "资费":
						if values["LIKE"] == [] and values["DISLIKE"] == []:
							nl_candidate = ["随便。", "都可以。", "无所谓。", "都行。"]
							nl += random.choice(nl_candidate)
						elif values["DISLIKE"] == []:
							nl_candidate = ["%s的。", "%s电影。"]
							nl += random.choice(nl_candidate) % list2str(values["LIKE"])
						else:
							for sentiment in values:
								if values[sentiment]:
									if sentiment == "LIKE":
										nl_candidate = ["想看%s电影。", "来几部%s电影。"]
										nl += random.choice(nl_candidate) % list2str(values[sentiment])
									elif sentiment == "DISLIKE":
										nl_candidate = ["不想看%s电影。", "不要%s电影。"]
										nl += random.choice(nl_candidate) % list2str(values[sentiment])
				else:
					nl_candidate = ["我想看电影。", "推荐一部电影吧。"]
					nl += random.choice(nl_candidate)
					content = user_act["inform"][user_domain]
					user_slots = list(content.keys())
					user_slots.sort(key=lambda x: len(content[x]), reverse=True)
					for user_slot in user_slots:
						values = {"LIKE": [], "DISLIKE": []}
						for value, sentiment in content[user_slot].items():
							values[sentiment].append(value)
						if user_slot == "主演":
							if values["LIKE"] == [] and values["DISLIKE"] == []:
								nl_candidate = ["谁演的都行。", "谁主演的都可以。"]
								nl += random.choice(nl_candidate)
							else:
								for sentiment in values:
									if values[sentiment]:
										if sentiment == "LIKE":
											nl_candidate = ["喜欢%s。", "喜欢看%s演的电影。", "想看%s演的电影。", "来几部%s的电影。"]
											nl += random.choice(nl_candidate) % list2str(values[sentiment])
										elif sentiment == "DISLIKE":
											nl_candidate = ["不喜欢看%s演的电影。", "不想看%s演的电影。", "不要%s演的。", "不想看%s演的。"]
											nl += random.choice(nl_candidate) % list2str(values[sentiment])
						elif user_slot == "导演":
							if values["LIKE"] == [] and values["DISLIKE"] == []:
								nl_candidate = ["谁导演的都行。", "谁拍的都行。", "谁导演的都可以。", "谁拍的都可以。"]
								nl += random.choice(nl_candidate)
							else:
								for sentiment in values:
									if values[sentiment]:
										if sentiment == "LIKE":
											nl_candidate = ["喜欢%s。", "喜欢看%s导演的电影。", "想看%s拍的电影。", "来几部%s的电影。"]
											nl += random.choice(nl_candidate) % list2str(values[sentiment])
										elif sentiment == "DISLIKE":
											nl_candidate = ["不喜欢看%s导演的电影。", "不想看%s拍的电影。", "不要%s拍的。", "不想看%s导演的。"]
											nl += random.choice(nl_candidate) % list2str(values[sentiment])
						elif user_slot == "年代":
							if values["LIKE"] == [] and values["DISLIKE"] == []:
								nl_candidate = ["啥时候的电影都行。", "什么年代的都行。", "什么年代的都可以。"]
								nl += random.choice(nl_candidate)
							else:
								for sentiment in values:
									if values[sentiment]:
										if sentiment == "LIKE":
											values_recent = []
											for value in ["今年", "去年", "前年"]:
												if value in values[sentiment]:
													values[sentiment].remove(value)
													values_recent.append(value)
											if values_recent:
												nl_candidate = ["想看%s的电影。", "来几部%s的电影。"]
												nl += random.choice(nl_candidate) % list2str(values_recent)
											if values[sentiment]:
												nl_candidate = ["喜欢%s年代的。", "喜欢看%s年代的电影。", "想看%s年代的电影。",
												                "来几部%s年代的电影。"]
												nl += random.choice(nl_candidate) % list2str(values[sentiment])
										elif sentiment == "DISLIKE":
											values_recent = []
											for value in ["今年", "去年", "前年"]:
												if value in values[sentiment]:
													values[sentiment].remove(value)
													values_recent.append(value)
											if values_recent:
												nl_candidate = ["不要%s的。", "不想看%s的电影。", "不喜欢%s的电影。"]
												nl += random.choice(nl_candidate) % list2str(values_recent)
											if values[sentiment]:
												nl_candidate = ["不要%s年代的。", "不喜欢%s年代的。", "不喜欢看%s年代的电影。", "不想看%s年代的电影。"]
												nl += random.choice(nl_candidate) % list2str(values[sentiment])
						elif user_slot == "类型":
							if values["LIKE"] == [] and values["DISLIKE"] == []:
								nl_candidate = ["啥类型的都行。", "什么题材的都行。", "类型无所谓。", "什么题材的都可以。", ]
								nl += random.choice(nl_candidate)
							else:
								for sentiment in values:
									if values[sentiment]:
										if sentiment == "LIKE":
											nl_candidate = ["想看%s电影。", "来几部%s电影。", "喜欢看%s类型的电影。", "爱看%s电影。"]
											nl += random.choice(nl_candidate) % list2str(values[sentiment])
										elif sentiment == "DISLIKE":
											nl_candidate = ["不想看%s电影。", "不要%s电影。", "不喜欢%s电影。", "不爱看%s类型的电影。"]
											nl += random.choice(nl_candidate) % list2str(values[sentiment])
						elif user_slot == "地区":
							if values["LIKE"] == [] and values["DISLIKE"] == []:
								nl_candidate = ["哪个国家的都可以", "哪国的都行。", "哪国的电影都行。", "地区无所谓。"]
								nl += random.choice(nl_candidate)
							else:
								for sentiment in values:
									if values[sentiment]:
										if sentiment == "LIKE":
											nl_candidate = ["喜欢%s电影。", "喜欢看%s的电影。", "想看%s电影。", "来几部%s的电影。"]
											nl += random.choice(nl_candidate) % list2str(values[sentiment])
										elif sentiment == "DISLIKE":
											nl_candidate = ["不喜欢%s电影。", "不喜欢看%s的电影。", "不想看%s电影。", "不要%s的电影。"]
											nl += random.choice(nl_candidate) % list2str(values[sentiment])
						elif user_slot == "资费":
							if values["LIKE"] == [] and values["DISLIKE"] == []:
								nl_candidate = ["免费和付费的都行。", "免费或付费的都可以。", "免不免费无所谓。"]
								nl += random.choice(nl_candidate)
							else:
								for sentiment in values:
									if values[sentiment]:
										if sentiment == "LIKE":
											nl_candidate = ["想看%s电影。", "来几部%s电影。"]
											nl += random.choice(nl_candidate) % list2str(values[sentiment])
										elif sentiment == "DISLIKE":
											nl_candidate = ["不想看%s电影。", "不要%s电影。"]
											nl += random.choice(nl_candidate) % list2str(values[sentiment])
			elif response_policy == "inform_major_key":
				content = user_act["inform"][user_domain]
				user_slot = list(content.keys())[0]
				user_value = list(content[user_slot].keys())[0]
				if sys_act_type == "inform_some_match" and response_mode == "usual":
					nl_candidate = ["《%s》。", "我想看《%s》。", "我想看《%s》这部影片。", "我选《%s》。"]
					nl += random.choice(nl_candidate) % user_value
				else:
					nl_candidate = ["我想看《%s》。", "我想看《%s》这部电影。", "帮我找下《%s》这部影片。"]
					nl += random.choice(nl_candidate) % user_value
			elif response_policy == "implicit_request":
				content = user_act["inform"][user_domain]
				user_slot = list(content.keys())[0]
				user_value = content[user_slot]
				nl_candidate = ["《%s》呢？", "那《%s》呢？"]
				nl += random.choice(nl_candidate) % user_value
			elif response_policy == "request":
				nl += random.choice(["", "这部影片", "该电影", "我想知道该影片", "我想问问这部电影"])
				user_slots = user_act["request"][user_domain]
				for user_slot in user_slots:
					nl += random.choice(user_nl_candidate[response_policy][user_domain][user_slot])
			elif response_policy == "confirm":
				nl += random.choice(["这部影片", "该电影"])
				user_slot = list(user_act["confirm"][user_domain].keys())[0]
				user_value = user_act["confirm"][user_domain][user_slot]
				if user_value in ["八零", "九零", "零零", "一零", "八十", "九十"]:
					user_value += "年代"
				nl += random.choice(user_nl_candidate["confirm"][user_domain][user_slot]) % user_value
			elif response_policy == "inform_major_key+request":
				# inform_major_key
				content = user_act["inform"][user_domain]
				user_slot = list(content.keys())[0]
				user_value = list(content[user_slot].keys())[0]
				nl += "《%s》" % user_value
				# request
				user_slots = user_act["request"][user_domain]
				for user_slot in user_slots:
					nl += random.choice(user_nl_candidate["request"][user_domain][user_slot])
			elif response_policy == "inform_major_key+confirm":
				# inform_major_key
				content = user_act["inform"][user_domain]
				user_slot = list(content.keys())[0]
				user_value = list(content[user_slot].keys())[0]
				nl += "《%s》" % user_value
				# confirm
				user_slot = list(user_act["confirm"][user_domain].keys())[0]
				user_value = user_act["confirm"][user_domain][user_slot]
				if user_value in ["八零", "九零", "零零", "一零", "八十", "九十"]:
					user_value += "年代"
				nl += random.choice(user_nl_candidate["confirm"][user_domain][user_slot]) % user_value
			elif response_policy == "first":
				nl += random.choice(["我想看", "我选"])
				nl += random.choice(user_nl_candidate["first"][user_domain])
			elif response_policy == "first+request":
				# first
				nl += random.choice(user_nl_candidate["first"][user_domain])
				# request
				user_slots = user_act["request"][user_domain]
				for user_slot in user_slots:
					nl += random.choice(user_nl_candidate["request"][user_domain][user_slot])
			elif response_policy == "first+confirm":
				# first
				nl += random.choice(user_nl_candidate["first"][user_domain])
				# confirm
				user_slot = list(user_act["confirm"][user_domain].keys())[0]
				user_value = user_act["confirm"][user_domain][user_slot]
				if user_value in ["八零", "九零", "零零", "一零", "八十", "九十"]:
					user_value += "年代"
				nl += random.choice(user_nl_candidate["confirm"][user_domain][user_slot]) % user_value
			elif response_policy == "second":
				nl += random.choice(["我想看", "我选"])
				nl += random.choice(user_nl_candidate["second"][user_domain])
			elif response_policy == "second+request":
				# second
				nl += random.choice(user_nl_candidate["second"][user_domain])
				# request
				user_slots = user_act["request"][user_domain]
				for user_slot in user_slots:
					nl += random.choice(user_nl_candidate["request"][user_domain][user_slot])
			elif response_policy == "second+confirm":
				# second
				nl += random.choice(user_nl_candidate["second"][user_domain])
				# confirm
				user_slot = list(user_act["confirm"][user_domain].keys())[0]
				user_value = user_act["confirm"][user_domain][user_slot]
				if user_value in ["八零", "九零", "零零", "一零", "八十", "九十"]:
					user_value += "年代"
				nl += random.choice(user_nl_candidate["confirm"][user_domain][user_slot]) % user_value
			elif response_policy == "third":
				nl += random.choice(["我想看", "我选"])
				nl += random.choice(user_nl_candidate["third"][user_domain])
			elif response_policy == "third+request":
				# third
				nl += random.choice(user_nl_candidate["third"][user_domain])
				# request
				user_slots = user_act["request"][user_domain]
				for user_slot in user_slots:
					nl += random.choice(user_nl_candidate["request"][user_domain][user_slot])
			elif response_policy == "third+confirm":
				# third
				nl += random.choice(user_nl_candidate["third"][user_domain])
				# confirm
				user_slot = list(user_act["confirm"][user_domain].keys())[0]
				user_value = user_act["confirm"][user_domain][user_slot]
				if user_value in ["八零", "九零", "零零", "一零", "八十", "九十"]:
					user_value += "年代"
				nl += random.choice(user_nl_candidate["confirm"][user_domain][user_slot]) % user_value
			elif response_policy == "last":
				nl += random.choice(["我想看", "我选"])
				nl += random.choice(user_nl_candidate["last"][user_domain])
			elif response_policy == "last+request":
				# last
				nl += random.choice(user_nl_candidate["last"][user_domain])
				# request
				user_slots = user_act["request"][user_domain]
				for user_slot in user_slots:
					nl += random.choice(user_nl_candidate["request"][user_domain][user_slot])
			elif response_policy == "last+confirm":
				# last
				nl += random.choice(user_nl_candidate["last"][user_domain])
				# confirm
				user_slot = list(user_act["confirm"][user_domain].keys())[0]
				user_value = user_act["confirm"][user_domain][user_slot]
				if user_value in ["八零", "九零", "零零", "一零", "八十", "九十"]:
					user_value += "年代"
				nl += random.choice(user_nl_candidate["confirm"][user_domain][user_slot]) % user_value
			elif response_policy == "other":
				# other
				nl += random.choice(["换几部电影。", "换几部其他电影。", "我想看其他的。", "推荐几部其他的影片。", "换一批。"])
			elif response_policy == "affirm":
				if "您" in sys_nl:
					nl += random.choice(["嗯", "是的", "没错"])
				else:
					nl += random.choice(["好", "好的", "可以", "行"])
			elif response_policy == "deny":
				if "您" in sys_nl:
					nl += random.choice(["不"])
				else:
					nl += random.choice(["不要", "不好", "不"])
			return nl
		
		def nlg_music(sys_act_type, sys_nl, response_mode, response_policy, user_act, user_domain):
			nl = ""
			if response_policy == "inform_slot":
				if sys_act_type == "request" and response_mode == "usual":
					content = user_act["inform"][user_domain]
					user_slot = list(content.keys())[0]
					values = {"LIKE": [], "DISLIKE": []}
					for value, sentiment in content[user_slot].items():
						values[sentiment].append(value)
					if user_slot == "歌手":
						if values["LIKE"] == [] and values["DISLIKE"] == []:
							nl_candidate = ["随便。", "都可以。", "谁唱的都行。", "无所谓。", "都行。"]
							nl += random.choice(nl_candidate)
						elif values["DISLIKE"] == []:
							nl_candidate = ["%s。", "喜欢%s。", "喜欢%s的音乐。", "喜欢听%s唱的歌。", "想听%s唱的歌。", "来几首%s的歌。"]
							nl += random.choice(nl_candidate) % list2str(values["LIKE"])
						else:
							for sentiment in values:
								if values[sentiment]:
									if sentiment == "LIKE":
										nl_candidate = ["喜欢%s。", "喜欢%s。", "喜欢%s的音乐。", "喜欢听%s唱的歌。", "想听%s唱的歌。",
										                "来几首%s的歌。"]
										nl += random.choice(nl_candidate) % list2str(values[sentiment])
									elif sentiment == "DISLIKE":
										nl_candidate = ["不喜欢听%s的歌。", "不想听%s的电影。", "不要%s唱的。", "不想听%s唱的。"]
										nl += random.choice(nl_candidate) % list2str(values[sentiment])
					elif user_slot == "专辑":
						if values["LIKE"] == [] and values["DISLIKE"] == []:
							nl_candidate = ["随便。", "都可以。", "哪张专辑的歌都行。", "对专辑没要求。", "哪张专辑都可以。", "无所谓。", "都行。"]
							nl += random.choice(nl_candidate)
						elif values["DISLIKE"] == []:
							nl_candidate = ["%s。", "喜欢听%s等专辑。", "来几首专辑%s里的歌。"]
							nl += random.choice(nl_candidate) % list2str(values["LIKE"])
						else:
							for sentiment in values:
								if values[sentiment]:
									if sentiment == "LIKE":
										nl_candidate = ["喜欢听%s等专辑。", "来几首专辑%s里的歌。"]
										nl += random.choice(nl_candidate) % list2str(values[sentiment])
									elif sentiment == "DISLIKE":
										nl_candidate = ["不喜欢听%s等专辑。", "不想听专辑%s里的歌。", "不要%s等专辑。"]
										nl += random.choice(nl_candidate) % list2str(values[sentiment])
					elif user_slot == "年代":
						if values["LIKE"] == [] and values["DISLIKE"] == []:
							nl_candidate = ["随便。", "都可以。", "啥时候的都行。", "什么年代的都行。", "无所谓。", "都行。"]
							nl += random.choice(nl_candidate)
						elif values["DISLIKE"] == []:
							values_recent = []
							for value in ["今年", "去年", "前年"]:
								if value in values[sentiment]:
									values[sentiment].remove(value)
									values_recent.append(value)
							if values_recent:
								nl_candidate = ["%s的。", "想听%s的歌。", "来几首%s的歌。"]
								nl += random.choice(nl_candidate) % list2str(values_recent)
							if values["LIKE"]:
								nl_candidate = ["%s年代的。", "喜欢%s年代的。", "喜欢听%s年代的歌。", "想听%s年代的歌。", "来几首%s年代的歌。"]
								nl += random.choice(nl_candidate) % list2str(values["LIKE"])
						else:
							for sentiment in values:
								if values[sentiment]:
									if sentiment == "LIKE":
										values_recent = []
										for value in ["今年", "去年", "前年"]:
											if value in values[sentiment]:
												values[sentiment].remove(value)
												values_recent.append(value)
										if values_recent:
											nl_candidate = ["%s的。", "想听%s的歌。", "来几首%s的歌。", "放首%s的歌。"]
											nl += random.choice(nl_candidate) % list2str(values_recent)
										if values[sentiment]:
											nl_candidate = ["%s年代的。", "喜欢%s年代的。", "喜欢听%s年代的歌。", "想听%s年代的歌曲。",
											                "来几首%s年代的歌。"]
											nl += random.choice(nl_candidate) % list2str(values[sentiment])
									elif sentiment == "DISLIKE":
										values_recent = []
										for value in ["今年", "去年", "前年"]:
											if value in values[sentiment]:
												values[sentiment].remove(value)
												values_recent.append(value)
										if values_recent:
											nl_candidate = ["不要%s的。", "不想听%s出的歌。"]
											nl += random.choice(nl_candidate) % list2str(values_recent)
										if values[sentiment]:
											nl_candidate = ["不要%s年代的。", "不喜欢%s年代的。", "不喜欢听%s年代的音乐。", "不想听%s年代的歌曲。"]
											nl += random.choice(nl_candidate) % list2str(values[sentiment])
					elif user_slot == "曲风":
						if values["LIKE"] == [] and values["DISLIKE"] == []:
							nl_candidate = ["随便。", "都可以。", "啥类型的都行。", "什么风格的都行。", "无所谓。", "都行。"]
							nl += random.choice(nl_candidate)
						elif values["DISLIKE"] == []:
							nl_candidate = ["%s。", "%s类型的。", "想听%s等曲风的歌。", "喜欢曲风是%s的音乐。", "来几首%s的音乐。"]
							nl += random.choice(nl_candidate) % list2str(values["LIKE"])
						else:
							for sentiment in values:
								if values[sentiment]:
									if sentiment == "LIKE":
										nl_candidate = ["爱听%s类型的歌。", "想听%s等曲风的歌。", "喜欢曲风是%s的音乐。", "来几首%s的音乐。"]
										nl += random.choice(nl_candidate) % list2str(values[sentiment])
									elif sentiment == "DISLIKE":
										nl_candidate = ["不爱听%s类型的歌。", "不想听%s等曲风的歌。", "不喜欢曲风是%s的音乐。", "不要%s的音乐。"]
										nl += random.choice(nl_candidate) % list2str(values[sentiment])
				else:
					nl_candidate = ["我想听歌。", "我想听音乐。", "放首歌听听。", "推荐一首歌吧。"]
					nl += random.choice(nl_candidate)
					content = user_act["inform"][user_domain]
					user_slots = list(content.keys())
					user_slots.sort(key=lambda x: len(content[x]), reverse=True)
					for user_slot in user_slots:
						values = {"LIKE": [], "DISLIKE": []}
						for value, sentiment in content[user_slot].items():
							values[sentiment].append(value)
						if user_slot == "歌手":
							if values["LIKE"] == [] and values["DISLIKE"] == []:
								nl_candidate = ["谁唱的都行。", "谁的歌都可以。", "谁唱的无所谓。"]
								nl += random.choice(nl_candidate)
							else:
								for sentiment in values:
									if values[sentiment]:
										if sentiment == "LIKE":
											nl_candidate = ["喜欢%s。", "喜欢%s。", "喜欢%s的音乐。", "喜欢听%s唱的歌。", "想听%s唱的歌。",
											                "来几首%s的歌。"]
											nl += random.choice(nl_candidate) % list2str(values[sentiment])
										elif sentiment == "DISLIKE":
											nl_candidate = ["不喜欢听%s的歌。", "不想听%s的电影。", "不要%s唱的。", "不想听%s唱的。"]
											nl += random.choice(nl_candidate) % list2str(values[sentiment])
						elif user_slot == "专辑":
							if values["LIKE"] == [] and values["DISLIKE"] == []:
								nl_candidate = ["哪张专辑的歌都行。", "对专辑没要求。", "哪张专辑都可以。"]
								nl += random.choice(nl_candidate)
							else:
								for sentiment in values:
									if values[sentiment]:
										if sentiment == "LIKE":
											nl_candidate = ["喜欢听%s等专辑。", "来几首专辑%s里的歌。"]
											nl += random.choice(nl_candidate) % list2str(values[sentiment])
										elif sentiment == "DISLIKE":
											nl_candidate = ["不喜欢听%s等专辑。", "不想听专辑%s里的歌。", "不要%s等专辑。"]
											nl += random.choice(nl_candidate) % list2str(values[sentiment])
						elif user_slot == "年代":
							if values["LIKE"] == [] and values["DISLIKE"] == []:
								nl_candidate = ["啥年代的歌都行。", "什么年代的都行。", "什么年代的音乐都可以。"]
								nl += random.choice(nl_candidate)
							else:
								for sentiment in values:
									if values[sentiment]:
										if sentiment == "LIKE":
											values_recent = []
											for value in ["今年", "去年", "前年"]:
												if value in values[sentiment]:
													values[sentiment].remove(value)
													values_recent.append(value)
											if values_recent:
												nl_candidate = ["想听%s的歌。", "来几首%s的歌。", "放首%s的歌。"]
												nl += random.choice(nl_candidate) % list2str(values_recent)
											if values[sentiment]:
												nl_candidate = ["喜欢%s年代的。", "喜欢听%s年代的歌。", "想听%s年代的歌曲。",
												                "来几首%s年代的歌。"]
												nl += random.choice(nl_candidate) % list2str(values[sentiment])
										elif sentiment == "DISLIKE":
											values_recent = []
											for value in ["今年", "去年", "前年"]:
												if value in values[sentiment]:
													values[sentiment].remove(value)
													values_recent.append(value)
											if values_recent:
												nl_candidate = ["不要%s的。", "不想听%s出的歌。"]
												nl += random.choice(nl_candidate) % list2str(values_recent)
											if values[sentiment]:
												nl_candidate = ["不要%s年代的。", "不喜欢%s年代的。", "不喜欢听%s年代的音乐。", "不想听%s年代的歌曲。"]
												nl += random.choice(nl_candidate) % list2str(values[sentiment])
						elif user_slot == "曲风":
							if values["LIKE"] == [] and values["DISLIKE"] == []:
								nl_candidate = ["啥类型的都行。", "什么曲风的都行。", "曲风无所谓。", "什么风格的都可以。"]
								nl += random.choice(nl_candidate)
							else:
								for sentiment in values:
									if values[sentiment]:
										if sentiment == "LIKE":
											nl_candidate = ["爱听%s类型的歌。", "想听%s等曲风的歌。", "喜欢曲风是%s的音乐。", "来几首%s的音乐。"]
											nl += random.choice(nl_candidate) % list2str(values[sentiment])
										elif sentiment == "DISLIKE":
											nl_candidate = ["不爱听%s类型的歌。", "不想听%s等曲风的歌。", "不喜欢曲风是%s的音乐。", "不要%s的音乐。"]
											nl += random.choice(nl_candidate) % list2str(values[sentiment])
			elif response_policy == "inform_major_key":
				content = user_act["inform"][user_domain]
				user_slot = list(content.keys())[0]
				user_value = list(content[user_slot].keys())[0]
				if sys_act_type == "inform_some_match" and response_mode == "usual":
					nl_candidate = ["《%s》。", "我想听《%s》。", "我选《%s》。", "放一首《%s》听听。"]
					nl += random.choice(nl_candidate) % user_value
				else:
					nl_candidate = ["我想听《%s》。", "我想听《%s》这首歌。", "来首《%s》。", "放一首《%s》听听。"]
					nl += random.choice(nl_candidate) % user_value
			elif response_policy == "implicit_request":
				content = user_act["inform"][user_domain]
				user_slot = list(content.keys())[0]
				values = content[user_slot]
				nl_candidate = ["《%s》呢？", "那《%s》呢？"]
				nl += random.choice(nl_candidate) % values
			elif response_policy == "request":
				nl += random.choice(["", "这首歌", "这首音乐", "我想知道这首歌", "我想问问这首歌"])
				user_slots = user_act["request"][user_domain]
				for user_slot in user_slots:
					nl += random.choice(user_nl_candidate[response_policy][user_domain][user_slot])
			elif response_policy == "confirm":
				nl += random.choice(["这首歌", "这首音乐"])
				user_slot = list(user_act["confirm"][user_domain].keys())[0]
				user_value = user_act["confirm"][user_domain][user_slot]
				if user_value in ["八零", "九零", "零零", "一零", "八十", "九十"]:
					user_value += "年代"
				nl += random.choice(user_nl_candidate["confirm"][user_domain][user_slot]) % user_value
			elif response_policy == "inform_major_key+request":
				# inform_major_key
				content = user_act["inform"][user_domain]
				user_slot = list(content.keys())[0]
				user_value = list(content[user_slot].keys())[0]
				nl += "《%s》" % user_value
				# request
				user_slots = user_act["request"][user_domain]
				for user_slot in user_slots:
					nl += random.choice(user_nl_candidate["request"][user_domain][user_slot])
			elif response_policy == "inform_major_key+confirm":
				# inform_major_key
				content = user_act["inform"][user_domain]
				user_slot = list(content.keys())[0]
				user_value = list(content[user_slot].keys())[0]
				nl += "《%s》" % user_value
				# confirm
				user_slot = list(user_act["confirm"][user_domain].keys())[0]
				user_value = user_act["confirm"][user_domain][user_slot]
				if user_value in ["八零", "九零", "零零", "一零", "八十", "九十"]:
					user_value += "年代"
				nl += random.choice(user_nl_candidate["confirm"][user_domain][user_slot]) % user_value
			elif response_policy == "first":
				nl += random.choice(["我想听", "我选"])
				nl += random.choice(user_nl_candidate["first"][user_domain])
			elif response_policy == "first+request":
				# first
				nl += random.choice(user_nl_candidate["first"][user_domain])
				# request
				user_slots = user_act["request"][user_domain]
				for user_slot in user_slots:
					nl += random.choice(user_nl_candidate["request"][user_domain][user_slot])
			elif response_policy == "first+confirm":
				# first
				nl += random.choice(user_nl_candidate["first"][user_domain])
				# confirm
				user_slot = list(user_act["confirm"][user_domain].keys())[0]
				user_value = user_act["confirm"][user_domain][user_slot]
				if user_value in ["八零", "九零", "零零", "一零", "八十", "九十"]:
					user_value += "年代"
				nl += random.choice(user_nl_candidate["confirm"][user_domain][user_slot]) % user_value
			elif response_policy == "second":
				nl += random.choice(["我想听", "我选"])
				nl += random.choice(user_nl_candidate["second"][user_domain])
			elif response_policy == "second+request":
				# second
				nl += random.choice(user_nl_candidate["second"][user_domain])
				# request
				user_slots = user_act["request"][user_domain]
				for user_slot in user_slots:
					nl += random.choice(user_nl_candidate["request"][user_domain][user_slot])
			elif response_policy == "second+confirm":
				# second
				nl += random.choice(user_nl_candidate["second"][user_domain])
				# confirm
				user_slot = list(user_act["confirm"][user_domain].keys())[0]
				user_value = user_act["confirm"][user_domain][user_slot]
				if user_value in ["八零", "九零", "零零", "一零", "八十", "九十"]:
					user_value += "年代"
				nl += random.choice(user_nl_candidate["confirm"][user_domain][user_slot]) % user_value
			elif response_policy == "third":
				nl += random.choice(["我想听", "我选"])
				nl += random.choice(user_nl_candidate["third"][user_domain])
			elif response_policy == "third+request":
				# third
				nl += random.choice(user_nl_candidate["third"][user_domain])
				# request
				user_slots = user_act["request"][user_domain]
				for user_slot in user_slots:
					nl += random.choice(user_nl_candidate["request"][user_domain][user_slot])
			elif response_policy == "third+confirm":
				# third
				nl += random.choice(user_nl_candidate["third"][user_domain])
				# confirm
				user_slot = list(user_act["confirm"][user_domain].keys())[0]
				user_value = user_act["confirm"][user_domain][user_slot]
				if user_value in ["八零", "九零", "零零", "一零", "八十", "九十"]:
					user_value += "年代"
				nl += random.choice(user_nl_candidate["confirm"][user_domain][user_slot]) % user_value
			elif response_policy == "last":
				nl += random.choice(["我想听", "我选"])
				nl += random.choice(user_nl_candidate["last"][user_domain])
			elif response_policy == "last+request":
				# last
				nl += random.choice(user_nl_candidate["last"][user_domain])
				# request
				user_slots = user_act["request"][user_domain]
				for user_slot in user_slots:
					nl += random.choice(user_nl_candidate["request"][user_domain][user_slot])
			elif response_policy == "last+confirm":
				# last
				nl += random.choice(user_nl_candidate["last"][user_domain])
				# confirm
				user_slot = list(user_act["confirm"][user_domain].keys())[0]
				user_value = user_act["confirm"][user_domain][user_slot]
				if user_value in ["八零", "九零", "零零", "一零", "八十", "九十"]:
					user_value += "年代"
				nl += random.choice(user_nl_candidate["confirm"][user_domain][user_slot]) % user_value
			elif response_policy == "other":
				# other
				nl += random.choice(["换几首歌。", "换几首其他音乐。", "我想听其他的。", "推荐几首其他的歌。", "换一批。"])
			elif response_policy == "affirm":
				if "您" in sys_nl:
					nl += random.choice(["嗯", "是的", "没错"])
				else:
					nl += random.choice(["好", "好的", "可以", "行"])
			elif response_policy == "deny":
				if "您" in sys_nl:
					nl += random.choice(["不"])
				else:
					nl += random.choice(["不要", "不好", "不"])
			return nl
		
		def nlg_weather(sys_act_type, sys_nl, response_mode, response_policy, user_act, user_domain):
			nl = ""
			if response_policy == "implicit_request":
				content = user_act["inform"][user_domain]
				user_slot = list(content.keys())[0]
				values = content[user_slot]
				nl_candidate = ["%s呢？", "那%s呢？"]
				nl += random.choice(nl_candidate) % values
			elif response_policy == "inform_major_key+request":
				# inform_major_key
				content = user_act["inform"][user_domain]
				user_slot = list(content.keys())[0]
				user_value = list(content[user_slot].keys())[0]
				nl += user_value
				# request
				user_slots = user_act["request"][user_domain]
				for user_slot in user_slots:
					nl += random.choice(user_nl_candidate["request"][user_domain][user_slot])
			return nl
		
		def nlg_time(sys_act_type, sys_nl, response_mode, response_policy, user_act, user_domain):
			nl = ""
			if response_policy == "request":
				user_slots = user_act["request"][user_domain]
				for user_slot in user_slots:
					nl += random.choice(user_nl_candidate[response_policy][user_domain][user_slot])
			return nl
		
		nl = ""
		nlg_fun = {"电影": nlg_movie, "音乐": nlg_music, "天气": nlg_weather, "时间": nlg_time}
		nl += nlg_fun[user_domain](sys_act_type, sys_nl, response_mode, response_policy, user_act, user_domain)
		return nl
	
	def task_desc_gen(sys_domain, response_mode, response_policy, user_act, user_domain):
		def list2str(items):
			return '、'.join(items)
		
		def desc_gen_movie(response_policy, user_act, user_domain, desc):
			if response_policy == "inform_slot":
				content = user_act["inform"][user_domain]
				user_slots = list(content.keys())
				desc["回复内容"] = "告诉对话系统您对于电影的[%s]有哪些要求。" % list2str(user_slots)
				desc["注"] = "√表示积极态度，×表示消极态度，〇表示无特殊要求"
				keyword = []
				for user_slot in user_slots:
					values = {"LIKE": [], "DISLIKE": []}
					for value, sentiment in content[user_slot].items():
						values[sentiment].append(value)
					if not values["LIKE"] and not values["DISLIKE"]:
						keyword.append("%s：〇" % user_slot)
					else:
						desc_slot = "%s：" % user_slot
						if values["LIKE"]:
							desc_slot += "√【%s】\t" % list2str(values["LIKE"])
						if values["DISLIKE"]:
							desc_slot += "×【%s】\t" % list2str(values["DISLIKE"])
						keyword.append(desc_slot)
				desc["关键词"] = "\n\t\t".join(keyword)
			elif response_policy == "inform_major_key":
				content = user_act["inform"][user_domain]
				user_slot = list(content.keys())[0]
				user_value = list(content[user_slot].keys())[0]
				desc["回复内容"] += "告诉对话系统您想看某部影片"
				desc["关键词"] += "%s:【%s】" % (user_slot, user_value)
			elif response_policy == "implicit_request":
				content = user_act["inform"][user_domain]
				user_slot = list(content.keys())[0]
				user_value = content[user_slot]
				desc["回复内容"] = "对话系统告知您某电影的某个属性后，您询问对话系统另一电影的同一属性。"
				desc["注"] = "不要提及该属性。"
				desc["关键词"] = "%s:【%s】" % (user_slot, user_value)
			elif response_policy == "request":
				user_slots = user_act["request"][user_domain]
				desc["回复内容"] = "询问对话系统某部影片的一些属性。"
				desc["注"] = "不要提及具体的影片名字。"
				desc["关键词"] = "属性【%s】" % list2str(user_slots)
			elif response_policy == "confirm":
				user_slot = list(user_act["confirm"][user_domain].keys())[0]
				user_value = user_act["confirm"][user_domain][user_slot]
				if user_value in ["八零", "九零", "零零", "一零", "八十", "九十"]:
					user_value += "年代"
				desc["回复内容"] = "向对话系统确认某影片的某属性是否是某个值。"
				desc["注"] = "不要提及具体的影片名字"
				desc["关键词"] = "%s:【%s】" % (user_slot, user_value)
			elif response_policy == "inform_major_key+request":
				# inform_major_key
				content = user_act["inform"][user_domain]
				user_slot = list(content.keys())[0]
				user_value = list(content[user_slot].keys())[0]
				desc["回复内容"] = "询问对话系统某部影片的一些属性。"
				desc["关键词"] = "%s:【%s】\t" % (user_slot, user_value)
				# request
				user_slots = user_act["request"][user_domain]
				desc["关键词"] += "属性:【%s】" % list2str(user_slots)
			elif response_policy == "inform_major_key+confirm":
				# inform_major_key
				content = user_act["inform"][user_domain]
				user_slot = list(content.keys())[0]
				user_value = list(content[user_slot].keys())[0]
				desc["回复内容"] = "向对话系统确认某影片的某属性是否是某个值。"
				desc["关键词"] = "%s:【%s】\t" % (user_slot, user_value)
				# confirm
				user_slot = list(user_act["confirm"][user_domain].keys())[0]
				user_value = user_act["confirm"][user_domain][user_slot]
				if user_value in ["八零", "九零", "零零", "一零", "八十", "九十"]:
					user_value += "年代"
				desc["关键词"] += "%s:【%s】" % (user_slot, user_value)
			elif response_policy == "first":
				desc["回复内容"] = "告诉对话系统您想看某部影片。"
				desc["注"] = "不要提及具体的电影名称。"
				desc["关键词"] = "序号:【第一个】"
			elif response_policy == "first+request":
				# first
				desc["回复内容"] = "询问对话系统某部影片的一些属性。"
				desc["注"] = "不要提及具体的电影名称。"
				desc["关键词"] = "序号:【第一个】\t"
				# request
				user_slots = user_act["request"][user_domain]
				desc["关键词"] += "属性:【%s】" % list2str(user_slots)
			elif response_policy == "first+confirm":
				# first
				desc["回复内容"] = "向对话系统确认某影片的某属性是否是某个值。"
				desc["注"] = "不要提及具体的电影名称。"
				desc["关键词"] = "序号:【第一个】\t"
				# confirm
				user_slot = list(user_act["confirm"][user_domain].keys())[0]
				user_value = user_act["confirm"][user_domain][user_slot]
				if user_value in ["八零", "九零", "零零", "一零", "八十", "九十"]:
					user_value += "年代"
				desc["关键词"] += "%s:【%s】" % (user_slot, user_value)
			elif response_policy == "second":
				desc["回复内容"] = "告诉对话系统您想看某部影片。"
				desc["注"] = "不要提及具体的电影名称。"
				desc["关键词"] = "序号:【第二个】"
			elif response_policy == "second+request":
				# second
				desc["回复内容"] = "询问对话系统某部影片的一些属性。"
				desc["注"] = "不要提及具体的电影名称。"
				desc["关键词"] = "序号:【第二个】\t"
				# request
				user_slots = user_act["request"][user_domain]
				desc["关键词"] += "属性:【%s】" % list2str(user_slots)
			elif response_policy == "second+confirm":
				# second
				desc["回复内容"] = "向对话系统确认某影片的某属性是否是某个值。"
				desc["注"] = "不要提及具体的电影名称。"
				desc["关键词"] = "序号:【第二个】\t"
				# confirm
				user_slot = list(user_act["confirm"][user_domain].keys())[0]
				user_value = user_act["confirm"][user_domain][user_slot]
				if user_value in ["八零", "九零", "零零", "一零", "八十", "九十"]:
					user_value += "年代"
				desc["关键词"] += "%s:【%s】" % (user_slot, user_value)
			elif response_policy == "third":
				desc["回复内容"] = "告诉对话系统您想看某部影片。"
				desc["注"] = "不要提及具体的电影名称。"
				desc["关键词"] = "序号:【第三个】"
			elif response_policy == "third+request":
				# third
				desc["回复内容"] = "询问对话系统某部影片的一些属性。"
				desc["注"] = "不要提及具体的电影名称。"
				desc["关键词"] = "序号:【第三个】\t"
				# request
				user_slots = user_act["request"][user_domain]
				desc["关键词"] += "属性:【%s】" % list2str(user_slots)
			elif response_policy == "third+confirm":
				# third
				desc["回复内容"] = "向对话系统确认某影片的某属性是否是某个值。"
				desc["注"] = "不要提及具体的电影名称。"
				desc["关键词"] = "序号:【第三个】\t"
				# confirm
				user_slot = list(user_act["confirm"][user_domain].keys())[0]
				user_value = user_act["confirm"][user_domain][user_slot]
				if user_value in ["八零", "九零", "零零", "一零", "八十", "九十"]:
					user_value += "年代"
				desc["关键词"] += "%s:【%s】" % (user_slot, user_value)
			elif response_policy == "last":
				desc["回复内容"] = "告诉对话系统您想看某部影片。"
				desc["注"] = "不要提及具体的电影名称。"
				desc["关键词"] = "序号:【最后一个】"
			elif response_policy == "last+request":
				# last
				desc["回复内容"] = "询问对话系统某部影片的一些属性。"
				desc["注"] = "不要提及具体的电影名称。"
				desc["关键词"] = "序号:【最后一个】\t"
				# request
				user_slots = user_act["request"][user_domain]
				desc["关键词"] += "属性:【%s】" % list2str(user_slots)
			elif response_policy == "last+confirm":
				# last
				desc["回复内容"] = "向对话系统确认某影片的某属性是否是某个值。"
				desc["注"] = "不要提及具体的电影名称。"
				desc["关键词"] = "序号:【最后一个】\t"
				# confirm
				user_slot = list(user_act["confirm"][user_domain].keys())[0]
				user_value = user_act["confirm"][user_domain][user_slot]
				if user_value in ["八零", "九零", "零零", "一零", "八十", "九十"]:
					user_value += "年代"
				desc["关键词"] += "%s:【%s】" % (user_slot, user_value)
			elif response_policy == "other":
				desc["回复内容"] = "告知对话系统想看其他电影"
				desc["关键词"] = "其他"
			elif response_policy == "affirm":
				desc["回复内容"] = "肯定或否定对话系统的问题。"
				desc["注"] = "√表示肯定，×表示否定"
				desc["关键词"] = "√"
			elif response_policy == "deny":
				desc["回复内容"] = "肯定或否定对话系统的问题。"
				desc["注"] = "√表示肯定，×表示否定"
				desc["关键词"] = "×"
			
			return desc
		
		def desc_gen_music(response_policy, user_act, user_domain, desc):
			if response_policy == "inform_slot":
				content = user_act["inform"][user_domain]
				user_slots = list(content.keys())
				desc["回复内容"] = "告知对话系统您对于音乐的[%s]有哪些要求" % list2str(user_slots)
				desc["注"] = "√表示积极态度，×表示消极态度，〇表示无特殊要求"
				keyword = []
				for user_slot in user_slots:
					values = {"LIKE": [], "DISLIKE": []}
					for value, sentiment in content[user_slot].items():
						values[sentiment].append(value)
					if not values["LIKE"] and not values["DISLIKE"]:
						keyword.append("%s：〇" % user_slot)
					else:
						desc_slot = "%s：" % user_slot
						if values["LIKE"]:
							desc_slot += "√【%s】\t" % list2str(values["LIKE"])
						if values["DISLIKE"]:
							desc_slot += "×【%s】\t" % list2str(values["DISLIKE"])
						keyword.append(desc_slot)
				desc["关键词"] = "\n\t\t".join(keyword)
			elif response_policy == "inform_major_key":
				content = user_act["inform"][user_domain]
				user_slot = list(content.keys())[0]
				user_value = list(content[user_slot].keys())[0]
				desc["回复内容"] += "告诉对话系统您想听某首歌曲"
				desc["关键词"] += "%s:【%s】" % (user_slot, user_value)
			elif response_policy == "implicit_request":
				content = user_act["inform"][user_domain]
				user_slot = list(content.keys())[0]
				user_value = content[user_slot]
				desc["回复内容"] = "对话系统告知您某首音乐的某个属性后，您询问对话系统另一首音乐的同一属性。"
				desc["注"] = "不要提及该属性。"
				desc["关键词"] = "%s:√【%s】" % (user_slot, user_value)
			elif response_policy == "request":
				user_slots = user_act["request"][user_domain]
				desc["回复内容"] = "询问对话系统某首歌曲的一些属性。"
				desc["注"] = "不要提及具体的歌曲名。"
				desc["关键词"] = "属性【%s】" % list2str(user_slots)
			elif response_policy == "confirm":
				user_slot = list(user_act["confirm"][user_domain].keys())[0]
				user_value = user_act["confirm"][user_domain][user_slot]
				if user_value in ["八零", "九零", "零零", "一零", "八十", "九十"]:
					user_value += "年代"
				desc["回复内容"] = "向对话系统确认某歌曲的某属性是否是某个值。"
				desc["注"] = "不要提及具体的歌曲名"
				desc["关键词"] = "%s:【%s】" % (user_slot, user_value)
			elif response_policy == "inform_major_key+request":
				# inform_major_key
				content = user_act["inform"][user_domain]
				user_slot = list(content.keys())[0]
				user_value = list(content[user_slot].keys())[0]
				desc["回复内容"] = "询问对话系统某首歌的一些属性。"
				desc["关键词"] = "%s:【%s】\t" % (user_slot, user_value)
				# request
				user_slots = user_act["request"][user_domain]
				desc["关键词"] += "属性:【%s】" % list2str(user_slots)
			elif response_policy == "inform_major_key+confirm":
				# inform_major_key
				content = user_act["inform"][user_domain]
				user_slot = list(content.keys())[0]
				user_value = list(content[user_slot].keys())[0]
				desc["回复内容"] = "向对话系统确认某歌曲的某属性是否是某个值。"
				desc["关键词"] = "%s:【%s】\t" % (user_slot, user_value)
				# confirm
				user_slot = list(user_act["confirm"][user_domain].keys())[0]
				user_value = user_act["confirm"][user_domain][user_slot]
				if user_value in ["八零", "九零", "零零", "一零", "八十", "九十"]:
					user_value += "年代"
				desc["关键词"] += "%s:【%s】" % (user_slot, user_value)
			elif response_policy == "first":
				desc["回复内容"] = "告诉对话系统您想听某首歌。"
				desc["注"] = "不要提及具体的歌曲名。"
				desc["关键词"] = "序号:【第一个】"
			elif response_policy == "first+request":
				# first
				desc["回复内容"] = "询问对话系统某首歌曲的一些属性。"
				desc["注"] = "不要提及具体的歌曲名。"
				desc["关键词"] = "序号:【第一个】\t"
				# request
				user_slots = user_act["request"][user_domain]
				desc["关键词"] += "属性:【%s】" % list2str(user_slots)
			elif response_policy == "first+confirm":
				# first
				desc["回复内容"] = "向对话系统确认某音乐的某属性是否是某个值。"
				desc["注"] = "不要提及具体的歌曲名。"
				desc["关键词"] = "序号:【第一个】\t"
				# confirm
				user_slot = list(user_act["confirm"][user_domain].keys())[0]
				user_value = user_act["confirm"][user_domain][user_slot]
				if user_value in ["八零", "九零", "零零", "一零", "八十", "九十"]:
					user_value += "年代"
				desc["关键词"] += "%s:【%s】" % (user_slot, user_value)
			elif response_policy == "second":
				desc["回复内容"] = "告诉对话系统您想听某首歌。"
				desc["注"] = "不要提及具体的歌曲名。"
				desc["关键词"] = "序号:【第二个】"
			elif response_policy == "second+request":
				# second
				desc["回复内容"] = "询问对话系统某首歌曲的一些属性。"
				desc["注"] = "不要提及具体的歌曲名。"
				desc["关键词"] = "序号:【第二个】\t"
				# request
				user_slots = user_act["request"][user_domain]
				desc["关键词"] += "属性:【%s】" % list2str(user_slots)
			elif response_policy == "second+confirm":
				# second
				desc["回复内容"] = "向对话系统确认某音乐的某属性是否是某个值。"
				desc["注"] = "不要提及具体的歌曲名。"
				desc["关键词"] = "序号:【第二个】\t"
				# confirm
				user_slot = list(user_act["confirm"][user_domain].keys())[0]
				user_value = user_act["confirm"][user_domain][user_slot]
				if user_value in ["八零", "九零", "零零", "一零", "八十", "九十"]:
					user_value += "年代"
				desc["关键词"] += "%s:【%s】" % (user_slot, user_value)
			elif response_policy == "third":
				desc["回复内容"] = "告诉对话系统您想听某首歌。"
				desc["注"] = "不要提及具体的歌曲名。"
				desc["关键词"] = "序号:【第三个】"
			elif response_policy == "third+request":
				# third
				desc["回复内容"] = "询问对话系统某首歌曲的一些属性。"
				desc["注"] = "不要提及具体的歌曲名。"
				desc["关键词"] = "序号:【第三个】\t"
				# request
				user_slots = user_act["request"][user_domain]
				desc["关键词"] += "属性:【%s】" % list2str(user_slots)
			elif response_policy == "third+confirm":
				# third
				desc["回复内容"] = "向对话系统确认某音乐的某属性是否是某个值。"
				desc["注"] = "不要提及具体的歌曲名。"
				desc["关键词"] = "序号:【第三个】\t"
				# confirm
				user_slot = list(user_act["confirm"][user_domain].keys())[0]
				user_value = user_act["confirm"][user_domain][user_slot]
				if user_value in ["八零", "九零", "零零", "一零", "八十", "九十"]:
					user_value += "年代"
				desc["关键词"] += "%s:【%s】" % (user_slot, user_value)
			elif response_policy == "last":
				desc["回复内容"] = "告诉对话系统您想听某首歌。"
				desc["注"] = "不要提及具体的歌曲名。"
				desc["关键词"] = "序号:【最后一个】"
			elif response_policy == "last+request":
				# last
				desc["回复内容"] = "询问对话系统某首歌曲的一些属性。"
				desc["注"] = "不要提及具体的歌曲名。"
				desc["关键词"] = "序号:【最后一个】\t"
				# request
				user_slots = user_act["request"][user_domain]
				desc["关键词"] += "属性:【%s】" % list2str(user_slots)
			elif response_policy == "last+confirm":
				# last
				desc["回复内容"] = "向对话系统确认某音乐的某属性是否是某个值。"
				desc["注"] = "不要提及具体的歌曲名。"
				desc["关键词"] = "序号:【最后一个】\t"
				# confirm
				user_slot = list(user_act["confirm"][user_domain].keys())[0]
				user_value = user_act["confirm"][user_domain][user_slot]
				if user_value in ["八零", "九零", "零零", "一零", "八十", "九十"]:
					user_value += "年代"
				desc["关键词"] += "%s:【%s】" % (user_slot, user_value)
			elif response_policy == "other":
				desc["回复内容"] = "告知对话系统想听其他歌曲"
				desc["关键词"] = "其他"
			elif response_policy == "affirm":
				desc["回复内容"] = "肯定或否定对话系统的问题。"
				desc["注"] = "√表示肯定，×表示否定"
				desc["关键词"] = "√"
			elif response_policy == "deny":
				desc["回复内容"] = "肯定或否定对话系统的问题。"
				desc["注"] = "√表示肯定，×表示否定"
				desc["关键词"] = "×"
			return desc
		
		def desc_gen_weather(response_policy, user_act, user_domain, desc):
			if response_policy == "implicit_request":
				content = user_act["inform"][user_domain]
				user_slot = list(content.keys())[0]
				user_value = content[user_slot]
				if user_slot == "城市":
					desc["回复内容"] = "对话系统告知您某城市的天气信息后，您询问对话系统另一个城市的同一信息。"
					desc["注"] = "输入的话中不要包含天气信息"
					desc["关键词"] = "%s:【%s】" % (user_slot, user_value)
				elif user_slot == "时间":
					desc["回复内容"] = "对话系统告知您某时间点的天气信息后，您询问对话系统另一时间点的同一信息。"
					desc["注"] = "输入的话中不要包含天气信息"
					desc["关键词"] = "%s:【%s】" % (user_slot, user_value)
			elif response_policy == "inform_major_key+request":
				# inform_major_key
				content = user_act["inform"][user_domain]
				user_slot = list(content.keys())[0]
				user_value = list(content[user_slot].keys())[0]
				if user_slot == "城市":
					desc["回复内容"] = "询问对话系统某城市的天气信息。"
				elif user_slot == "时间":
					desc["回复内容"] = "询问对话系统某时间点的天气信息。"
				desc["关键词"] = "%s【%s】\t" % (user_slot, user_value)
				# request
				user_slots = user_act["request"][user_domain]
				desc["关键词"] += "属性【%s】" % list2str(user_slots)
			return desc
		
		def desc_gen_time(response_policy, user_act, user_domain, desc):
			if response_policy == "request":
				user_slots = user_act["request"][user_domain]
				desc["回复内容"] = "询问对话系统一些有关时间的信息。\n"
				desc["关键词"] = "%s" % list2str(user_slots)
			return desc
		
		desc = {"回复方式": "", "回复内容": "", "注": "", "关键词": ""}
		# 回复方式
		if response_mode == "usual":
			desc["回复方式"] = "常规回复"
		elif response_mode == "casual":
			desc["回复方式"] = "非常规回复"
		elif response_mode == "switch_domain":
			desc["回复方式"] = "转移话题(%s->%s)" % (sys_domain, user_domain)
		# 回复内容
		desc_gen_fun = {"电影": desc_gen_movie, "音乐": desc_gen_music, "天气": desc_gen_weather, "时间": desc_gen_time}
		desc = desc_gen_fun[user_domain](response_policy, user_act, user_domain, desc)
		return desc
	
	def response_gen(round):
		sys_act_type = list(round["sys_act"].keys())[0]
		# choose response mode
		response_mode = random_pick(user_mode_prop[sys_act_type])
		round["response_mode"] = response_mode
		
		# choose domain
		if sys_act_type == "hello":
			user_domain = random_pick(user_domain_prop)
		else:
			sys_domain = list(round["sys_act"][sys_act_type].keys())[0]
			user_domain = sys_domain
			if response_mode == "switch_domain":
				while user_domain == sys_domain:
					user_domain = random_pick(user_domain_prop)
		round["user_domain"] = user_domain
		
		# choose policy
		policy_prop = user_policy_prop[sys_act_type][response_mode][user_domain]
		response_policy = random_pick(policy_prop)
		round["response_policy"] = response_policy
		
		# generate user act
		user_act = act_gen(round, sys_act_type, response_mode, user_domain, response_policy)
		round["user_act"] = user_act
		
		# generate user natural language
		user_nl = user_nlg(sys_act_type, round["sys_nl"], response_mode, round["response_policy"], user_act,
		                   user_domain)
		round["user_nl"] = user_nl
		
		# generate task_description
		sys_domain = {} if sys_act_type == "hello" else list(round["sys_act"][sys_act_type].keys())[0]
		task_desc = task_desc_gen(sys_domain, response_mode, round["response_policy"],
		                          user_act, user_domain)
		round["task_description"] = task_desc
		return round
	
	rounds = list(map(response_gen, rounds_sys))
	
	return rounds


def save_excel(filename, database):
	book = xlwt.Workbook(encoding="utf-8")
	sheet1 = book.add_sheet('sheet1', cell_overwrite_ok=True)
	slots = ["id", "系统语句", "回复方式", "回复内容", "关键词", "注", "回复示例", "语料1", "语料2"]
	ncols = len(slots)
	row = 0
	for col in range(ncols):
		sheet1.col(col).width = (20 * 256)  # xlwt中列宽的值表示方法：默认字体0的1/256为衡量单位。width = 256 * 20 表示20个字符宽度
		sheet1.write(0, col, slots[col])
	for data in database:
		row += 1
		for slot in slots:
			if slot == "id":
				sheet1.write(row, slots.index(slot), data["id"])
			elif slot == "系统语句":
				sheet1.write(row, slots.index(slot), data["sys_nl"])
			elif slot == "回复方式":
				sheet1.write(row, slots.index(slot), data["task_description"]["回复方式"])
			elif slot == "回复内容":
				sheet1.write(row, slots.index(slot), data["task_description"]["回复内容"])
			elif slot == "关键词":
				sheet1.write(row, slots.index(slot), data["task_description"]["关键词"])
			elif slot == "注":
				sheet1.write(row, slots.index(slot), data["task_description"]["注"])
			elif slot == "回复示例":
				sheet1.write(row, slots.index(slot), data["user_nl"])
	book.save('%s' % filename)
	print('%s 保存成功！' % filename)


if __name__ == "__main__":
	rounds_sys = rounds_sys_gen()
	with open("rounds_sys.json", 'w', encoding="utf-8") as f:
		json.dump(rounds_sys, f, ensure_ascii=False, indent=2)
	
	with open("rounds_sys.json", 'r', encoding="utf-8") as f:
		rounds_sys = json.load(f)
	rounds_dialog = user_response_gen(rounds_sys)
	with open("rounds_dialog.json", 'w', encoding="utf-8") as f:
		json.dump(rounds_dialog, f, ensure_ascii=False, indent=2)
	for a in rounds_dialog:
		print("回复方式：%s" % a["task_description"]["回复方式"])
		print("回复内容：%s" % a["task_description"]["回复内容"])
		print("关键词：%s" % a["task_description"]["关键词"])
		if a["task_description"]["注"]:
			print("注：%s" % a["task_description"]["注"])
		print()
# save_excel("多领域对话系统语料采集_10000.xls", rounds_dialog)


