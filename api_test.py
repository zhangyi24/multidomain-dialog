#!/usr/bin/env python3
# coding=utf-8

from dialog.src import dialog_system_web

bot = dialog_system_web.DialogSystem()
while True:
	input_usr = input('输入：')
	dialog_history = bot.update(input_usr)
	print(dialog_history)
