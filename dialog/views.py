from django.shortcuts import render
from django.http import HttpResponse
from dialog.src import dialog_system_web


class DialogSystem:

    def __init__(self):
        self.turn = 0
        self.dialog_history = []

    def update(self, usr):
        if usr == "重来":
            self.dialog_history = []
            return self.dialog_history
        else:
            self.dialog_history.append(("usr:" + usr, self.response(usr)))
            return self.dialog_history

    def response(self, usr):
        return "sys：good"

dialogsystems = []
max_user_num = 3
for ii in range(max_user_num):
    dialogsystems.append(dialog_system_web.DialogSystem())
count = 0
def init_action(request):
    global count
    config = {'id': count}
    count = (count + 1) % max_user_num
    return render(request, 'init_page.html', {'config': config})

def edit_action(request, usr_id):
    title = request.POST.get('title', '重来')
    dialog_history = dialogsystems[int(usr_id)].update(title)
    return render(request, 'edit_page.html', {'data': dialog_history[1:]})
