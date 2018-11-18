# -*- coding:UTF-8 -*-

from dialog.src import dialog_system_web
from socket import *

if __name__ == '__main__':
    bot = dialog_system_web.DialogSystem()
    # 对话系统ip
    HOST = '101.6.68.41'
    PORT = 3000
    BUFSIZ = 8192
    ADDR = (HOST, PORT)
    tcpSerSock = socket(AF_INET, SOCK_STREAM)
    tcpSerSock.setsockopt(SOL_SOCKET, SO_REUSEADDR, 1)
    tcpSerSock.bind(ADDR)
    tcpSerSock.listen(5)
    while True:
        print ('waiting for connection..')
        tcpCliSock, addr = tcpSerSock.accept()
        print ('...connected from:', addr)
    
        while True:
            user_utterance = tcpCliSock.recv(BUFSIZ).decode('utf-8')
            if not user_utterance:
                break
            print(user_utterance)
            system_reply = bot.update(user_utterance)[-1][-1].rstrip('。')
            print(system_reply)
            tcpCliSock.send(system_reply.encode('utf-8'))
        
            tcpCliSock.close

    tcpSerSock.close