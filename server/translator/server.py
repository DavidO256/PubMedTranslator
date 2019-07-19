import threading
import codecs
import socket


class TranslationServer:

    def __init__(self):
        self.requests = dict()
        self.server = socket.socket()
        self.request_thread = threading.Thread(target=self.request_loop)
#        self.translation_thread = threading.Thread(target=self.translation_loop)

    def request_loop(self):
        while True:
            client, _ = self.server.accept()
            request = cclient.recv(512)
            if request not in self.requests:
                self.requests[request] = list()
            self.requests[request].append(client)


    def send_result(self, request, result):
        for client in self.requests[request]:
            client.send(result)
