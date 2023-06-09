from __future__ import print_function
import logging

import grpc
from .clip_text_pb2 import text_input, text_feature
from .clip_text_pb2_grpc import ClipTextServiceStub

import numpy as np
import time
from PIL import Image

import base64

class clip_grpc:
    def __init__(self, ip, port=50000):
        self.channel = grpc.insecure_channel(f'{ip}:{port}')
        self.stub = ClipTextServiceStub(self.channel)

        self.temp = 0.

        # print(tic1-tic0, tic2-tic1, tic3-tic2)

    def send_and_recieve(self, goal):
        request_message = text_input(my_text=goal)
        
        tic = time.time()
        response = self.stub.function_operate_clip_text(request_message)
        toc = time.time()
        self.temp += toc-tic
        return response.s_list


if __name__ == "__main__":
    # logging.basicConfig()
    clip_test = clip_grpc()

    tic = time.time()

    for i in range(100):
        clip_feature = clip_test.send_and_recieve("chair")

    toc = time.time()
    print('~~~!')
    print('client time for 100 times: ', toc-tic)
    print('Time wait for server response: ', clip_test.temp)



