from __future__ import print_function
import logging

import grpc
from .clip_image_pb2 import raw_rgb, rgb_feature
from .clip_image_pb2_grpc import ClipServiceStub

import numpy as np
import time
from PIL import Image

import base64

class clip_grpc:
    def __init__(self, ip, port=50000):
        self.channel = grpc.insecure_channel(f'{ip}:{port}')
        self.stub = ClipServiceStub(self.channel)

        self.temp = 0.

        # print(tic1-tic0, tic2-tic1, tic3-tic2)

    def send_and_recieve(self, image):
        image_size = image.shape
        assert image_size[2] == 3, "input should be rgb in shape: (a, b, 3)"
        width = image_size[0]
        length = image_size[1]

        base64_image = base64.b64encode(image)
        request_message = raw_rgb(my_rgb=base64_image, width=width, height=length)
        
        
        tic = time.time()
        response = self.stub.function_operate_clip(request_message)
        toc = time.time()
        self.temp += toc-tic
        return response.s_list


if __name__ == "__main__":
    # logging.basicConfig()
    clip_test = clip_grpc(ip='10.244.29.236', port=50001)

    # image = np.asarray(Image.open("elephant_small.jpg"), dtype=np.uint8)
    image = np.ones([640, 480, 3], dtype=np.uint8)

    tic = time.time()

    for i in range(500):
        clip_feature = clip_test.send_and_recieve(image)
        # print(f'===== step {i}, get_clip_featue {clip_feature}')

    toc = time.time()
    print('~~~!')
    print('client time for 100 times: ', toc-tic)
    print('Time wait for server response: ', clip_test.temp)



