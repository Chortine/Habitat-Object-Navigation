import os
import sys
gpu_id = sys.argv[1]
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
from concurrent import futures
import grpc
import logging
import time

from clip_text_pb2_grpc import add_ClipTextServiceServicer_to_server, \
    ClipTextServiceServicer
from clip_text_pb2 import text_input, text_feature
import numpy as np
import clip
import torch


class function(ClipTextServiceServicer):
    def __init__(self):
        """ CLIP"""
        clip_vit = True
        self.CLIP_device = 'cuda' #'cuda'

        # tic2 = time.time()
        if clip_vit:
            clip_model = 'ViT-B-32.pt'
        else:
            clip_model = 'RN50.pt'
        self.model, self.preprocess = clip.load(f"/cephfs/zhangtianchu/clip_model/{clip_model}",
                                                device=self.CLIP_device)
        
        self.computation_time = 0
        print("Server finished inital")


    # 这里实现我们定义的接口
    def function_operate_clip_text(self, request, context):
        tic = time.time()
        text_tensor = clip.tokenize([request.my_text]).to(self.CLIP_device)
        with torch.no_grad():
            text_features_tensor = self.model.encode_text(text_tensor)
        text_features = text_features_tensor.clone().detach().cpu().numpy()[0].tolist()
        toc = time.time()
        self.computation_time += toc-tic
        print(self.computation_time)
        print('~~~!')
        return text_feature(s_list=text_features)


def serve(port='50001', cpu_binding_num=[66, 67, 68, 69, 70, 71]):
    """ cpu binding"""
    if cpu_binding_num is not None:
        count = psutil.cpu_count()
        print('total num of cpu: ', count)

        p = psutil.Process()
        p.cpu_affinity(cpu_binding_num)
    
    """grpc server"""
    # 这里通过thread pool来并发处理server的任务
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))

    # 将对应的任务处理函数添加到rpc server中
    add_ClipTextServiceServicer_to_server(function(), server)

    # 这里使用的非安全接口，世界gRPC支持TLS/SSL安全连接，以及各种鉴权机制
    server.add_insecure_port('[::]:' + port)
    server.start()
    try:
        while True:
            time.sleep(60 * 60 * 24 * 7)
    except KeyboardInterrupt:
        server.stop(0)


if __name__ == "__main__":
    logging.basicConfig()
    serve(port=sys.argv[2], cpu_binding_num=None)  # 0~71, None for not binding