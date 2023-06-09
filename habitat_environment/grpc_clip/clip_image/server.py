import os
import sys

gpu_id = sys.argv[1]
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
import psutil
from concurrent import futures
import grpc
import logging
import time

from clip_image_pb2_grpc import add_ClipServiceServicer_to_server, \
    ClipServiceServicer
from clip_image_pb2 import raw_rgb, rgb_feature
import numpy as np
from PIL import Image
import torch
import base64


encoder = 'CLIP'  # 'CLIP' or 'R3M'
if encoder == 'CLIP':
    import clip
elif encoder == 'R3M':
    import omegaconf
    import hydra
    import torchvision.transforms as T
    from r3m import load_r3m


class function(ClipServiceServicer):
    def __init__(self):
        """ CLIP"""
        self.device = 'cuda'  # 'cuda', 'cpu'

        # tic2 = time.time()
        if encoder == 'R3M':
            self.r3m = load_r3m("resnet50", '/cephfs/zhangtianchu/r3m_model')  # resnet18, resnet34
            # self.r3m = load_r3m("resnet50", '/home/cvpr/r3m_model')
            self.r3m.eval()
            self.r3m.to(self.device)
            self.transforms = T.Compose([T.Resize(256),
                                    T.CenterCrop(224),
                                    T.ToTensor()])  # ToTensor() divides by 255

        elif encoder == 'CLIP':
            clip_model = 'ViT-B-32.pt' # or 'RN50.pt'
            # print(clip_model)
            self.model, self.preprocess = clip.load(f"/cephfs/zhangtianchu/clip_model/{clip_model}",
                                                device=self.device)

        self.computation_time = 0
        self.count = 0

        # print("Server finished inital")

    # 这里实现我们定义的接口
    def function_operate_clip(self, request, context):
        decode_image = base64.b64decode(request.my_rgb)
        image_array = np.frombuffer(decode_image, dtype=np.uint8).reshape(request.width, request.height, -1)

        # time.sleep(5)

        
        if encoder == 'CLIP':
            image_PIL = Image.fromarray(image_array, 'RGB')
            image_tensor = self.preprocess(image_PIL).unsqueeze(0).to(self.device)
            with torch.no_grad():
                image_features_tensor = self.model.encode_image(image_tensor)
            image_features = image_features_tensor.clone().detach().cpu().numpy()[0].tolist()
        elif encoder == 'R3M':
            preprocessed_image = self.transforms(Image.fromarray(image_array.astype(np.uint8))).reshape(-1, 3, 224, 224)
            # tic = time.time()
            preprocessed_image.to(self.device)
            with torch.no_grad():
                embedding = self.r3m(preprocessed_image * 255.0)  ## R3M expects image input to be [0-255]
            image_features = embedding.clone().detach().cpu().numpy()[0].tolist()

        # toc = time.time()
        # self.computation_time += toc - tic
        # self.count += 1
        # print(self.computation_time/self.count)
        # print('~~~!')
        # image_features = [1., 2.]
        # if image_array[0,0,0] == 1:
        #     image_features = [1., 1.]
        # if image_array[0,0,0] == 0:
        #     image_features = [0., 0.]
        return rgb_feature(s_list=image_features)


def serve(port='50001', cpu_binding_num=None):
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
    add_ClipServiceServicer_to_server(function(), server)

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
