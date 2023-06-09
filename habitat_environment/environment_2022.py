import habitat
import cv2
import numpy as np
import yaml
import os
import sys
import math
import time
import copy
from collections import deque
from statistics import mean
import time
import csv
from collections import OrderedDict

current_dir = os.path.dirname(__file__)
sys.path.append(os.path.dirname(current_dir))
from habitat_environment.grpc_clip.clip_image.client import clip_grpc as grpc_image
from scipy.special import softmax
from colorama import Fore, Back, Style
from habitatsim_environment.utils.exploration_map import ExplorationMap
from habitatsim_environment.utils.fog_map import FogMapFlex

# from habitat_environment.grpc_clip.clip_text.client import clip_grpc as grpc_text

# temp record
DEPTH = {'HFOV': 79,
         'MIN_DEPTH': 0.5,
         'MAX_DEPTH': 5.,
         'POSITION': [0., 0.88, 0]}

mp3d_goal_to_cat = ["chair", "table", "picture", "cabinet", "cushion", "sofa", "bed",
                    "chest_of_drawers", "plant", "sink", "toilet", "stool", "towel",
                    "tv_monitor", "shower", "bathtub", "counter", "fireplace", "gym_equipment",
                    "seating", "clothes"]

six_cat_to_id = OrderedDict({"chair": 0, "bed": 1, "plant": 2, "toilet": 3, "tv_monitor": 4, "sofa": 5})

usual_rooms = ["bedroom",  # 卧室
               "bathroom",  # 卫生间
               "study room",  # 书房
               "balcony",  # 阳台
               "living room",  # 客厅
               "dining room",  # 餐厅
               "kitchen",  # 厨房
               "garage",  # 车库
               "games room",  # 游戏室
               "store room",  # 储物间
               "cloakroom",  # 衣帽间
               "gym",  # 健身房
               "yard",  # 庭院
               "empty room"
               ]

usual_structure = ['stairs',  # 楼梯
                   'barrier',  # 墙面
                   'corridor blocked',  # 走廊
                   'corridor navigable',
                   'exit',  # 门
                   # 'room space',  # 房间,
                   'large room',
                   # 'narrow room',
                   'outdoors',
                   'abcdefg'
                   ]

usual_objects = ['chair',  # 椅子
                 'chairs',
                 'bed',  # 床
                 'plant',  # 植物
                 'toilet',  # 厕所
                 'tv_monitor',  # 显示器
                 'sofa',  # 沙发
                 'table',  # 桌子
                 # 'fireplace',  # 壁炉
                 'gym equipment',  # 健身器材
                 'cabinet',  # 衣柜
                 'mirror',  # 镜子
                 'window',  # 窗户
                 'swimming pool',  # 泳池，
                 'empty wall'
                 ]


def print_dict(a_dict):
    print_string = ''
    for key, value in a_dict.items():
        print_string += f'{key.ljust(30, "`")}: {str(value).rjust(20, "`")}\n'
    print(print_string)


class HabitatEnv:
    def __del__(self):
        self.habitat_env.close()
        # cv2.destroyAllWindows()

    def __init__(self,
                 habitat_data_dir,
                 encoder_model_dir,
                 render=True,
                 image_encoder='',
                 image_grpc_addr=None,
                 gpu_id=0,
                 num_episode_per_scene=-1,
                 max_step_per_episode=500,
                 inference=True,
                 scene_name=None,
                 env_config_path=None,
                 on_cloud=False,
                 split='train',
                 rgb_height=480,
                 rgb_width=640,
                 depth_height=480,
                 depth_width=640,
                 dataset_version='hm3d',
                 shuffle=True,
                 action_num=4,
                 objects_to_view=None,
                 objects_to_clip=None,
                 structure_to_clip=None,
                 room_to_clip=None,
                 noisy_camera=True,
                 noisy_action=True,
                 specified_goal=None,
                 use_enlarged_val=False,
                 param_fog_map=None,
                 param_exploration_map=None
                 ):

        if objects_to_view is None:
            objects_to_view = ['other', 'chair', 'bed', 'plant', 'toilet', 'tv_monitor', 'sofa', 'table', 'stairs']
        if objects_to_clip is None:
            objects_to_clip = usual_objects
        # 视野里可以有多种物体，但只能有一种structure
        if structure_to_clip is None:
            structure_to_clip = usual_structure
        # 视野里也只能有一种room
        if room_to_clip is None:
            room_to_clip = usual_rooms
        assert dataset_version in ['hm3d', 'mp3d'], f'dataset version must be one of ["hm3d", "mp3d"]'
        self.dataset_version = dataset_version
        self.max_game_len = max_step_per_episode
        self.num_episode_per_scene = num_episode_per_scene
        self.scene_name = scene_name
        self.random_scene = True if scene_name is None else False
        self.scene_name_list = []
        self.content_path = None
        self.habitat_data_dir = habitat_data_dir
        self.encoder_model_dir = encoder_model_dir
        self.image_encoder = image_encoder
        self.inference = inference
        self.on_cloud = on_cloud
        self.split = split
        self.render = render
        self.gpu_id = gpu_id
        self.rgb_height = rgb_height
        self.rgb_width = rgb_width
        self.depth_height = depth_height
        self.depth_width = depth_width
        self.shuffle = shuffle
        self.action_num = action_num
        self.objects_to_view = objects_to_view
        self.objects_to_clip = objects_to_clip
        self.structure_to_clip = structure_to_clip
        self.room_to_clip = room_to_clip
        self.noisy_camera = noisy_camera
        self.noisy_action = noisy_action
        self.specified_goal = specified_goal
        self.use_enlarged_val = use_enlarged_val
        self.param_fog_map = param_fog_map
        self.param_exploration_map = param_exploration_map

        self.count_left_right = 0
        # self.hls_rate = np.random.randint(low=[-10, -40, -40], high=[10, 40, 40])
        self.cat_mapping_dict = self.read_category_mapping_csv()
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)  # let the environment only see this gpu

        self.grpc_client_image = None
        self.grpc_client_text = None

        if image_grpc_addr is None:
            self.init_image_encoder_local()
        else:
            self.init_grpc(image_grpc_addr)

        # self.generate_clip_feature_and_save()
        self.clip_texts_dict = self.load_clip_feature_from_file()

        self.habitat_config = habitat.get_config(
            os.path.join(current_dir, 'configs/official_challenge_objectnav2022.local.rgbd.yaml'))
        self.reconfigure_habitat()
        self.states_to_render = set([])
        self.states_as_obs = set([])
        self.states_as_info = set([])
        self.states_in_reward = set([])
        self.states_as_timing = set([])
        self.states_to_update = set([])
        if env_config_path is None:
            env_config_path = os.path.join(current_dir, 'configs/habitat_env_config.yaml')
        self.parse_config(env_config_path)

        # init something:
        self.states_dict = {key: None for key in self.states_to_update}
        self.timing_dict = {key: 0.0 for key in self.states_as_timing}
        self.episode_count = 0
        self.episode_count_per_scene = 0
        self.step_count = 0
        self.done = False
        self.move_action = np.random.choice([0, 1, 2]) if self.shuffle else 1
        self.stop_action = 0

        # fog map class
        self.my_exploration_map = ExplorationMap(param=self.param_exploration_map)
        self.my_fog_map = FogMapFlex(param_fog_map=self.param_fog_map)

        # init env and call reset
        self.habitat_env = habitat.Env(config=copy.deepcopy(self.habitat_config))
        self.reset()

    def generate_clip_feature_and_save(self):
        """
        To generate clip text feature and dump them to a file for later use
        """
        import pickle
        text_feature_dict = {}
        text_list = []
        for room in self.room_to_clip:
            text_list.append(room)
        for structure in self.structure_to_clip:
            text_list.append(structure)
        for obj in self.objects_to_clip:
            text_list.append(obj)
        text = self.clip.tokenize(text_list).to(self.clip_device)
        with self.torch.no_grad():
            text_features = self.model.encode_text(text)
            text_features = text_features / text_features.norm(dim=1, keepdim=True)
        clip_output = text_features.clone().detach().cpu().numpy()
        clip_output = clip_output.astype(np.float32)
        for obj, feature in zip(text_list, clip_output):
            text_feature_dict.update({obj: feature})
        with open(os.path.join(current_dir, 'clip_texts.pickle'), 'wb') as handle:
            pickle.dump(text_feature_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def load_clip_feature_from_file(self):
        import pickle
        with open(os.path.join(current_dir, 'clip_texts.pickle'), 'rb') as handle:
            text_feature_dict = pickle.load(handle)
        return text_feature_dict

    def reset(self):
        # reinit something:
        self.episode_count += 1
        self.episode_count_per_scene += 1
        self.step_count = 0
        self.done = False
        self.move_action = np.random.choice([0, 1, 2]) if self.shuffle else 1
        self.stop_action = 0
        self.count_left_right = 0
        self.hls_rate = np.random.randint(low=[-10, -50, -30], high=[10, 40, 40])
        self.states_dict = {key: None for key in self.states_to_update}
        self.timing_dict = {key: 0 for key in self.states_as_timing}
        if self.episode_count_per_scene > self.num_episode_per_scene and self.random_scene:
            self.change_scene(scene_name=np.random.choice(self.scene_name_list))
            self.episode_count_per_scene = 1

        """ iterate until a valid episode has been found """
        while True:
            self.habitat_env.reset()
            metrics = self.habitat_env.get_metrics()
            if math.isinf(metrics['distance_to_goal']) or math.isnan(metrics['distance_to_goal']):
                continue
            else:
                break
        self.apply_action_and_get_obs(self.move_action, self.stop_action)

        # episode-wise states
        self.goal_name = self.habitat_env.current_episode.object_category
        self.goal_id = self.habitat_observations["objectgoal"][0]
        self.init_distance_to_goal = self.habitat_metrics['distance_to_goal']
        self.check_floor = deque(maxlen=5)  # After moving forward, the heights are the same, then it ends climbing

        self.my_exploration_map.reset()
        self.my_fog_map.reset()

        self.update_state()
        obs_dict = {}
        obs_dict.update({key: self.states_dict[key] for key in self.states_as_obs})
        if self.inference:
            self.render_and_print()
        return obs_dict

    def init_image_encoder_local(self):
        if 'clip_' in self.image_encoder:
            self.torch = __import__('torch')
            # self.torch.torch.set_num_threads(32)
            self.clip = __import__('clip')
            self.PIL = __import__('PIL')
            self.clip_device = "cuda" if self.inference and not self.on_cloud else "cuda"
            clip_model = 'ViT-B-32.pt' if self.image_encoder == 'clip_vit' else 'RN50.pt' if self.image_encoder == 'clip_resnet' else ''
            clip_path = os.path.join(self.encoder_model_dir, clip_model)
            self.model, self.preprocess = self.clip.load(clip_path, device=self.clip_device)

        elif 'r3m_' in self.image_encoder:
            self.torch = __import__('torch')
            import omegaconf
            import hydra
            import torchvision.transforms as T
            from r3m import load_r3m
            self.PIL = __import__('PIL')
            self.r3m_device = "cuda" if self.inference and not self.on_cloud else "cuda"

            self.r3m_model = load_r3m("resnet50", self.encoder_model_dir)  # resnet18, resnet34
            self.r3m_model.eval()
            self.r3m_model.to(self.r3m_device)
            self.transforms = T.Compose([T.Resize(256),
                                         T.CenterCrop(224),
                                         T.ToTensor()])  # ToTensor() divides by 255

    def init_grpc(self, image_grpc_addr):
        self.grpc_client_image = None
        self.grpc_client_text = None
        if image_grpc_addr is not None:
            self.grpc_client_image = grpc_image(ip=image_grpc_addr[0], port=image_grpc_addr[1])

    def step(self, move_action, stop_action):
        self.step_count += 1
        self.move_action = move_action
        self.stop_action = stop_action
        if move_action == 1:
            self.count_left_right += 1
        elif move_action == 2:
            self.count_left_right -= 1
        self.apply_action_and_get_obs(self.move_action, self.stop_action)
        self.update_state()
        reward_dict, self.done, done_info = self.check_done_and_compute_reward()
        obs_dict = {}
        info_dict = {}
        obs_dict.update({key: self.states_dict[key] for key in self.states_as_obs})
        info_dict.update({key: self.states_dict[key] for key in self.states_as_info})
        info_dict.update({key: self.timing_dict[key] for key in self.states_as_timing})
        info_dict.update(done_info)
        self.render_and_print() if self.render else True
        return obs_dict, reward_dict, self.done, info_dict

    def wrap_action(self, action):
        if action == 'left_forward':
            self.habitat_env.step('MOVE_FORWARD')
            self.habitat_env.step('TURN_LEFT')
            obs = self.habitat_env.step('MOVE_FORWARD')
        elif action == 'right_forward':
            self.habitat_env.step('MOVE_FORWARD')
            self.habitat_env.step('TURN_RIGHT')
            obs = self.habitat_env.step('MOVE_FORWARD')
        return obs

    def add_noise_to_action(self, current_action):
        """ between last and current apply_action() """
        current_state = self.habitat_env.sim.agents[0].get_state()
        # xy噪音有穿墙问题
        # add_x = np.random.normal(0.0, 0.1)
        # add_y = np.random.normal(0.0, 0.1)
        if current_action == 0:  # forward
            add_yaw = np.random.normal(0.0, np.pi / 180)
        else:
            add_yaw = np.random.normal(0.0, np.pi / 60)
        # add_yaw = np.pi/18
        current_pos = current_state.position
        current_rot = current_state.rotation
        current_yaw = np.arctan2(current_rot.y, current_rot.w)

        # current_pos[0] += add_x
        # current_pos[1] += add_y
        current_yaw += add_yaw
        new_rot = np.quaternion(np.cos(current_yaw), 0, np.sin(current_yaw), 0)

        new_state = copy.deepcopy(current_state)
        new_state.position = current_pos
        new_state.rotation = new_rot
        self.habitat_env.sim.agents[0].set_state(new_state, reset_sensors=True, infer_sensor_states=True,
                                                 is_initial=False)

    def add_noise_to_camera(self, noise_types=None):
        image = self.habitat_observations['rgb']
        for noise_type in noise_types:
            if noise_type == 'gauss':
                row, col, ch = image.shape
                mean = 0
                var = 0.1
                sigma = var ** 0.5
                gauss = np.random.normal(mean, sigma, (row, col, ch))
                gauss = gauss.reshape(row, col, ch)
                image = image + gauss
                image = image.astype(np.uint8)
                # show = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                # cv2.imshow('noisy', show)
                # cv2.waitKey(1)
                # self.habitat_observations['rgb'] = noisy
            if noise_type == 'hls':  # hue, lightness, saturation
                # 颜色空间转换
                image = image.astype(np.float32) / 255.0
                hls_image = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
                MAX_VALUE = 100
                lightness = self.hls_rate[1]
                saturation = self.hls_rate[2]
                hue = self.hls_rate[0]
                # lightness
                hls_image[:, :, 1] = (1.0 + lightness / float(MAX_VALUE)) * hls_image[:, :, 1]
                hls_image[:, :, 1][hls_image[:, :, 1] > 1] = 1
                # saturation
                hls_image[:, :, 2] = (1.0 + saturation / float(MAX_VALUE)) * hls_image[:, :, 2]
                hls_image[:, :, 2][hls_image[:, :, 2] > 1] = 1
                # hue
                hls_image[:, :, 0] = (1.0 + hue / float(MAX_VALUE)) * hls_image[:, :, 0]
                hls_image[:, :, 0][hls_image[:, :, 0] > 360] = 360
                # HLS2BGR
                rgb_image = cv2.cvtColor(hls_image, cv2.COLOR_HLS2BGR) * 255
                image = rgb_image.astype(np.uint8)
                # show = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                # cv2.imshow('bright', show)
                # cv2.waitKey(1)
        self.habitat_observations['rgb'] = image

    def apply_action_and_get_obs(self, move_action, stop_action):
        self.habitat_action = self.interpret_action(move_action, stop_action)
        start_time = time.time()
        if self.noisy_action:
            self.add_noise_to_action(move_action)
        self.habitat_observations = self.habitat_env.step(self.habitat_action)
        self.habitat_observations['rgb'] = cv2.resize(self.habitat_observations['rgb'],
                                                      (self.rgb_width, self.rgb_height), interpolation=cv2.INTER_AREA)
        self.habitat_observations['depth'] = cv2.resize(self.habitat_observations['depth'],
                                                        (self.depth_width, self.depth_height),
                                                        interpolation=cv2.INTER_AREA)
        self.habitat_observations['depth'] = np.expand_dims(self.habitat_observations['depth'], axis=2)
        if self.dataset_version == 'mp3d':
            goal_name = mp3d_goal_to_cat[self.habitat_observations["objectgoal"][0]]
            self.habitat_observations["objectgoal"][0] = six_cat_to_id[goal_name]
        self.habitat_metrics = self.habitat_env.get_metrics()
        self.timing_dict.update({'habitat_env_time': time.time() - start_time})
        if self.noisy_camera:
            self.add_noise_to_camera(noise_types=['gauss', 'hls'])

    def update_state(self):
        # ======= deal with the observations and metrics ======= #
        if 'rgb' in self.states_to_update:
            self.states_dict.update({'rgb': self.habitat_observations['rgb']})
        if 'semantic' in self.states_to_update:
            self.states_dict.update({'semantic': self.habitat_observations['semantic']})
        if 'top_down_map' in self.states_to_update:
            self.states_dict.update({'top_down_map': self.habitat_metrics['top_down_map']['map']})
        if 'depth' in self.states_to_update:
            self.states_dict.update({'depth': self.habitat_observations['depth']})
        if 'fog_of_war_gt' in self.states_to_update:
            self.states_dict.update({'fog_of_war_gt': self.habitat_metrics['top_down_map']['fog_of_war_mask']})
        if 'robot_xy_in_map' in self.states_to_update:
            self.states_dict.update({'robot_xy_in_map': self.habitat_metrics['top_down_map']['agent_map_coord']})
        if 'robot_yaw_in_map' in self.states_to_update:
            self.states_dict.update({'robot_yaw_in_map': self.habitat_metrics['top_down_map']['agent_angle']})
        if 'collision_count' in self.states_to_update:
            self.states_dict.update({'collision_count': self.habitat_metrics['collisions']['count']})
        if 'collision' in self.states_to_update:
            self.states_dict.update({'collision': self.habitat_metrics['collisions']['is_collision']})
        if 'distance_to_goal' in self.states_to_update:
            last_distance_to_goal = self.states_dict['distance_to_goal']
            last_distance_to_goal = last_distance_to_goal if last_distance_to_goal is not None else self.init_distance_to_goal
            self.states_dict.update({'last_distance_to_goal': last_distance_to_goal})
            self.states_dict.update({'distance_to_goal': self.habitat_metrics['distance_to_goal']})
        if 'last_distance_to_goal' in self.states_to_update:
            pass  # has updated in distance_to_goal
        if 'success' in self.states_to_update:
            self.states_dict.update({'success': self.habitat_metrics['success']})
        if 'spl' in self.states_to_update:
            self.states_dict.update({'spl': self.habitat_metrics['spl']})
        if 'softspl' in self.states_to_update:
            self.states_dict.update({'softspl': self.habitat_metrics['softspl']})
        if 'robot_xy' in self.states_to_update:
            self.states_dict.update(
                {'robot_xy': [-self.habitat_observations["gps"][2], self.habitat_observations["gps"][0]]})
        if 'robot_delta_xyz' in self.states_to_update:
            if self.states_dict['robot_xyz'] is None:
                self.states_dict.update({'robot_delta_xyz': np.array([0, 0, 0], dtype=np.float32)})
            else:
                self.states_dict.update({'robot_delta_xyz': np.asarray(
                    [-self.habitat_observations["gps"][2], self.habitat_observations["gps"][0],
                     self.habitat_observations["gps"][1]], dtype=np.float32) - self.states_dict['robot_xyz']})
        if 'robot_xyz' in self.states_to_update:
            self.states_dict.update({'robot_xyz': np.array([-self.habitat_observations["gps"][2],
                                                            self.habitat_observations["gps"][0],
                                                            self.habitat_observations["gps"][1]], dtype=np.float32)})
        if 'robot_yaw' in self.states_to_update:
            self.states_dict.update({'robot_yaw': self.habitat_observations['compass'][0]})
        if 'robot_relative_xy' in self.states_to_update:
            relative_position = np.array(
                [-self.habitat_observations["gps"][2], self.habitat_observations["gps"][0]],
                dtype=np.float32)
            self.states_dict.update({'robot_relative_xy': relative_position})
        if 'robot_yaw_trigo' in self.states_to_update:
            robot_yaw_trigo = np.array([np.sin(self.habitat_observations["compass"][0]),
                                        np.cos(self.habitat_observations["compass"][0])], dtype=np.float32)
            self.states_dict.update({'robot_yaw_trigo': robot_yaw_trigo})
        # elif 'clip_goal' == state and self.step_count < 2:
        #     clip_goal = self.observations_clip_text(self.goal_name)
        #     self.states_dict.update({'clip_goal': clip_goal})
        if 'image_feature' in self.states_to_update:
            start_time = time.time() if 'clip_time' in self.states_as_timing else 0
            image_feature = self.encode_image()
            self.timing_dict.update(
                {'clip_time': time.time() - start_time}) if 'clip_time' in self.states_as_timing else True
            self.states_dict.update({'image_feature': image_feature})
        if 'onehot_goal' in self.states_to_update:
            onehot_goal = np.zeros(6, dtype=np.float32)
            onehot_goal[self.goal_id] = 1
            self.states_dict.update({'onehot_goal': onehot_goal})
        if 'goal_name' in self.states_to_update:
            self.states_dict.update({'goal_name': self.goal_name})
        if 'my_fog_of_war' in self.states_to_update:
            self.update_map()
        if 'action_onehot' in self.states_to_update:
            # last action should be the one that causes current observation
            action_onehot = np.zeros(self.action_num, dtype=np.int8)
            action_onehot[self.move_action] = 1
            last_action_onehot = self.states_dict['action_onehot'] if self.states_dict[
                                                                          'action_onehot'] is not None else action_onehot
            self.states_dict.update({'action_onehot': action_onehot})
            self.states_dict.update({'last_action_onehot': last_action_onehot})
        if 'last_action_onehot' in self.states_to_update:
            pass  # has updated in 'action_onehot'
        if 'object_in_view_percentage' in self.states_to_update:
            object_in_view_percentage = self.update_goal_in_view_percentage()
            goal_in_view = False
            for i, obj in enumerate(self.objects_to_view):
                if self.goal_name == obj and object_in_view_percentage[i] > 0.01:
                    goal_in_view = True
            self.states_dict.update({'object_in_view_percentage': object_in_view_percentage})
            self.states_dict.update({'goal_in_view': np.asarray([goal_in_view, not goal_in_view], dtype=np.int8)})
        if 'goal_in_view' in self.states_to_update:
            pass
        if 'clip_semantic' in self.states_to_update:
            assert 'image_feature' in self.states_to_update, 'prerequisite "image_feature" to update "clip_semantic"'
            objects_prob, room_prob, struct_prob = self.update_clip_semantic()
            self.states_dict.update({'clip_semantic': {'objects_prob': objects_prob,
                                                       'room_prob': room_prob,
                                                       'struct_prob': struct_prob}})
        if 'region_info' in self.states_to_update:
            current_viewed_region = self.get_current_viewed_region()
            if self.states_dict['region_info'] is None:
                self.states_dict['region_info'] = {}
            if 'current_viewed_region' in self.states_dict['region_info'].keys():
                self.states_dict['region_info']['last_viewed_region'] = self.states_dict['region_info'][
                    'current_viewed_region']
            else:
                self.states_dict['region_info']['count_in_current_region'] = 0
                self.states_dict['region_info']['last_viewed_region'] = current_viewed_region
            self.states_dict['region_info']['current_viewed_region'] = current_viewed_region

            # get true region after the update of viewed region
            if 'current_true_region' in self.states_dict['region_info'].keys():
                self.states_dict['region_info']['last_true_region'] = self.states_dict['region_info'][
                    'current_true_region']
            else:
                self.states_dict['region_info']['last_true_region'] = self.states_dict['region_info'][
                    'current_viewed_region']

            self.states_dict['region_info']['current_true_region'] = self.get_current_true_region()

            # get if current region is first visited region
            if 'visited_regions' not in self.states_dict['region_info'].keys():
                self.states_dict['region_info']['visited_regions'] = [
                    self.states_dict['region_info']['current_true_region']]
                self.states_dict['region_info']['first_visit'] = True

            # get if region changed
            if self.states_dict['region_info']['current_true_region'] != self.states_dict['region_info'][
                'last_true_region']:
                self.states_dict['region_info']['region_changed'] = True

            else:
                self.states_dict['region_info']['region_changed'] = False

            if self.states_dict['region_info']['region_changed']:
                if self.states_dict['region_info']['current_true_region'] not in self.states_dict['region_info'][
                    'visited_regions']:
                    self.states_dict['region_info']['first_visit'] = True
                    self.states_dict['region_info']['visited_regions'].append(
                        self.states_dict['region_info']['current_true_region'])
                else:
                    self.states_dict['region_info']['first_visit'] = False

            # check if goal in current true region
            if 'goal_in_current_true_region' not in self.states_dict['region_info'].keys():
                goal_in_region = self.check_goal_in_region(self.states_dict['region_info']['current_true_region'])
                self.states_dict['region_info']['goal_in_current_true_region'] = goal_in_region
                self.states_dict['region_info']['goal_in_last_true_region'] = goal_in_region

            elif self.states_dict['region_info']['region_changed']:
                goal_in_region = self.check_goal_in_region(self.states_dict['region_info']['current_true_region'])
                self.states_dict['region_info']['goal_in_last_true_region'] = self.states_dict['region_info'][
                    'goal_in_current_true_region']
                self.states_dict['region_info']['goal_in_current_true_region'] = goal_in_region

    def update_clip_semantic(self):
        # start_time = time.time()
        clip_rgb = self.states_dict['image_feature'] / np.linalg.norm(self.states_dict['image_feature'], axis=-1)
        obj_list = [self.clip_texts_dict[obj] for obj in self.objects_to_clip]
        objects_prob = softmax(100. * np.dot(obj_list, clip_rgb))
        # decode for rooms
        room_list = [self.clip_texts_dict[room] for room in self.room_to_clip]
        room_prob = softmax(100. * np.dot(room_list, clip_rgb))
        # decode for structures
        struct_list = [self.clip_texts_dict[struct] for struct in self.structure_to_clip]
        struct_prob = softmax(100. * np.dot(struct_list, clip_rgb))
        # print(f'---- time for clip semantic ---- {time.time()-start_time}')
        # for debugging:
        # rgb = self.habitat_observations["rgb"]
        # pil_image = self.PIL.Image.fromarray(rgb.astype('uint8'), 'RGB')
        # rgb_clip = self.preprocess(pil_image).unsqueeze(0).to(self.clip_device)
        # text = self.clip.tokenize(self.objects_to_clip[0]).to(self.clip_device)
        # text2 = self.clip.tokenize(self.room_to_clip).to(self.clip_device)
        # with self.torch.no_grad():
        #     # image_features = self.model.encode_image(rgb_clip)
        #     logits_per_image, logits_per_text = self.model(rgb_clip, text)
        #     probs = logits_per_image.softmax(dim=-1).cpu().numpy()
        #     logits_per_image, logits_per_text = self.model(rgb_clip, text2)
        #     probs2 = logits_per_image.softmax(dim=-1).cpu().numpy()
        # print(f'++++++++++++++++ struct probs {probs} ++++++++++++++ ')
        # print(f'++++++++++++++++ room probs {probs2} ++++++++++++++ ')
        return objects_prob, room_prob, struct_prob

    def read_category_mapping_csv(self):
        with open(os.path.join(current_dir, 'matterport_category_mappings.tsv')) as file:
            line_list = []
            tsv_file = csv.reader(file, delimiter="\t")
            for line in tsv_file:
                line_list.append(line[0].split('    '))
        line_list = np.asarray(line_list)
        mapping1 = line_list[:, [1, -1]]
        mapping2 = line_list[:, [2, -1]]
        mapping = np.append(mapping1, mapping2, axis=0)
        mapping = np.unique(mapping, axis=0)
        mapping_dict = {key[0]: mapping[i, 1] for i, key in enumerate(mapping)}
        # 修复已知错误
        mapping_dict['sofa seat'] = 'sofa'
        return mapping_dict

    def update_goal_in_view_percentage(self):
        object_in_view_semantic = np.zeros(np.shape(self.habitat_observations['semantic']), dtype=np.uint8)
        all_object_id_in_view = np.unique(self.habitat_observations['semantic'])
        object_name_list = []
        for object_id in all_object_id_in_view:
            if object_id >= len(self.habitat_env.sim.semantic_scene.objects):
                continue
            an_object = self.habitat_env.sim.semantic_scene.objects[object_id]
            # object_name = object.category.name(mapping="mpcat40")
            raw_object_name = an_object.category.name(mapping="")
            object_name = self.cat_mapping_dict[
                raw_object_name] if raw_object_name in self.cat_mapping_dict.keys() else raw_object_name
            object_name_list.append(object_name)
            if object_name in self.objects_to_view:
                object_in_view_semantic = np.where(self.habitat_observations['semantic'] == object_id,
                                                   self.objects_to_view.index(object_name), object_in_view_semantic)
        object_in_view_percentage = np.zeros(len(self.objects_to_view))
        for i in range(len(self.objects_to_view)):
            object_in_view_percentage[i] = np.count_nonzero(object_in_view_semantic == i) / np.size(
                object_in_view_semantic)
            # print(f'goal {self.objects_to_view[i]}, percentage {object_in_view_percentage[i]}')
        # cv2.imshow('goals_in_view', cv2.applyColorMap(object_in_view_semantic * 20, cv2.COLORMAP_JET))
        return object_in_view_percentage

    def get_current_viewed_region(self):
        region_id_map = np.ones(np.shape(self.habitat_observations['semantic'])) * -1
        all_object_id_in_view = np.unique(self.habitat_observations['semantic'])
        for object_id in all_object_id_in_view:
            if object_id >= len(self.habitat_env.sim.semantic_scene.objects):
                continue
            an_object = self.habitat_env.sim.semantic_scene.objects[object_id]
            raw_object_name = an_object.category.name(mapping="")
            object_name = self.cat_mapping_dict[
                raw_object_name] if raw_object_name in self.cat_mapping_dict.keys() else raw_object_name
            # if 'wall' in object_name:
            region_id = an_object.region.id
            region_id = int(region_id.strip('_'))
            region_id_map = np.where(self.habitat_observations['semantic'] == object_id, region_id, region_id_map)
        all_region_ids = np.unique(region_id_map)
        max_pixels = 0
        max_pixels_region = -1
        for idx in all_region_ids:
            if idx == -1:
                continue
            current_pixels = np.count_nonzero(region_id_map == idx)
            if current_pixels > max_pixels:
                max_pixels = current_pixels
                max_pixels_region = idx
        # cv2.imshow('region_id_map', cv2.applyColorMap(np.asarray(region_id_map, dtype=np.uint8) * 20, cv2.COLORMAP_JET))
        return int(max_pixels_region)

    def get_current_true_region(self):
        if not self.habitat_metrics['collisions']['is_collision'] and \
                self.habitat_action == 'MOVE_FORWARD' and \
                self.states_dict['region_info']['current_viewed_region'] == self.states_dict['region_info'][
            'last_viewed_region']:
            # if not collided and see this region in 3 consecutive time
            self.states_dict['region_info']['count_in_current_region'] += 1
        elif self.states_dict['region_info']['current_viewed_region'] != self.states_dict['region_info'][
            'last_viewed_region']:
            self.states_dict['region_info']['count_in_current_region'] = 0
        if self.states_dict['region_info']['count_in_current_region'] >= 4:
            # update true region to current viewed one
            current_true_region = self.states_dict['region_info']['current_viewed_region']
        else:
            current_true_region = self.states_dict['region_info']['last_true_region']
        return int(current_true_region)

    def check_goal_in_region(self, region_id):
        for region in self.habitat_env.sim.semantic_scene.regions:
            if region.id == f'_{region_id}':
                habitat_region = region
                break
        objects_in_region = habitat_region.objects
        for obj in objects_in_region:
            raw_object_name = obj.category.name(mapping="")
            object_name = self.cat_mapping_dict[
                raw_object_name] if raw_object_name in self.cat_mapping_dict.keys() else raw_object_name
            if object_name == self.goal_name:
                return True
        return False

    def render_and_print(self):
        print_dict_reset = {}
        print_dict_done = {}
        print_dict_step = {}
        for state in self.states_to_render:
            if 'rgb' == state:
                rgb = cv2.cvtColor(self.states_dict['rgb'], cv2.COLOR_BGR2RGB)
                cv2.imshow('rgb', rgb)
                cv2.waitKey(1)
            elif 'semantic' == state:
                sem = np.asarray(self.states_dict['semantic'], dtype=np.uint8)
                sem = cv2.applyColorMap(sem, cv2.COLORMAP_JET)
                cv2.imshow("semantic segmentation", sem)
                cv2.waitKey(1)
            elif 'depth' == state:
                cv2.imshow("depth", self.states_dict['depth'])
                cv2.waitKey(1)
            elif 'top_down_map' == state:
                top_down_map = np.asarray(self.states_dict['top_down_map'])
                show_top_down_map = cv2.applyColorMap(top_down_map * 50, cv2.COLORMAP_JET)
                show_top_down_map = cv2.resize(show_top_down_map, (480, 480), interpolation=cv2.INTER_AREA)
                cv2.imshow("top_down_map", show_top_down_map)
                cv2.waitKey(1)
            elif 'fog_of_war_gt' == state:
                fog_of_war_mask = np.asarray(self.states_dict['fog_of_war_gt'])
                fog_of_war_mask = cv2.resize(fog_of_war_mask, (480, 480), interpolation=cv2.INTER_AREA)
                cv2.imshow("fog_of_war_gt", fog_of_war_mask * 128)
                cv2.waitKey(1)
            elif 'my_fog_of_war' == state:
                floor = self.states_dict['floor']['true']['floor']
                print_dict_step.update({'floor': floor})
                my_fog_of_war = self.states_dict['my_fog_of_war']
                # my_fog_of_war_fine = self.states_dict['my_fog_of_war_fine']

                cv2.imshow("fine grid", np.uint8(my_fog_of_war['occupation_binary'] * 128))
                # cv2.imshow("my_fog_of_war_seen", np.uint8(my_fog_of_war['region_scanned'] * 128))
                # cv2.imshow("my_fog_of_war_Block", np.uint8(my_fog_of_war['occupation_binary'] * 128))
                # cv2.imshow("my_fog_of_war_fine", np.uint8(my_fog_of_war['fine_map'] * 128))
                # cv2.imshow("my_fog_of_war_been_to", np.uint8(my_fog_of_war['been_to'] * 128))
                # cv2.imshow("Bedroom", np.uint8(my_fog_of_war['room_info'][0] * 128 * 2))
                # cv2.imshow("Bathroom", np.uint8(my_fog_of_war['room_info'][1] * 128 * 2))
                # cv2.imshow("Living room", np.uint8(my_fog_of_war['room_info'][4] * 128 * 2))
                # cv2.imshow("Dining room", np.uint8(my_fog_of_war[8] * 128*2))
                # cv2.imshow("Kitchen", np.uint8(my_fog_of_war['room_info'][6] * 128 * 2))
                # cv2.imshow("corridor", np.uint8(my_fog_of_war['room_info'][9] * 128 * 2))

                # show exploration map
                # cv2.imshow("explore_scan",
                #            np.uint8(self.states_dict['my_explore_map']['region_scanned'] * 128))
                # cv2.imshow("explore_been", np.uint8(self.states_dict['my_explore_map']['been_to'] * 128))
            # print
            elif 'goal_name' == state:
                print_dict_reset.update({'goal_name': self.goal_name})
                print_dict_step.update({'goal_name': self.goal_name})
            elif 'success' == state:
                print_dict_done.update({'success': self.states_dict['success']})
            elif 'object_in_view_percentage' == state:
                print_string = ''
                for i, perc in enumerate(self.states_dict['object_in_view_percentage']):
                    print_string += f'{self.objects_to_view[i]}:{round(perc, 3)}; '
                    print_dict_step.update({'object_in_view_percentage': print_string})
                print_dict_step.update({'goal_in_view': self.states_dict['goal_in_view']})
            elif 'clip_semantic' == state:
                monologue = 'I am new to the world ^_^ '
                # print structure
                print_string = ''
                is_room = False
                max_struct = np.argmax(self.states_dict['clip_semantic']['struct_prob'])
                for i, perc in enumerate(self.states_dict['clip_semantic']['struct_prob']):
                    # perc = round(perc_, 3)
                    if i == max_struct:
                        monologue += f'I think I am in a {self.structure_to_clip[i]}, '
                        if 'room' in self.structure_to_clip[i]:
                            is_room = True
                        print_string += f'{Fore.RED + self.structure_to_clip[i] + Style.RESET_ALL}:{perc:.3f}; '
                    else:
                        print_string += f'{self.structure_to_clip[i]}:{perc:.3f}; '
                    if i % 4 == 0 and i > 0:
                        print_string += '\n'
                print_dict_step.update({'struct_prob': print_string})
                # print for room
                print_string = ''
                max_room = np.argmax(self.states_dict['clip_semantic']['room_prob'])
                for i, perc in enumerate(self.states_dict['clip_semantic']['room_prob']):
                    if i == max_room:
                        if is_room:
                            monologue += f'probably a {self.room_to_clip[i]}. '
                        print_string += f'{Fore.RED + self.room_to_clip[i] + Style.RESET_ALL}:{perc:.3f}; '
                    else:
                        print_string += f'{self.room_to_clip[i]}:{perc:.3f}; '
                    if i % 4 == 0 and i > 0:
                        print_string += '\n'
                print_dict_step.update({'room_prob': print_string})
                # print for objects
                print_string = ''
                max_obj = np.argmax(self.states_dict['clip_semantic']['objects_prob'])
                monologue += 'And I can see: '
                monologue_updated = False
                for i, perc in enumerate(self.states_dict['clip_semantic']['objects_prob']):
                    if perc > 0.2:
                        monologue += f'{self.objects_to_clip[i]}, '
                        monologue_updated = True
                    if i == max_obj:
                        print_string += f'{Fore.RED + self.objects_to_clip[i] + Style.RESET_ALL}:{perc:.3f}; '
                    else:
                        print_string += f'{self.objects_to_clip[i]}:{perc:.3f}; '
                    if i % 4 == 0 and i > 0:
                        print_string += '\n'
                if not monologue_updated:
                    monologue += 'nothing interesting'
                print_dict_step.update({'objects_prob': print_string})
                monologue = f'{Fore.BLUE + monologue + Style.RESET_ALL}'
                print_dict_step.update({'monologue': monologue})
            else:
                assert False, f'Cannot render the state {state}'
        if self.step_count < 1 and len(print_dict_reset) > 0:
            print(f'Episode {self.episode_count} starts. Scene {self.scene_name}. Info {print_dict_reset}')
        if self.step_count >= 1 and len(print_dict_step) > 0:
            print(f'----- Episode {self.episode_count}, step {self.step_count} -----')
            print_dict(print_dict_step)
        if self.done and len(print_dict_done) > 0:
            print(
                f'Episode {self.episode_count_per_scene}//{self.num_episode_per_scene} ends. Total steps {self.step_count}//{self.max_game_len}. Info {print_dict_done}')

    def check_done_and_compute_reward(self):
        done = False
        reward_metric_dict = {}
        info = {}
        # compute reward and also metrics

        if self.habitat_action == 'STOP' and self.states_dict["success"]:
            reward_metric_dict['success'] = 7.0
        else:
            reward_metric_dict['success'] = 0.0

        if self.habitat_action == 'STOP' and not self.states_dict["success"]:
            reward_metric_dict['false_success'] = -4.0
        else:
            reward_metric_dict['false_success'] = -0.0

        reward_metric_dict['collision'] = -0.0001 if self.states_dict['collision'] else 0.0

        if not self.habitat_action == 'STOP' and (self.step_count > self.max_game_len or self.habitat_env.episode_over):
            reward_metric_dict['max_step'] = -0.1
        else:
            reward_metric_dict['max_step'] = 0.0

        # Closer to one of the goal
        closer_to_goal = (self.states_dict['last_distance_to_goal'] - self.states_dict[
            'distance_to_goal']) / self.init_distance_to_goal
        reward_metric_dict['closer_to_goal'] = 8. * closer_to_goal

        reward_metric_dict['new_area_scan'] = self.states_dict['new_pixel_scanned'] / 1000

        # punishment for not visiting new area
        reward_metric_dict['new_area_stand'] = -0.01 if not self.states_dict['stand_at_new_place'] else 0.0

        # spl is between 0 and 1. Higher spl is better
        reward_metric_dict['spl'] = self.habitat_metrics['spl'] * 2

        if self.habitat_action == 'STOP' or self.step_count > self.max_game_len or self.habitat_env.episode_over:
            done = True

        # add done when turn too much
        if self.count_left_right > 50 or self.count_left_right < -50:
            # turn too much to one side
            done = True
            info['done_of_turn'] = False
        else:
            info['done_of_turn'] = False

        # distance to goal when done
        if not self.states_dict["success"] and done:
            reward_metric_dict['final_distance_to_goal'] = -self.states_dict['distance_to_goal'] * 0.6
        else:
            reward_metric_dict['final_distance_to_goal'] = 0.0

        return reward_metric_dict, done, info

    def interpret_action(self, move_action, stop_action):
        assert 0 <= move_action < len(self.habitat_config.TASK.POSSIBLE_ACTIONS) - 1, "invalid action"
        if stop_action:
            action = 0
        else:
            action = move_action + 1
        return self.habitat_config.TASK.POSSIBLE_ACTIONS[int(action)]

    def encode_image(self):
        if self.grpc_client_image is not None:
            rgb = np.ascontiguousarray(self.habitat_observations["rgb"])
            return np.asarray(self.grpc_client_image.send_and_recieve(rgb))
        elif 'clip_' in self.image_encoder:
            rgb = self.habitat_observations["rgb"]
            pil_image = self.PIL.Image.fromarray(rgb.astype('uint8'), 'RGB')
            rgb_clip = self.preprocess(pil_image).unsqueeze(0).to(self.clip_device)
            with self.torch.no_grad():
                image_features = self.model.encode_image(rgb_clip)
                #
                # text = self.clip.tokenize(self.objects_to_view).to(self.clip_device)
                # logits_per_image, logits_per_text = self.model(rgb_clip, text)
                # probs = logits_per_image.softmax(dim=-1).cpu().numpy()
                # print(probs)
            clip_output = image_features.clone().detach().cpu().numpy()[0]
            clip_output = clip_output.astype(np.float32)
            return clip_output
        elif 'r3m_' in self.image_encoder:
            rgb = self.habitat_observations["rgb"]
            preprocessed_image = self.transforms(self.PIL.Image.fromarray(rgb.astype(np.uint8))).reshape(-1, 3, 224,
                                                                                                         224)
            preprocessed_image.to(self.r3m_device)
            with self.torch.no_grad():
                embedding = self.r3m_model(preprocessed_image * 255.0)  ## R3M expects image input to be [0-255]
            r3m_output = embedding.clone().detach().cpu().numpy()[0]
            r3m_output = r3m_output.astype(np.float32)
            return r3m_output
        else:
            return None

    def update_map(self):
        if self.states_dict['floor'] is None:
            self.states_dict.update({'floor': {'true': {'stair': True,
                                                        'floor': None,  # in case init on stairs
                                                        'record': {}},
                                               'detected': {'stair': True, 'floor': 0}}})
            floor = None
            self.last_position = self.habitat_observations["gps"]
        else:
            self._get_true_floor()
            floor = self.states_dict['floor']['true']['floor']
            # print('==== Floor ====', floor)

        map = self.my_fog_map.step(floor=floor,
                                   depth=self.states_dict["depth"],
                                   clip_rgb=self.states_dict["image_feature"],
                                   gps=self.habitat_observations["gps"],
                                   compass=self.habitat_observations["compass"][0],
                                   collision=self.states_dict["collision"])

        self.states_dict['my_fog_of_war'] = map
        self.states_dict['visiting_new_area'] = self.my_fog_map.get_new_pixels()

        explore_map = self.my_exploration_map.step(floor=floor,
                                                   depth=self.states_dict["depth"],
                                                   gps=self.habitat_observations["gps"],
                                                   compass=self.habitat_observations["compass"][0])
        self.states_dict['my_explore_map'] = explore_map
        self.states_dict['stand_at_new_place'] = self.my_exploration_map.if_stand_at_new_place()
        self.states_dict['new_pixel_scanned'] = self.my_exploration_map.get_new_pixels()
        self.last_position = self.habitat_observations["gps"]

    def _get_true_floor(self):
        # If moving forward, append height to list
        if -0.01 < self.last_position[0] - self.habitat_observations["gps"][0] < 0.01 \
                and -0.01 < self.last_position[2] - self.habitat_observations["gps"][2] < 0.01:
            return
        self.check_floor.append(self.habitat_observations["gps"][1])

        # Check whether it is on stairs
        if len(self.check_floor) == self.check_floor.maxlen:
            self.states_dict['floor']['true']['stair'] = False
            for i in range(len(self.check_floor) - 1):
                if self.check_floor[-1] - self.check_floor[i] < -0.2 \
                        or self.check_floor[-1] - self.check_floor[i] > 0.2:
                    self.states_dict['floor']['true']['stair'] = True
                    break

        # If on stairs
        if self.states_dict['floor']['true']['stair']:
            self.states_dict['floor']['true']['floor'] = None
            return

        # On the floor
        if not self.states_dict['floor']['true']['stair']:
            # If last position is also on floor. Then floor is the same
            if self.states_dict['floor']['true']['floor'] is not None:
                return
            # If it quit climbing, check whether it enter a new floor, and record
            else:
                in_record = False
                for floor in self.states_dict['floor']['true']['record'].keys():
                    temp = self.states_dict['floor']['true']['record'][floor] - self.habitat_observations["gps"][1]
                    if -0.75 < temp < 0.75:
                        self.states_dict['floor']['true']['floor'] = floor
                        in_record = True
                        break
                if not in_record:
                    floor_list = list(self.states_dict['floor']['true']['record'].keys())
                    if len(floor_list) == 0:
                        self.states_dict['floor']['true']['record'][str(0)] = self.habitat_observations["gps"][1]
                        self.states_dict['floor']['true']['floor'] = 0
                        return

                    floor_list = [int(floor_list[i]) for i in range(len(floor_list))]
                    max_floor = max(floor_list)
                    max_height = max(self.states_dict['floor']['true']['record'].values())
                    min_floor = min(floor_list)
                    min_height = min(self.states_dict['floor']['true']['record'].values())
                    if self.habitat_observations["gps"][1] > max_height:
                        self.states_dict['floor']['true']['record'][str(max_floor + 1)] = \
                            self.habitat_observations["gps"][1]
                        self.states_dict['floor']['true']['floor'] = max_floor + 1
                    elif self.habitat_observations["gps"][1] < min_height:
                        self.states_dict['floor']['true']['record'][str(min_floor - 1)] = \
                            self.habitat_observations["gps"][1]
                        self.states_dict['floor']['true']['floor'] = min_floor - 1
                    else:
                        self.states_dict['floor']['true']['record']['100'] = self.habitat_observations["gps"][1]
                        self.states_dict['floor']['true']['floor'] = 100
                        print('=== Error in create floor record ===')

    def parse_config(self, path):
        with open(path, "r") as stream:
            try:
                config = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                assert False, exc
        self.states_to_render = set(config['states_to_render'])
        if not self.render:
            self.states_to_render = set([])
        self.states_as_obs = set(config['states_as_obs'])
        self.states_as_info = set(config['states_as_info'])
        self.states_in_reward = set(config['states_in_reward'])
        self.states_as_timing = set(config['states_as_timing'])  # states_as_timing is special
        self.states_to_update = self.states_to_render | self.states_as_obs | self.states_as_info | self.states_in_reward
        # the coherent states: if one updated, must update another
        for coherent_states in [{'distance_to_goal', 'last_distance_to_goal'},
                                {'action_onehot', 'last_action_onehot'},
                                {'my_fog_of_war', 'floor'}]:
            if len(self.states_to_update.intersection(coherent_states)) > 0:
                self.states_to_update = self.states_to_update | coherent_states

    def change_scene(self, scene_name):
        self.habitat_env.close()
        self.habitat_config.defrost()
        """ change the scene and reconfigure """
        self.scene_name = scene_name
        self.habitat_config.defrost()
        self.habitat_config.DATASET.DATA_PATH = os.path.join(self.content_path, self.scene_name)
        self.habitat_config.SEED = np.random.randint(1, 100000)
        self.habitat_config.freeze()
        self.habitat_env = habitat.Env(config=copy.deepcopy(self.habitat_config))

    def reconfigure_habitat(self):
        self.habitat_config.defrost()
        self.habitat_config.DATASET.SPLIT = self.split
        data_path = os.path.join(self.habitat_data_dir, self.habitat_config.DATASET.DATA_PATH)
        episode_dir = 'objectgoal_hm3d' if self.dataset_version == 'hm3d' else 'objectnav_mp3d_v1'
        split = self.habitat_config.DATASET.SPLIT if self.specified_goal is None else f'{self.specified_goal}_{self.split}'
        if split == 'val' and self.use_enlarged_val:
            split = 'val_enlarged'
        data_path = data_path.format(episode_dir=episode_dir, split=split)
        scene_dir = os.path.join(self.habitat_data_dir, self.habitat_config.DATASET.SCENES_DIR)
        # hm3d_dir = os.path.join(self.habitat_data_dir,
        #                         self.habitat_config.DATASET.HM3D_DIR) if self.dataset_version == 'hm3d' else ''

        self.content_path = os.path.join(os.path.dirname(data_path), 'content/')

        if self.scene_name is not None and not self.random_scene:
            # if not none, use the specific scene
            data_path = os.path.join(self.content_path, self.scene_name)
        else:
            # use the random selected scene name
            self.scene_name_list = os.listdir(self.content_path)
            self.scene_name = np.random.choice(self.scene_name_list)
            data_path = os.path.join(self.content_path, self.scene_name)

        self.habitat_config.DATASET.DATA_PATH = data_path
        self.habitat_config.DATASET.SCENES_DIR = scene_dir
        # self.habitat_config.DATASET.HM3D_DIR = hm3d_dir
        self.habitat_config.ENVIRONMENT.ITERATOR_OPTIONS.NUM_EPISODE_SAMPLE = self.num_episode_per_scene  # TODO: better strategy
        self.habitat_config.SIMULATOR.HABITAT_SIM_V0.GPU_DEVICE_ID = 0
        self.habitat_config.SEED = np.random.randint(1, 100000)
        self.habitat_config.ENVIRONMENT.MAX_EPISODE_STEPS = self.max_game_len
        self.habitat_config.ENVIRONMENT.ITERATOR_OPTIONS.SHUFFLE = self.shuffle
        self.habitat_config.TASK.GPS_SENSOR.DIMENSIONALITY = 3
        # self.habitat_config.SIMULATOR.SEMANTIC_SENSOR.WIDTH = self.rgb_width
        # self.habitat_config.SIMULATOR.SEMANTIC_SENSOR.HEIGHT = self.rgb_height
        self.habitat_config.SIMULATOR.SCENE_DATASET = os.path.join(scene_dir,
                                                                   self.habitat_config.SIMULATOR.SCENE_DATASET.format(
                                                                       version=self.dataset_version))
        self.habitat_config.freeze()


def keyboard_control():
    test_scene = 'svBbv1Pavdk.json.gz'
    # test_scene = 'NEVASPhcrxR.json.gz'
    # test_scene = '5LpN3gDmAk7.json.gz'
    # test_scene = '4ok3usBNeis.json.gz'
    # test_scene = None
    # test_scene = '1pXnuDYAj8r.json.gz'
    env = HabitatEnv(render=True,
                     habitat_data_dir='/home/tianchu/Documents/code_qy/cvpr2022',
                     encoder_model_dir='/home/tianchu/Documents/code_qy/cvpr2022/clip_model/',
                     # habitat_data_dir='/home/cvpr',
                     # encoder_model_dir='/home/cvpr/r3m_model/',
                     image_encoder='clip_vit',
                     image_grpc_addr=None,
                     scene_name=test_scene,
                     num_episode_per_scene=-1,
                     dataset_version='hm3d',
                     split='val',
                     action_num=6,
                     shuffle=True,
                     rgb_width=320,
                     rgb_height=320,
                     noisy_action=False,
                     noisy_camera=False,
                     specified_goal=None,
                     use_enlarged_val=False
                     )
    last_xy = [0, 0]
    episode_rew = 0
    accumulated_dis = 0
    last_xy = [0, 0]
    while True:
        if env.done:
            env.reset()
            episode_rew = 0
            accumulated_dis = 0
            last_xy = [0, 0]
        k = cv2.waitKey(0)
        # k = ord("w")
        move_action = 0 if k == ord("w") else 1 if k == ord("a") else 2 if k == ord("d") else 3 if k == ord(
            "q") else 4 if k == ord("e") else 0
        stop_action = 1 if k == ord("f") else 0
        obs, rew, done, info = env.step(move_action, stop_action)
        episode_rew += rew['closer_to_goal']
        distance_from_start = np.linalg.norm(obs['robot_xyz'][:2])
        accumulated_dis += np.linalg.norm(np.asarray(obs['robot_xyz'][:2]) - last_xy)
        print_dict(rew)
        # ---------- debug spl reward ---------- #
        # print(f'------- init distance_to_goal {env.init_distance_to_goal}')
        # print(f'-------distance_to_goal {obs["distance_to_goal"]}')
        # print(f'------distance_from_start {distance_from_start}')
        # print(f'------accumulated_dis {accumulated_dis}')
        # print(f'------eps rew closer_to_goal {episode_rew}')
        # print(f'------spl {obs["spl"]}')

        # ---------- debug region ---------- #
        # print_dict(env.states_dict["region_info"])
        # print_dict(rew)
        # # print(f'------ region info {env.states_dict["region_info"]}')
        # if env.states_dict['region_info']['region_changed']:
        #     print(f'{Fore.RED}================= REGION CHANGED !!!!! =============={Style.RESET_ALL}')

        # --------- debug check collision ------- #
        gps = env.habitat_observations['gps']
        current_xy = [-gps[2], gps[0]]
        delta_xy_norm = np.linalg.norm(np.asarray(current_xy) - last_xy)
        if env.habitat_action == "MOVE_FORWARD" and delta_xy_norm < 0.24:
            collide = True
        else:
            collide = False
        last_xy = copy.deepcopy(current_xy)
        print(f'============ collide: {collide}')
        print(f'============ delta_xy_norm: {delta_xy_norm}')
        if k == 27:  # ESC key
            break
    cv2.destroyAllWindows()


def try_grpc_on_cloud():
    test_scene = 'gmuS7Wgsbrx.json.gz'
    # test_scene = 'NEVASPhcrxR.json.gz'
    # test_scene = '5LpN3gDmAk7.json.gz'
    # test_scene = None
    test_scene = '1pXnuDYAj8r.json.gz'
    env = HabitatEnv(render=False,
                     habitat_data_dir='/cephfs/zhangtianchu',
                     encoder_model_dir='/home/cvpr/clip_model/',
                     image_encoder='clip_resnet',
                     scene_name=test_scene,
                     dataset_version='mp3d',
                     action_num=6,
                     shuffle=True,
                     rgb_width=320,
                     rgb_height=240,
                     # image_grpc_addr=('10.244.15.10', 50030),
                     )
    last_xy = [0, 0]
    while True:
        if env.done:
            env.reset()
        keyboard_action = 1
        obs, rew, done, info = env.step(keyboard_action, 0)


def debug_scene_change():
    # test_scene = 'gmuS7Wgsbrx.json.gz'
    test_scene = 'svBbv1Pavdk.json.gz'
    # test_scene = '29hnd4uzFmX.json.gz'
    env = HabitatEnv(render=True,
                     habitat_data_dir='/home/tianchu/Documents/code_qy/cvpr2022',
                     encoder_model_dir='/home/cvpr/clip_model/',
                     image_encoder='clip_resnet',
                     scene_name=test_scene,
                     num_episode_per_scene=50
                     )
    while True:
        obs, rew, done, info = env.step(1, 0)
        env.change_scene(scene_name=test_scene)
        env.reset()


if __name__ == "__main__":
    keyboard_control()
    # try_grpc_on_cloud()
    # debug_scene_change()
