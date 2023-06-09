import os
# os.environ['MKL_NUM_THREADS'] = '2'
import time
import random
import math
import json
import numpy as np

import cv2
from typing import Type, List
import habitat
from habitat.sims.habitat_simulator.actions import HabitatSimActions
# import flow.api as flow
from habitat_environment.settings import task_name_to_yaml_config
from pathlib import Path
from collections import deque

from habitat_environment.grpc_clip.clip_image.client import clip_grpc as grpc_image
from habitat_environment.grpc_clip.clip_text.client import clip_grpc as grpc_text

current_dir = os.path.dirname(__file__)

CATEGORY_ID_NOT_IN_TASK = 10086

# Will use outside. Do not want to store all the observation in buffer.
observation_space = {
    "CommonFeatureSet": {
        "clip_image": 1024,
        "clip_goal": 1024,
        "last_action": 4,
        "relative_position": 2,
        "yaw": 2
    },
    "SpatialFeatureSet": {
    },
    "Custom": {
        "relative_map": [201, 201],
        "rgb": [120, 160, 3],
        "depth": [120, 160, 1]
    }
}
six_cat_to_id = {"chair": 0, "bed": 1, "plant": 2, "toilet": 3, "tv_monitor": 4, "sofa": 5}


def load_valid_scenes():
    mission_folder_avail = []
    fh = open(os.path.join(current_dir, 'configs/valid_scenes_objectnav_2021.txt'))
    for line in fh.readlines():
        mission_folder_avail.append(line[:-1])
    fh.close()

    mission_folder_small = []
    fh = open(os.path.join(current_dir, 'configs/data_set_objnav_small.txt'))
    for line in fh.readlines():
        mission_folder_small.append(line[:-1])
    fh.close()

    mission_folder = []
    for mission in mission_folder_avail:
        if mission in mission_folder_small:
            mission_folder.append(mission)
    return mission_folder


class HabitatEnv:

    def __init__(self,
                 task,
                 scene_directory=None,
                 render=False,
                 large_visual_obs=False,
                 use_clip=True,
                 image_grpc_ip=None,
                 image_grpc_port=None,
                 text_grpc_ip=None,
                 text_grpc_port=None,
                 env_id=-1,
                 gpu_id=0,
                 num_episode_per_scene=50,
                 max_step_per_episode=500,
                 local=True,
                 scene_name='ur6pFq6Qu1A.json.gz',
                 clip_vit=False,
                 use_six_cat=False):

        self.local_run = local
        self.num_episode_per_scene = num_episode_per_scene
        self.max_game_len = max_step_per_episode
        self.should_render = render
        self.use_large_visual_obs = large_visual_obs
        self.use_clip = use_clip
        self.scene_name = scene_name
        if use_six_cat:
            self.content_dir = 'content_2022'
        else:
            self.content_dir = 'content'
        self.use_six_cat = use_six_cat
        if self.use_six_cat:
            text_grpc_ip = None

        if self.local_run:
            self.env_start_time = time.time()
            self.episode_start_time = time.time()
            self.step_time_deque = deque(maxlen=max_step_per_episode)

        self.grpc_client_image = None
        self.grpc_client_text = None
        if image_grpc_ip is not None:
            self.grpc_client_image = grpc_image(ip=image_grpc_ip, port=image_grpc_port)
        if text_grpc_ip is not None:
            self.grpc_client_text = grpc_text(ip=text_grpc_ip, port=text_grpc_port)

        import os
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

        # os.environ['MKL_NUM_THREADS'] = '2'

        """ set dataset path and scene path """
        config = habitat.get_config(str(Path(__file__).parent.resolve()) + task_name_to_yaml_config[task])
        config.defrost()
        if scene_directory is not None:
            config.DATASET.SCENES_DIR = scene_directory
        # get names of all scene data
        self.dataset_path = str(Path(__file__).parent.resolve()) + config.DATASET.DATA_PATH
        # self.dataset: List[str] = os.listdir(self.dataset_path + 'content/')
        self.dataset = load_valid_scenes()
        # print(f"===========++++++++++++++++list dataset{len(self.dataset)}, {self.dataset}")
        """ initialize habitat environment on a random scene """
        if self.scene_name is None:
            model_name = self.dataset[random.randrange(0, len(self.dataset))]
        else:
            model_name = self.scene_name
        config.DATASET.DATA_PATH = self.dataset_path + f'{self.content_dir}/' + str(model_name)
        config.ENVIRONMENT.ITERATOR_OPTIONS.NUM_EPISODE_SAMPLE = self.num_episode_per_scene  # TODO: better strategy
        config.SIMULATOR.HABITAT_SIM_V0.GPU_DEVICE_ID = 0
        # change visual input resolution if specified

        if self.should_render and self.use_large_visual_obs:
            config.SIMULATOR.RGB_SENSOR.WIDTH = 1024
            config.SIMULATOR.RGB_SENSOR.HEIGHT = 768
            config.SIMULATOR.DEPTH_SENSOR.WIDTH = 1024
            config.SIMULATOR.DEPTH_SENSOR.HEIGHT = 768
        config.SEED = np.random.randint(1, 100000)
        config.ENVIRONMENT.MAX_EPISODE_STEPS = self.max_game_len
        config.freeze()
        self.config = config
        self.habitat_env = habitat.Env(config=self.config)

        """ Read mapping from task object to 3D scene environment object id """
        self.category_to_task_category_id = None
        self.category_to_mp3d_category_id = None

        json_path = self.dataset_path + f'{self.config.DATASET.SPLIT}.json'
        with open(json_path, 'r') as json_file:
            load_dict = json.load(json_file)
            for key, value in load_dict.items():
                if key == 'category_to_task_category_id':
                    self.category_to_task_category_id = value
                if key == 'category_to_mp3d_category_id':
                    self.category_to_mp3d_category_id = value

            self.task_id_to_cat_name = {v: k for k, v in self.category_to_task_category_id.items()}
            self.mp3d_id_to_cat_name = {v: k for k, v in self.category_to_mp3d_category_id.items()}

        if self.category_to_task_category_id is not None and self.category_to_mp3d_category_id is not None:
            # we are doing an object nav task
            # TODO: better architecture that explicitly distinguishes different task types
            self.mp3d_category_id_to_task_category_id = {}
            assert len(self.category_to_mp3d_category_id) == len(self.category_to_task_category_id), \
                f"Wrong category id mapping in {self.config.DATASET.SPLIT}.json"
            for key, _ in self.category_to_task_category_id.items():
                self.mp3d_category_id_to_task_category_id[self.category_to_mp3d_category_id[key]] = \
                    self.category_to_task_category_id[key]

        """ initialize CLIP """
        if self.use_clip and self.grpc_client_image is None:
            self.torch = __import__('torch')
            # self.torch.torch.set_num_threads(32)
            self.clip = __import__('clip')
            self.PIL = __import__('PIL')
            self.CLIP_device = "cuda"
            if clip_vit:
                clip_model = 'ViT-B-32.pt'
            else:
                clip_model = 'RN50.pt'
            if self.local_run:
                self.model, self.preprocess = self.clip.load(f"/home/cvpr/clip_model/{clip_model}",
                                                             device=self.CLIP_device)
            else:
                self.model, self.preprocess = self.clip.load(f"/cephfs/zhangtianchu/clip_model/{clip_model}",
                                                             device=self.CLIP_device)

        """ parameters of the environment """
        self.n_classes = 4
        self.semantic_scene = self.habitat_env.sim.semantic_annotations()
        self.update_object_idmap()

        self.episode_count = 0
        self.step_count = 0
        self.success_right_count = 0
        self.success_false_count = 0
        self.object_goal_type = None
        self.object_goal_clip = None
        self.reachable_goal_id = None

        # Used in reward:
        # Global map is a fine grid map. It use True/False recording the area it has visited
        # The map's shape is decided by the ground truth, which means the shape is different between diff data sets.
        # If the agent goes forward, it will go to the adjacent grid or stay in the the same grid.
        # The grid unit size is similar to the step-forward length
        # The agent resets at, for example, (123, 234) grid.
        # Global map aims to encourage the agent exploring
        self.my_metrics = {"new_area": False}  # self define metrics
        self.global_map = None
        self.floor_id = 0

        # Used in obs  # todo: need optimize
        # Relative map is a coarse grid map. It use True/False recording the area it has visited
        # The map's shape is decided by the maximum step, because we do not know the house size.
        # The agent resets at the center of the map.
        # Relative map aims to tell the agent which direction it comes from.
        self.relative_map_ratio = 250. / 200.  # shrink ratio, by meter, One step 0.25m in default. 0.25m*500*2=250m， 200*200‘s final grid size
        self.relative_map = np.zeros([201, 201], dtype=bool)
        self.relative_map_center = np.array([100, 100], dtype=np.int32)

        self.local_map = np.zeros([max_step_per_episode], dtype=bool)

        self.summary_info = {}
        self.obs = {}
        self.habitat_observations = {}

        """ Reward """
        self.total_distance2goal = None  # The distance between initial position and the chosen goal
        self.last_distance2goal = None

    def reset(self):
        """
        Change scene if number of episode in current scene is larger than the threshold.
        Otherwise, just start a new episode in the same scene.
        """
        self.step_count = 0
        self.episode_count += 1
        self.episode_start_time = time.time()

        if self.episode_count > self.num_episode_per_scene:
            self.change_scene(random.randrange(0, len(self.dataset)))
            self.episode_count = 1

        """ iterate until a valid episode has been found """
        while True:
            self.habitat_observations = self.habitat_env.reset()
            self.metrics = self.habitat_env.get_metrics()
            if math.isinf(self.metrics['distance_to_goal']) or math.isnan(self.metrics['distance_to_goal']):
                continue

            self.object_goal_type = self.habitat_observations["objectgoal"][0]
            goal_name = self.task_id_to_cat_name[self.object_goal_type]
            if self.use_clip and self.grpc_client_text is None:
                if self.use_six_cat:
                    self.object_goal_clip = np.asarray([0, 0, 0, 0, 0, 0], dtype=np.float32)
                    self.object_goal_clip[int(six_cat_to_id[goal_name])] = 1
                else:
                    self.object_goal_clip = self._observations_clip_text(goal_name)
            if self.use_clip and self.grpc_client_text is not None:
                self.object_goal_clip = np.asarray(self.grpc_client_text.send_and_recieve(goal_name))

            self.reachable_goal_id = []
            for goal in self.habitat_env.current_episode.goals:
                for view_point in goal.view_points:
                    view_point_position = view_point.agent_state.position
                    current_position = self.habitat_env.sim.get_agent_state().position
                    distance_to_target = self.habitat_env.sim.geodesic_distance(
                        current_position, view_point_position
                    )

                    if distance_to_target < float("inf"):
                        self.reachable_goal_id.append(goal.object_id)
                        break
            break

        self.obs["last_action"] = np.zeros(self.n_classes, dtype=np.float32)
        self.init_global_pos = self.habitat_observations["gps"]
        self.global_map = None
        self.floor_id = 0
        self.relative_map = np.zeros([201, 201], dtype=bool)  # center at [100, 100]

        self.observe()

        self.render()

        self.obs["last_action"] = np.zeros(self.n_classes, dtype=np.float32)

        """ Reward """
        self.total_distance2goal = self.metrics['distance_to_goal']
        self.last_distance2goal = self.metrics['distance_to_goal']
        return self.obs

    def update_object_idmap(self):
        self.obj_id_to_task_cat_id = [CATEGORY_ID_NOT_IN_TASK] * len(self.semantic_scene.objects)
        self.obj_id_to_mp3d_cat_id = [0] * len(self.semantic_scene.objects)

        for obj in self.semantic_scene.objects:
            obj_category_id_mp3d = obj.category.index()  # category id in Matterport scope
            obj_instance_id = int(obj.id.split("_")[-1])  # obj.id is in the form of <level_id>_<region_id>_<object_id>
            if obj_category_id_mp3d in self.category_to_mp3d_category_id.values():
                obj_category_id_task = self.mp3d_category_id_to_task_category_id[obj_category_id_mp3d]
                self.obj_id_to_task_cat_id[obj_instance_id] = obj_category_id_task

            self.obj_id_to_mp3d_cat_id[obj_instance_id] = obj_category_id_mp3d

    def step(self, rl_action):
        if self.local_run:
            start_time = time.time()
        self.step_count += 1
        self.metrics = self.habitat_env.get_metrics()

        last_action_one_hot = np.zeros(self.n_classes, dtype=np.float32)
        last_action_one_hot[rl_action] = 1.
        self.obs["last_action"] = last_action_one_hot
        self.habitat_action = self._interpret_action(rl_action)

        if not self.habitat_env.episode_over:
            self.habitat_observations = self.habitat_env.step(self.habitat_action)
        else:
            raise Exception("Episode has finished. Should not call step() before calling reset(). ")

        # checking for done condition
        eps_done = False
        if self.habitat_action == 'STOP' or self.step_count > self.max_game_len or self.habitat_env.episode_over:
            eps_done = True

        self.observe()
        reward = self.reward_jing()

        self.render()
        if eps_done and self.local_run:
            print(
                f'================done!==============, episode {self.episode_count}; max_len is {self.max_game_len}, episode over = {self.habitat_env.episode_over}, success right rate = {self.success_right_count / self.episode_count}, success false rate = {self.success_false_count / self.episode_count}')
        if self.local_run:
            self.step_time_deque.append(time.time() - start_time)
            # print(f'=========average step time = {sum(self.step_time_deque)/len(self.step_time_deque)}')
        # print(f'+++++++torch thread = {self.torch.torch.get_num_threads()}')
        return self.obs, reward, eps_done, self.summary_info

    def observe(self):
        """ return all necessary information """
        self.metrics = self.habitat_env.get_metrics()

        # is_goal_in_view, bag_of_cat_id, bag_of_cat_name, goal_objects_in_view = self._observations_semantic()
        goal_name = self.task_id_to_cat_name[self.object_goal_type]

        self.summary_info = {
            'distance_to_goal': self.metrics['distance_to_goal'],
            'success': self.metrics['success'],
            'spl': self.metrics['spl'],
            'softspl': self.metrics['softspl'],
            'top_down_map': self.metrics['top_down_map'],
            'collisions': self.metrics['collisions'],
            # 'is_goal_in_view': is_goal_in_view,
            # 'goal_objects_in_view': goal_objects_in_view,
            'goal_name': goal_name,
            # 'bag_of_cat_id': bag_of_cat_id,
            # 'bag_of_cat_name': bag_of_cat_name
        }

        # self.obs['rgb_raw'] = self.habitat_observations['rgb']
        # self.obs['semantic'] = self.habitat_observations['semantic']
        # self.obs['depth'] = self.habitat_observations['depth']
        self.obs['goal_obj_cat_id'] = self.habitat_observations['objectgoal']
        self.obs['compass'] = self.habitat_observations['compass']
        self.obs['gps'] = self.habitat_observations['gps']

        rgb = self._observations_rgb()
        self.obs['rgb'] = rgb

        if self.use_clip and self.grpc_client_image is None:
            start_clip_time = time.time()
            self.obs['clip_image'] = self._observations_clip_image()
            self.obs['clip_goal'] = self.object_goal_clip
            self.summary_info['clip_time'] = time.time() - start_clip_time
        # rgb = np.ones([2, 2, 3], dtype=np.uint8)
        if self.use_clip and self.grpc_client_image is not None:
            start_clip_time = time.time()
            rgb = np.ascontiguousarray(self.habitat_observations["rgb"])
            self.obs['clip_image'] = np.asarray(self.grpc_client_image.send_and_recieve(rgb))
            self.obs['clip_goal'] = self.object_goal_clip
            self.summary_info['clip_time'] = time.time() - start_clip_time

        relative_position = np.array(self.habitat_observations["gps"] - self.init_global_pos, dtype=np.float32)
        self.obs["relative_position"] = relative_position / 125.  # todo: normalized by 500 steps * 0.25m/step = 125
        self.obs["yaw"] = np.array([np.sin(self.habitat_observations["compass"][0]),
                                    np.cos(self.habitat_observations["compass"][0])], dtype=np.float32)

        self.obs['summary_info'] = self.summary_info
        self._observations_relative_map(relative_position)

    def reward_jing(self):
        reward = 0
        if self.habitat_action == 'STOP':
            if self.metrics["success"]:
                if self.local_run:
                    print(f'================done success!!! {self.step_count}==============')
                self.success_right_count += 1
                reward += 5.
            else:
                if self.local_run:
                    print('================done success false!!!==============')
                self.success_false_count += 1
                reward += -2.
        if self.step_count > self.max_game_len:
            if self.local_run:
                print('================done max!!!==============')
            reward += -0.3
        if self.metrics['collisions']:
            if self.local_run:
                # print('================done collision!!!==============')
                pass
            reward += -0.0001

        # Closer to one of the goal
        closer_to_goal = (self.last_distance2goal - self.metrics['distance_to_goal']) / self.total_distance2goal
        reward += 10. * closer_to_goal
        # print('closer_to_goal: ', closer_to_goal)

        # go to new area
        reward += self._reward_global_map() * 0.002

        self.last_distance2goal = self.metrics['distance_to_goal']

        return reward

    def reward_foo(self):
        """ User defined reward """
        self._reward_global_map()
        return 1 if self.metrics['success'] else 0

    def _reward_global_map(self):
        """ GLOBAL MAP"""
        # When habitat_env init, the returned metrics['top_down_map'] is None.
        # ==> We need to initial grid map in the first step
        if self.global_map is None and self.metrics['top_down_map'] is not None:
            origin_shape = self.metrics['top_down_map']['map'].shape
            # It goes around 18 grids during one step. Not sure +1 or not, just add it in case index are out of range...
            shrink_shape = (int(origin_shape[0] / 20. + 1), int(origin_shape[1] / 20. + 1))
            self.global_map = np.zeros(shrink_shape, dtype=bool)
            # print('initial self.global_map')

        if self.global_map is not None:
            present_coord_origin = self.metrics['top_down_map']['agent_map_coord']
            present_coord_shrink = [int(present_coord_origin[0] / 20.), int(present_coord_origin[1] / 20.)]
            floor = self.metrics
            if not self.global_map[present_coord_shrink[0]][present_coord_shrink[1]]:
                self.global_map[present_coord_shrink[0]][present_coord_shrink[1]] = True
                self.my_metrics["new_area"] = True
                # print(present_coord_shrink, '  new point')
            else:
                self.my_metrics["new_area"] = False
                # print(present_coord_shrink, '  old point')

        return 1.0 if self.my_metrics["new_area"] else 0.0

    def _observations_relative_map(self, relative_position):
        """ RELATIVE MAP"""
        relative_map_grid = np.array(relative_position / self.relative_map_ratio,
                                     dtype=np.int32) + self.relative_map_center
        self.relative_map[relative_map_grid[0]][relative_map_grid[1]] = True
        self.obs["relative_map"] = self.relative_map
        # print(relative_map_grid, 'relative_map_grid')

    def render(self):
        if self.should_render:
            # rgb
            cv_rgb = self.obs['rgb'][:, :, [2, 1, 0]]
            cv2.imshow("RGB", cv_rgb)
            cv2.waitKey(1)
            # depth
            # depth = self.obs['depth']
            # cv2.imshow("depth", depth)

            # map
            if self.metrics['top_down_map'] is not None:
                fog_of_war_mask = np.asarray(self.metrics['top_down_map']['fog_of_war_mask'])
                # cv2.imshow("fog_of_war_mask", fog_of_war_mask * 128)
                top_down_map = np.asarray(self.metrics['top_down_map']['map'])
                show_top_down_map = cv2.applyColorMap(top_down_map * 50, cv2.COLORMAP_JET)
                show_top_down_map = cv2.resize(show_top_down_map, (640, 640), interpolation=cv2.INTER_AREA)
                cv2.imshow("top_down_map", show_top_down_map)
            #
            # # semantic
            # if self.habitat_observations['semantic'] is not None:
            #     cv2.imshow("semantic segmentation", self.habitat_observations['semantic'] / 1000)

    def change_scene(self, scene_id):
        self.habitat_env.close()
        """ change the scene and reconfigure """
        if self.scene_name is None:
            data_name = str(self.dataset[scene_id])
        else:
            data_name = self.scene_name
        self.config.defrost()
        self.config.DATASET.DATA_PATH = self.dataset_path + f'{self.content_dir}/' + data_name
        self.config.SEED = np.random.randint(1, 100000)
        self.config.freeze()
        self.habitat_env = habitat.Env(config=self.config)

        """ update semantic info for all object instances in the current scene """
        self.semantic_scene = self.habitat_env.sim.semantic_annotations()
        self.update_object_idmap()

    def _interpret_action(self, action):
        assert 0 <= action < len(self.config.TASK.POSSIBLE_ACTIONS), "invalid action"
        return self.config.TASK.POSSIBLE_ACTIONS[int(action)]

    def _observations_rgb(self):
        rgb = self.habitat_observations["rgb"]
        return (rgb / 255.).astype(np.float32)

    def _observations_clip_image(self):
        rgb = self.habitat_observations["rgb"]
        PIL_image = self.PIL.Image.fromarray(rgb.astype('uint8'), 'RGB')
        rgb_clip = self.preprocess(PIL_image).unsqueeze(0).to(self.CLIP_device)
        with self.torch.no_grad():
            image_features = self.model.encode_image(rgb_clip)
        clip_output = image_features.clone().detach().cpu().numpy()[0]
        clip_output = clip_output.astype(np.float32)
        return clip_output

    def _observations_clip_text(self, text):
        text = self.clip.tokenize([text]).to(self.CLIP_device)
        with self.torch.no_grad():
            text_features = self.model.encode_text(text)
        clip_text_output = text_features.clone().detach().cpu().numpy()[0]
        clip_text_output = clip_text_output.astype(np.float32)
        return clip_text_output

    def _observations_semantic(self):
        object_id_image = self.habitat_observations['semantic']  # image of object id, not category id

        all_object_id_in_view = np.unique(object_id_image)
        mp3d_category_id_in_view = set()
        category_name_in_view = set()

        goal_in_view = False
        goal_objects_in_view = set()  # a set of tuple in the form of (obj_id, sum_pixel)

        for obj_id in all_object_id_in_view:
            mp3d_category_id_in_view.add(self.obj_id_to_mp3d_cat_id[obj_id])
            if not self.obj_id_to_task_cat_id[obj_id] == CATEGORY_ID_NOT_IN_TASK:
                category_name_in_view.add(self.task_id_to_cat_name[self.obj_id_to_task_cat_id[obj_id]])
            if self.obj_id_to_task_cat_id[obj_id] == self.object_goal_type:
                sum_pixel = np.sum(object_id_image == obj_id)
                goal_in_view = True if sum_pixel >= 40 else False
                goal_objects_in_view.add((obj_id, sum_pixel))

        return goal_in_view, mp3d_category_id_in_view, category_name_in_view, goal_objects_in_view


if __name__ == '__main__':
    # Ning Yuan
    # env = HabitatEnv(task='objectnav2021',
    #                  scene_directory='/home/ningyuan/habitat-2021/',
    #                  render=False,
    #                  large_visual_obs=True,
    #                  use_clip=False,
    #                  num_episode_per_scene=5,
    #                  max_step_per_episode=2000)

    # Wang Jing
    between_step_time = deque(maxlen=500)
    between_step_time.append(0)
    step_time = deque(maxlen=500)
    LOCAL_RUN = False
    if LOCAL_RUN:
        scence_directory = '/home/habitat-2021'
    else:
        scence_directory = '/cephfs/wangjing'
    gpu_id = 0
    env = HabitatEnv(task="objectnav2022",
                     scene_directory=scence_directory,
                     render=False,
                     large_visual_obs=False,
                     use_clip=True,
                     env_id=0,
                     gpu_id=gpu_id,
                     image_grpc_ip='10.244.19.20',
                     image_grpc_port=50001,
                     text_grpc_ip='10.244.19.20',
                     text_grpc_port=50002,
                     num_episode_per_scene=1000,
                     max_step_per_episode=1000,
                     local=LOCAL_RUN,
                     scene_name='ur6pFq6Qu1A.json.gz',
                     # scene_name=None,
                     clip_vit=False)

    env.reset()

    FORWARD_KEY = "w"
    LEFT_KEY = "a"
    RIGHT_KEY = "d"
    FINISH = "f"

    last_start_time = None
    done = False
    while True:
        if not done:
            cv2.waitKey(1)
            keystroke = "w"
            if keystroke == ord(FORWARD_KEY):
                keyboard_action = 1
            elif keystroke == ord(LEFT_KEY):
                keyboard_action = 2
            elif keystroke == ord(RIGHT_KEY):
                keyboard_action = 3
            elif keystroke == ord(FINISH):
                keyboard_action = 0
                print('PRESSED STOP BUTTON \n')
            else:
                keyboard_action = HabitatSimActions.MOVE_FORWARD
            start_time = time.time()
            if last_start_time is not None:
                between_step_time.append(start_time - last_start_time)
            last_start_time = start_time
            observer_output, r, done, info = env.step(keyboard_action)
            step_time.append(time.time() - start_time)
            print(f'=========average between step time = {sum(between_step_time) / len(between_step_time)}')
            print(f'=========average step time = {sum(step_time) / len(step_time)}')
            # time.sleep(0.0482)
            # print(info['bag_of_cat_name'], info['goal_name'], info['is_goal_in_view'])
        else:
            env.reset()
            done = False
