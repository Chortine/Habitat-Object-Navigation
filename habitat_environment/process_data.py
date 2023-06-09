# process the json.gz file
import gzip
import json
import copy
import os

from habitat_environment.environment_2022 import HabitatEnv
import numpy as np


def inspect_all_goals(split_dir):
    output = {}
    for goal in ["chair", "bed", "plant", "toilet", "tv_monitor", "sofa"]:
        output[goal] = inspect_all_rooms(goal, split_dir)
    # with open('mp3d_train_goal_distribution.json', 'w', encoding='utf-8') as f:
    #     json.dump(output, f, ensure_ascii=False, indent=4)
    pass


def inspect_all_rooms(goal, split_dir):
    output = {}
    content_dir = os.path.join(split_dir, 'content')
    roomlist = os.listdir(content_dir)
    split = list(split_dir.split('/'))[-1]
    target_content_dir = content_dir.replace(f'/{split}', f'/{goal}_{split}')
    for room in roomlist:
        room_path = os.path.join(target_content_dir, room)
        episode_num = inspect_one_room(room_path)
        # if episode_num == 0:
        #     os.remove(room_path)
        output[room] = episode_num
    return output


def inspect_one_room(source_path):
    if not os.path.isfile(source_path):
        return 0
    with gzip.open(source_path, 'r') as file:
        json_bytes = file.read()
    json_str = json_bytes.decode('utf-8')  # 2. string (i.e. JSON)
    data = json.loads(json_str)
    episode_num = len(data['episodes'])
    return episode_num


def process_all_goals(split_dir):
    for goal in ["chair", "bed", "plant", "toilet", "tv_monitor", "sofa"]:
        process_all_rooms(goal, split_dir)


def process_all_rooms(goal, split_dir):
    content_dir = os.path.join(split_dir, 'content')
    roomlist = os.listdir(content_dir)
    split = list(split_dir.split('/'))[-1]
    target_content_dir = content_dir.replace(f'/{split}', f'/{goal}_{split}')
    for room in roomlist:
        source_path = os.path.join(content_dir, room)
        target_path = os.path.join(target_content_dir, room)
        process_one_room(goal, source_path, target_path)


def process_one_room(goal, source_path, target_path=None):
    with gzip.open(source_path, 'r') as file:
        json_bytes = file.read()
    json_str = json_bytes.decode('utf-8')  # 2. string (i.e. JSON)
    data = json.loads(json_str)

    new_episodes = []
    count_episode = 0
    for episode in data['episodes']:
        if episode['object_category'] == goal:
            new_episode = copy.deepcopy(episode)
            new_episode['episode_id'] = count_episode
            new_episodes.append(new_episode)
            count_episode += 1
    new_data = {}
    for key, value in data.items():
        if key != 'episodes':
            new_data[key] = copy.deepcopy(value)
    new_data['episodes'] = new_episodes

    json_str = json.dumps(new_data)
    json_bytes = json_str.encode('utf-8')
    if target_path is not None:
        with gzip.open(target_path, 'w') as file:  # 4. fewer bytes (i.e. gzip)
            file.write(json_bytes)


def generate_more_episodes_for_val(val_path):
    content_dir = os.path.join(val_path, 'content')
    roomlist = os.listdir(content_dir)
    for room in roomlist:
        if room in os.listdir('/home/tianchu/Documents/code_qy/cvpr2022/habitat-challenge-data/objectgoal_hm3d/val_enlarged/content'):
            continue
        source_path = os.path.join(content_dir, room)
        generate_episodes_for_one_room(source_path)


def generate_episodes_for_one_room(room_path):
    # 1. read val episode data
    target_path = room_path.replace(f'/val/', f'/val_enlarged/')
    with gzip.open(room_path, 'r') as file:
        json_bytes = file.read()
    json_str = json_bytes.decode('utf-8')  # 2. string (i.e. JSON)
    data = json.loads(json_str)
    example_episode = data['episodes'][0]
    goals_by_category = data['goals_by_category']
    goals = []
    for goal in list(goals_by_category.keys()):
        goal = list(goal.split('.glb_'))[-1]
        goals.append(goal)
    episodes_per_goal = int(50000/len(goals))
    # 2. open the scene in Habitat to get the navigable point
    # episodes_per_goal = 8333
    # goals = ["chair", "bed", "plant", "toilet", "tv_monitor", "sofa"]
    val_scene = list(room_path.split('/'))[-1]
    my_env = HabitatEnv(
        render=False,
        habitat_data_dir='/home/tianchu/Documents/code_qy/cvpr2022',
        encoder_model_dir='/home/tianchu/Documents/code_qy/cvpr2022/clip_model/',
        image_encoder='',
        image_grpc_addr=None,
        scene_name=val_scene,
        dataset_version='hm3d',
        split='val',
        action_num=6,
        shuffle=False,
        rgb_width=320,
        rgb_height=320,
        noisy_action=False,
        noisy_camera=False,
        specified_goal=None
    )
    episode_id = 0
    data['episodes'] = []
    for goal in goals:
        for i in range(episodes_per_goal):
            # travel to random place:
            random_pos = my_env.habitat_env.sim.pathfinder.get_random_navigable_point()
            rot = np.random.uniform(0, 2*np.pi)
            rot = [0, float(np.sin(rot)), 0, float(np.cos(rot))]

            new_episode = copy.deepcopy(example_episode)
            new_episode['episode_id'] = episode_id
            new_episode['start_position'] = [float(random_pos[0]), float(random_pos[1]), float(random_pos[2])]
            new_episode['start_rotation'] = rot  # xyzw # TODO: check
            new_episode['object_category'] = goal
            new_episode['info'] = {}
            data['episodes'].append(new_episode)
            episode_id += 1
    # 3. redump the data to a new file
    json_str = json.dumps(data)
    json_bytes = json_str.encode('utf-8')
    with gzip.open(target_path, 'w') as file:  # 4. fewer bytes (i.e. gzip)
        file.write(json_bytes)
    print(f'------------- processed scene{val_scene} ----------------')

if __name__ == '__main__':
    # ======== process room and generate split episodes for each goal ========== #
    # process_all_goals('/home/tianchu/Documents/code_qy/cvpr2022/habitat-challenge-data/objectnav_mp3d_v1/train')
    # process_all_goals('/home/tianchu/Documents/code_qy/cvpr2022/habitat-challenge-data/objectgoal_hm3d/train')
    # process_all_goals('')

    # ======= inspect the room and delete the room that has zero episode for the goal ============ #
    # inspect_all_goals('/home/tianchu/Documents/code_qy/cvpr2022/habitat-challenge-data/objectnav_mp3d_v1/train')
    # inspect_all_goals('/home/tianchu/Documents/code_qy/cvpr2022/habitat-challenge-data/objectgoal_hm3d/train')
    # inspect_all_goals('/home/tianchu/Documents/code_qy/cvpr2022/habitat-challenge-data/objectgoal_hm3d/val')

    # ======== generate new episode for val dataset ======== #
    # generate_episodes_for_one_room(
    #     '/home/tianchu/Documents/code_qy/cvpr2022/habitat-challenge-data/objectgoal_hm3d/val/content/XB4GS9ShBRE.json.gz')
    generate_more_episodes_for_val('/home/tianchu/Documents/code_qy/cvpr2022/habitat-challenge-data/objectgoal_hm3d/val')