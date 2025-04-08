import json
import copy
import os
import random
import gzip

# Vars
setup = 'setup5'
action_names = ['STOP', 'MOVE_FORWARD', 'TURN_LEFT', 'TURN_RIGHT', 'LOOK_UP', 'LOOK_DOWN']

# load reference replay
reference_replay = json.load(open(f'../reference_replay.json'))

# Translate the actions in the reference replay to the new action space
translated_reference_replay = {}
for episode_key, action_list in reference_replay.items():
    scene_id = episode_key.split('?')[0]
    episode_id = episode_key.split('?')[1]
    if scene_id not in translated_reference_replay:
        translated_reference_replay[scene_id] = {}
    new_action_list = []
    for action in action_list:
        new_action_list.append({'action': action_names[action['action']]})
    translated_reference_replay[scene_id][episode_id] = new_action_list

# example = json.load(open(f'../data/datasets/objectnav/objectnav_hm3d_hd_full/train/content/1S7LAXRdDqK.json'))
#
# for episode in example['episodes']:
#     if len(episode['reference_replay']) >= 500:
#         print('hey')

# Get the list of available datasets
dataset_paths = os.listdir(f'../data/datasets/objectnav/objectnav_hm3d_hd_{setup}/val/content/')

# Load the datasets
datasets = []
for d in dataset_paths:
    datasets.append(json.load(gzip.open(f'../data/datasets/objectnav/objectnav_hm3d_hd_{setup}/val/content/{d}')))

for dataset in datasets:
    for episode in dataset['episodes']:
        scene_id = episode['scene_id']
        scene_id = 'data/scene_datasets/' + scene_id
        episode_id = episode['episode_id']
        episode['reference_replay'] = translated_reference_replay[scene_id][episode_id]

# Copy the object files
# os.system(f'cp data/datasets/objectnav/objectnav_hm3d_hd/train/train.json.gz '
#           f'data/datasets/objectnav/objectnav_hm3d_hd_{setup}/train')
# os.system(f'cp data/datasets/objectnav/objectnav_hm3d_hd_setup1/val/val.json.gz '
#           f'data/datasets/objectnav/objectnav_hm3d_hd_{setup}/val')

# save the new datasets
for dataset in datasets:
    dataset_name = list(dataset['goals_by_category'].keys())[0].split('.')[0] + '.json'
    json.dump(dataset, open(f'../data/datasets/objectnav/objectnav_hm3d_hd_{setup}/val/content/{dataset_name}', 'w'))
    # Compress the files as gz
    os.system(f'gzip -f ../data/datasets/objectnav/objectnav_hm3d_hd_{setup}/val/content/{dataset_name}')
