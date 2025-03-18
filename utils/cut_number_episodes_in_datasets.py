import json
import copy
import os
import random
import gzip

# Vars
n_episodes = 1000
setup = 'setup3'

# Get the list of available datasets
dataset_paths = os.listdir(f'../data/datasets/objectnav/v2/objectnav_hm3d_hd_{setup}/val/content/')

# Load the datasets
datasets = []
for d in dataset_paths:
    datasets.append(json.load(gzip.open(f'../data/datasets/objectnav/v2/objectnav_hm3d_hd_{setup}/val/content/{d}')))

# Select a 20% val episodes from the datasets
val_episodes = []
for d in datasets:
    val_episodes.append(random.sample(d['episodes'], 2000))

# create new datasets with the validation episodes
val_datasets = copy.deepcopy(datasets)
for i, val_dataset in enumerate(val_datasets):
    val_dataset['episodes'] = val_episodes[i]

# remove the validation episodes from the original datasets
for i, dataset in enumerate(datasets):
    dataset['episodes'] = [episode for episode in dataset['episodes'] if episode not in val_episodes[i]]

# make dirs
os.makedirs(f'data/datasets/objectnav/objectnav_hm3d_hd_{setup}/val/content/', exist_ok=True)
os.makedirs(f'data/datasets/objectnav/objectnav_hm3d_hd_{setup}/train/content/', exist_ok=True)

# Copy the object files
os.system(f'cp data/datasets/objectnav/objectnav_hm3d_hd/train/train.json.gz '
          f'data/datasets/objectnav/objectnav_hm3d_hd_{setup}/train')
os.system(f'cp data/datasets/objectnav/objectnav_hm3d_hd_setup1/val/val.json.gz '
          f'data/datasets/objectnav/objectnav_hm3d_hd_{setup}/val')

# save the new datasets
for i, val_dataset in enumerate(val_datasets):
    dataset_name = list(val_dataset['goals_by_category'].keys())[0].split('.')[0] + '.json'
    json.dump(val_dataset, open(f'data/datasets/objectnav/objectnav_hm3d_hd_{setup}/val/content/{dataset_name}', 'w'))
    json.dump(datasets[i], open(f'data/datasets/objectnav/objectnav_hm3d_hd_{setup}/train/content/{dataset_name}', 'w'))
    # Compress the files as gz
    os.system(f'gzip data/datasets/objectnav/objectnav_hm3d_hd_{setup}/val/content/{dataset_name}')
    os.system(f'gzip data/datasets/objectnav/objectnav_hm3d_hd_{setup}/train/content/{dataset_name}')
