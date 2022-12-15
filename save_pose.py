# This file is created to check if the traj file can be visualized properly
# and convert traj.p to .npy
from bvh import Bvh
from mocap.skeleton import Skeleton
from mocap.pose import load_bvh_file, interpolated_traj
import matplotlib
import numpy as np
import torch
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import math
import pickle
from utils.visualize import DrawSkeleton
import os
import yaml

skeleton = Skeleton()
exclude_bones = {'Thumb', 'Index', 'Middle', 'Ring', 'Pinky', 'End', 'Toe'}
dataset_path = '/data1/lty/dataset/egopose_dataset/datasets/'
bvh_path = os.path.join(dataset_path, 'traj')
config_ = '/data1/lty/dataset/egopose_dataset/datasets/meta/'
config_list = ['meta_subject_01.yml', 'meta_subject_02.yml', 'meta_subject_03.yml',
               'meta_subject_04.yml', 'meta_subject_05.yml']
npath = 'data/'
for yml in config_list:
    config_path = os.path.join(config_, yml)
    with open(config_path, 'r') as f:
        config = yaml.load(f.read(),Loader=yaml.FullLoader)
    bvh_list = config['train'] + config['test']
    for bvh_name in bvh_list:
        bvh_file = os.path.join(bvh_path, bvh_name+'.bvh')
        print(bvh_file)
        skeleton = Skeleton()
        skeleton.load_from_bvh(bvh_file, exclude_bones)
        poses, bone_addr = load_bvh_file(bvh_file, skeleton)
        poses = interpolated_traj(poses)
        ndarray_name = npath + bvh_name
        print("npy name:", ndarray_name)
        np.save(ndarray_name, poses)
        
        