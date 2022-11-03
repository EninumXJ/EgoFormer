import numpy as np
import math
from bvh import Bvh
import pickle
import yaml
from mocap.pose import load_bvh_file
from mocap.skeleton import Skeleton
from utils.transformation import quaternion_slerp, quaternion_from_euler, euler_from_quaternion

fname = '/data1/lty/dataset/egopose_dataset/datasets/traj/1205_take_15.bvh'
with open(fname) as f:
    mocap = Bvh(f.read())
print("Number of joints: ",len(mocap.get_joints_names()))
print("Number of frames: ", mocap.nframes)
print(mocap.get_joints_names())
### channels of a joint is position & rotation in X Y Z
print(mocap.joint_channels('Hips'))
print(mocap.offset)

skeleton = Skeleton()
exclude_bones = {'Thumb', 'Index', 'Middle', 'Ring', 'Pinky', 'End', 'Toe'}
spec_channels = {'LeftForeArm': ['Zrotation'], 'RightForeArm': ['Zrotation'],
                 'LeftLeg': ['Xrotation'], 'RightLeg': ['Xrotation']}
# skeleton.load_from_bvh(fname, exclude_bones, spec_channels)
# poses, bone_addr = load_bvh_file(fname, skeleton)
#print("poses: ", poses)
#print("bone_addr: ", bone_addr)
pname = '/data1/lty/dataset/egopose_dataset/datasets/traj/1205_take_15_traj.p'
traj = pickle.load(open(pname, 'rb'))
print("traj.shape: ", traj.shape)
# config_path = '/data1/lty/dataset/egopose_dataset/datasets/meta/meta_subject_01.yml'
# with open(config_path, 'r') as f:
#     config = yaml.load(f.read(),Loader=yaml.FullLoader)
# train_split = config['train']
# print(train_split)
# test_split = config['test']
# train_sync = [config['video_mocap_sync'][i] for i in train_split]
# print(train_sync)