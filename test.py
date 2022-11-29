import numpy as np
import math
from bvh import Bvh
import pickle
import yaml
from mocap.pose import load_bvh_file
from mocap.skeleton import Skeleton
from utils.transformation import quaternion_slerp, quaternion_from_euler, euler_from_quaternion
import os.path
import torch

fname = '/data1/lty/dataset/egopose_dataset/datasets/traj/1205_take_15.bvh'
with open(fname) as f:
    mocap = Bvh(f.read())
print("Number of joints: ",len(mocap.get_joints_names()))
print("Number of frames: ", mocap.nframes)
print(mocap.get_joints_names())
### channels of a joint is position & rotation in X Y Z
print(mocap.joint_channels('Hips'))
print(mocap.joint_offset('Hips'))
#print(mocap.joint_parent('Spine').name)

skeleton = Skeleton()
exclude_bones = {'Thumb', 'Index', 'Middle', 'Ring', 'Pinky', 'End', 'Toe'}
spec_channels = {'LeftForeArm': ['Zrotation'], 'RightForeArm': ['Zrotation'],
                 'LeftLeg': ['Xrotation'], 'RightLeg': ['Xrotation']}
# skeleton.load_from_bvh(fname, exclude_bones, spec_channels)
# poses, bone_addr = load_bvh_file(fname, skeleton)
# print("poses: ", poses)
# print("bone_addr: ", bone_addr)
pname = '/data1/lty/dataset/egopose_dataset/datasets/traj/1205_take_15_traj.p'
traj = pickle.load(open(pname, 'rb'))
traj_frame = torch.Tensor(traj[:][0]).unsqueeze(0)
print("traj ", traj.shape)
joint = traj[69][6:9]
print("joint: ",joint)

# config_path = '/data1/lty/dataset/egopose_dataset/datasets/meta/meta_subject_01.yml'
# with open(config_path, 'r') as f:
#     config = yaml.load(f.read(),Loader=yaml.FullLoader)
# train_split = config['train']
# print(train_split)
# test_split = config['test']
# train_sync = [config['video_mocap_sync'][i] for i in train_split]
# print(train_sync)
str1 = "abcdfr"
print(str1[:-4])
a = [[1,2,3,4],
     [5,6,7,8]]
print(a[0][0:2])
dataset_path = "/data1/lty/dataset/ego_datasets"
img_dir = "r0310_take_24"
path = os.path.join(dataset_path, "fpv_frames", img_dir)
print(path)

a = [[[[1.,2.,3.], [3.,2.,1.], [4.,5.,6.]],[[1.,2.,3.], [3,2,1], [4,5,6]]], [[[1,2,3], [3,2,1], [4,5,6]],[[1,2,3], [3,2,1], [4,5,6]]]]
b = [[[[1,2,3], [3,2,1], [4,5,6]],[[1,2,3], [3,2,1], [4,5,6]]], [[[1,2,3], [3,2,1], [4,5,6]],[[1,2,3], [3,2,1], [4,5,6]]]]
c = [[1,2,3], [4,5,6], [7,8,9]]
a = torch.Tensor(a)
c = torch.tensor(c)
print(a.shape)
print(c.size)
print(c.shape.append(1))
# d = torch.cat([a.view(-1,3,3),c.view(1,3,3)], dim=0)
# print(d.shape)
# b = torch.zeros_like(a)
# a = torch.where(a > 3., b, a)
# print(a)
