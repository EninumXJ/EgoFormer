# This file is created to check if the traj file can be visualized properly
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

def get_rotation_matrix(x_rad, y_rad, z_rad):
    # Rz = torch.tensor([[math.cos(z_deg/180.*math.pi), -math.sin(z_deg/180.*math.pi), 0],
    #                    [math.sin(z_deg/180.*math.pi), math.cos(z_deg/180.*math.pi), 0],
    #                    [0, 0, 1]])
    # Rx = torch.tensor([[1, 0, 0],
    #                    [0, math.cos(x_deg/180.*math.pi), -math.sin(x_deg/180.*math.pi)],
    #                    [0, math.sin(x_deg/180.*math.pi), math.cos(x_deg/180.*math.pi)]])
    # Ry = torch.tensor([[math.cos(y_deg/180.*math.pi), 0, math.sin(y_deg/180.*math.pi)],
    #                    [0, 1, 0],
    #                    [-math.sin(y_deg/180.*math.pi), 0, math.cos(y_deg/180.*math.pi)]])
    # Rz = torch.tensor([[math.cos(z_deg), -math.sin(z_deg), 0],
    #                     [math.sin(z_deg), math.cos(z_deg), 0],
    #                     [0, 0, 1]])
    Rx = torch.tensor([[1, 0, 0],
                        [0, math.cos(x_rad), -math.sin(x_rad)],
                        [0, math.sin(x_rad), math.cos(x_rad)]])
    Ry = torch.tensor([[math.cos(y_rad), 0, math.sin(y_rad)],
                        [0, 1, 0],
                        [-math.sin(y_rad), 0, math.cos(y_rad)]])
    Rz = torch.tensor([[math.cos(z_rad), -math.sin(z_rad), 0],
                        [math.sin(z_rad), math.cos(z_rad), 0],
                        [0, 0, 1]])
    return Rx@Ry@Rz


# traj:将要读取的traj文件; index: 该关节在数组中的索引位置; idx:某一帧的帧数
def load_joint_rotation(traj, index, idx):
    if(index[1]-index[0] == 3):
        joint_rotation = traj[idx][index[0]:index[1]]
        return get_rotation_matrix(joint_rotation[0], joint_rotation[1], joint_rotation[2])
    # elif index[0] == 47 or index[0] == 54:   ## index = 47 and 54 is LeftLeg and RightLeg (X_rotation)
    #     joint_rotation = traj[idx][index[0]]
    #     return get_rotation_matrix(joint_rotation, 0, 0)   # X_rotation
    # elif index[0] == 30 or index[0] == 40:   ## index = 30 and 40 is RightForeArm and LeftForeArm (Z_rotation)
    #     joint_rotation = traj[idx][index[0]]
    #     return get_rotation_matrix(0, 0, joint_rotation)   # Z_rotation

def _load_offset(mocap):
    # Hips->Spine->Spine1->Spine2->Spine3->Neck->Head
    joint_name = ['Hips', 'Spine', 'Spine1', 'Spine2', 'Spine3', 'Neck', 'Head',
                    'RightShoulder', 'RightArm', 'RightForeArm', 'RightHand',
                    'LeftShoulder', 'LeftArm', 'LeftForeArm', 'LeftHand',
                    'RightUpLeg', 'RightLeg', 'RightFoot',
                    'LeftUpLeg', 'LeftLeg', 'LeftFoot']
    joint_offset = {}
    for joint in joint_name:
        offset_ = mocap.joint_offset(joint)
        offset = torch.Tensor([[offset_[0], offset_[1], offset_[2]]]).T
        joint_offset[joint] = offset
    return joint_offset

def _load_rotation(traj, idx):
    # bvh rotation: Z axis, X axis, Y axis
    # Hip has 6 channels: [translation, rotation]
    # idx: current frame in a bvh file
    # traj = pickle.load(open(directory, 'rb'))
    x_hips = torch.Tensor([[traj[idx][0], traj[idx][1], traj[idx][2]]]).T
    r_hips = load_joint_rotation(traj, (3,6), idx) 
    r_spine = load_joint_rotation(traj, (6,9), idx)     
    r_spine1 = load_joint_rotation(traj, (9,12), idx) 
    r_spine2 = load_joint_rotation(traj, (12,15), idx)
    r_spine3 = load_joint_rotation(traj, (15,18), idx) 
    r_neck = load_joint_rotation(traj, (18,21), idx) 
    r_head = load_joint_rotation(traj, (21,24), idx) 
    r_RightShoulder = load_joint_rotation(traj, (24,27), idx)  
    r_RightArm = load_joint_rotation(traj, (27,30), idx) 
    r_RightForeArm = load_joint_rotation(traj, (30,33), idx) 
    r_RightHand = load_joint_rotation(traj, (33,36), idx) 
    r_LeftShoulder = load_joint_rotation(traj, (36, 39), idx)   
    r_LeftArm = load_joint_rotation(traj, (39, 42), idx) 
    r_LeftForeArm = load_joint_rotation(traj, (42, 45), idx) 
    r_LeftHand = load_joint_rotation(traj, (45, 48), idx)
    r_RightUpLeg = load_joint_rotation(traj, (48, 51), idx) 
    r_RightLeg = load_joint_rotation(traj, (51, 54), idx) 
    r_RightFoot = load_joint_rotation(traj, (54, 57), idx) 
    r_LeftUpLeg = load_joint_rotation(traj, (57, 60), idx) 
    r_LeftLeg = load_joint_rotation(traj, (60, 63), idx) 
    r_LeftFoot = load_joint_rotation(traj, (63, 66), idx) 
    
    R_hips = r_hips
    R_spine = R_hips@r_spine
    R_spine1 = R_spine@r_spine1
    R_spine2 = R_spine1@r_spine2
    R_spine3 = R_spine2@r_spine3
    R_neck = R_spine3@r_neck
    R_head = R_neck@r_head
    R_rightShoulder = R_spine3@r_RightShoulder
    R_rightArm = R_rightShoulder@r_RightArm
    R_rightForeArm = R_rightArm@r_RightForeArm
    R_rightHand = R_rightForeArm@r_RightHand
    R_leftShoulder = R_spine3@r_LeftShoulder
    R_leftArm = R_leftShoulder@r_LeftArm
    R_leftForeArm = R_leftArm@r_LeftForeArm
    R_leftHand = R_leftForeArm@r_LeftHand
    R_rightUpLeg = R_hips@r_RightUpLeg
    R_rightLeg = R_rightUpLeg@r_RightLeg
    R_rightFoot = R_rightLeg@r_RightFoot
    R_leftUpLeg = R_hips@r_LeftUpLeg
    R_leftLeg = R_leftUpLeg@r_LeftLeg
    R_leftFoot = R_leftLeg@r_LeftFoot
    return {"translation":x_hips, "Hips":R_hips, "Spine":R_spine, "Spine1":R_spine1, "Spine2":R_spine2, "Spine3":R_spine3, "Neck":R_neck, "Head":R_head,
            "RightShoulder":R_rightShoulder, "RightArm":R_rightArm, "RightForeArm":R_rightForeArm, "RightHand":R_rightHand,
            "LeftShoulder":R_leftShoulder, "LeftArm":R_leftArm, "LeftForeArm":R_leftForeArm, "LeftHand":R_leftHand,
            "RightUpLeg":R_rightUpLeg, "RightLeg":R_rightLeg, "RightFoot":R_rightFoot,
            "LeftUpLeg":R_leftUpLeg, "LeftLeg":R_leftLeg, "LeftFoot":R_leftFoot}

def _load_keypoint_positon(rotation, offset):
    # hips = rotation['translation']@rotation['Hips']
    hips = rotation['translation']
    spine3 = hips + rotation['Spine']@offset['Spine'] \
                  + rotation['Spine1']@offset['Spine1'] \
                  + rotation['Spine2']@offset['Spine2'] \
                  + rotation['Spine3']@offset['Spine3'] 
    neck = spine3 + rotation['Neck']@offset['Neck']
    head = neck + rotation['Head']@offset['Head']
    RightShoulder = spine3 + rotation['RightShoulder']@offset['RightShoulder']
    RightArm = RightShoulder + rotation['RightArm']@offset['RightArm']
    RightForeArm = RightArm + rotation['RightForeArm']@offset['RightForeArm']
    RightHand = RightForeArm + rotation['RightHand']@offset['RightHand']
    LeftShoulder = spine3 + rotation['LeftShoulder']@offset['LeftShoulder']
    LeftArm = LeftShoulder + rotation['LeftArm']@offset['LeftArm']
    LeftForeArm = LeftArm + rotation['LeftForeArm']@offset['LeftForeArm']
    LeftHand = LeftForeArm + rotation['LeftHand']@offset['LeftHand']
    RightUpLeg = hips + rotation['RightUpLeg']@offset['RightUpLeg']
    RightLeg = RightUpLeg + rotation['RightLeg']@offset['RightLeg']
    RightFoot = RightLeg + rotation['RightFoot']@offset['RightFoot']
    LeftUpLeg = hips + rotation['LeftUpLeg']@offset['LeftUpLeg']
    LeftLeg = LeftUpLeg + rotation['LeftLeg']@offset['LeftLeg']
    LeftFoot = LeftLeg + rotation['LeftFoot']@offset['LeftFoot']
    return torch.cat([hips.T, neck.T, head.T, RightShoulder.T, RightArm.T, RightForeArm.T, RightHand.T, LeftShoulder.T, LeftArm.T,
                      LeftForeArm.T, LeftHand.T, RightUpLeg.T, RightLeg.T, RightFoot.T, LeftUpLeg.T, LeftLeg.T, LeftFoot.T], dim=-1)


bvh_file = '/data1/lty/dataset/egopose_dataset/datasets/traj/1205_take_15.bvh'
# bvh_file = '/data1/lty/dataset/CMU_MoCap/01/01_01.bvh'
with open(bvh_file) as f:
    mocap = Bvh(f.read())
# print(mocap.get_joints_names())
# print([str(item) for item in mocap.root])
# print("bvh frames is:", mocap.nframes)
# print("bvh frame time is:", mocap.frame_time)
# print(mocap.joint_channels('Hips'))
# print(mocap.joint_offset('Hips'))
# print(mocap.joint_channels('RightShoulder'))
# print(mocap.joint_channels('RightArm'))
# print(mocap.joint_channels('RightForeArm'))
# print(mocap.joint_channels('RightHand'))
# print(mocap.joint_parent('RightShoulder').name)
# skeleton = Skeleton()
# exclude_bones = {'Thumb', 'Index', 'Middle', 'Ring', 'Pinky', 'End', 'Toe'}
# # # spec_channels = {'LeftForeArm': ['Zrotation'], 'RightForeArm': ['Zrotation'],
# # #                 'LeftLeg': ['Xrotation'], 'RightLeg': ['Xrotation']}
# skeleton.load_from_bvh(bvh_file, exclude_bones)
# poses, bone_addr = load_bvh_file(bvh_file, skeleton)
# print(bone_addr)
# poses = interpolated_traj(poses)
# np.save('1205_take_15', poses)
# print(poses.shape)

poses = np.load('1205_take_15.npy')
poses = poses[:-1]
print(poses.shape)

for i in range(poses.shape[0]):
    ind_frame_in_mocap = i
    rotation = _load_rotation(poses, ind_frame_in_mocap)
    offset = _load_offset(mocap)
    keypoints = _load_keypoint_positon(rotation, offset)
    path = 'results/train04/Skeleton_' + "%04d" % i + '.jpg'
    DrawSkeleton(keypoints[0], image_name=path)