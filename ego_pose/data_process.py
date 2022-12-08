import os
import yaml
from torch.utils.data import Dataset
import torch
from PIL import Image
from mocap.pose import load_bvh_file, interpolated_traj
from bvh import Bvh
import pickle
import math
import numpy as np
import torch.nn.functional as F

subject_config = "meta_subject_01.yaml"

def get_rotation_matrix(x_rad, y_rad, z_rad):
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

def load_joint_offset(mocap, joint):
    joint_offset = mocap.joint_offset(joint)
    return torch.Tensor([[joint_offset[0], joint_offset[1], joint_offset[2]]]).T

# traj:将要读取的traj文件; index: 该关节在数组中的索引位置; idx:某一帧的帧数
def load_joint_rotation(traj, index, idx):
    if(index[1]-index[0] == 3):
        joint_rotation = traj[idx][index[0]:index[1]]
        return get_rotation_matrix(joint_rotation[0], joint_rotation[1], joint_rotation[2])

class MoCapDataset(Dataset):
    def __init__(self, dataset_path, config_path, image_tmpl, image_transform=None, mocap_fr=30, test_mode=False):
        with open(config_path, 'r') as f:
            config = yaml.load(f.read(),Loader=yaml.FullLoader)
        self.dataset_path = dataset_path
        self.capture = config['capture']  # frame rate of video
        self.test_mode = test_mode
        if self.test_mode == False:
            self.data_list = config['train']
            self.data_sync = [config['video_mocap_sync'][i] for i in self.data_list]
        else:
            self.data_list = config['test']
            self.data_sync = [config['video_mocap_sync'][i] for i in self.data_list]
        self.mocap_fr = mocap_fr
        self.image_tmpl = image_tmpl
        self.transform = image_transform

        self.data_dict = []
        length = 0
        for i in range(len(self.data_sync)):
            self.data_dict.append(range(length, length + self.data_sync[i][2] - self.data_sync[i][1]))
            length += self.data_sync[i][2] - self.data_sync[i][1]

    def _get_video_ind(self):
        self.data_dict = []
        len = 0
        for i in range(len(self.data_sync)):
            self.data_dict.append(range(len, len + self.data_sync[i][2] - self.data_sync[i][1]))
            len += self.data_sync[i][2] - self.data_sync[i][1]

    def _load_image(self, directory, idx):
        return Image.open(os.path.join(directory, self.image_tmpl.format(idx))).convert('RGB')

    # load mocap offset from bvh file
    def _load_offset(self, mocap):
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
            
    def _load_rotation(self, traj, idx):
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
        return {"translation":x_hips, "Hips":R_hips, "Spine":R_spine, "Spine1":R_spine1, "Spine2":R_spine2, "Spine3":R_spine3,
                "Neck":R_neck, "Head":R_head, "RightShoulder":R_rightShoulder, "RightArm":R_rightArm, "RightForeArm":R_rightForeArm, 
                "RightHand":R_rightHand, "LeftShoulder":R_leftShoulder, "LeftArm":R_leftArm, "LeftForeArm":R_leftForeArm, 
                "LeftHand":R_leftHand, "RightUpLeg":R_rightUpLeg, "RightLeg":R_rightLeg, "RightFoot":R_rightFoot,
                "LeftUpLeg":R_leftUpLeg, "LeftLeg":R_leftLeg, "LeftFoot":R_leftFoot}

    # traj: .p文件名称
    # idx: 当前的视频帧 
    def _load_transform(self, offset, traj, idx):
        #bvh_file = os.path.join(traj[:-5], '.bvh')  ## '0213_take_01_traj.p'->'0213_take_01'+'.bvh'
        rotation = self._load_rotation(traj, idx)
        # print("frame: ", idx)
        Translation = rotation["translation"] + rotation["Spine"]@offset["Spine"] + rotation["Spine1"]@offset["Spine1"] + \
                      rotation["Spine2"]@offset["Spine2"] + rotation["Spine3"]@offset["Spine3"] + rotation["Neck"]@offset["Neck"] + rotation["Head"]@offset["Head"]
        Rotation = rotation["Head"]
        return Translation.T, Rotation
    
    def _load_f_u(self, rotation, offset):
        # bvh_file = os.path.join(traj_file[:-5], '.bvh')  ## '0213_take_01_traj.p'->'0213_take_01'+'.bvh'
        f = rotation["Head"]@offset["Head"]
        F.normalize(f, dim=1)
        # 我们假设u向量是f向量绕y轴顺时针旋转90度
        u = get_rotation_matrix(0, -90/180*math.pi, 0)@f
        return f.T, u.T

    # 读取身体各个关节在本地坐标系中的坐标位置 一个包含51个元素的tensor
    def _load_keypoint_positon(self, rotation, offset):
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
        return torch.cat([hips.T, neck.T, head.T, RightShoulder.T, RightArm.T, RightHand.T, LeftShoulder.T, LeftArm.T,
                          LeftHand.T, RightUpLeg.T, RightLeg.T, RightFoot.T, LeftUpLeg.T, LeftLeg.T, LeftFoot.T], dim=-1)

    def _build_motion(self, offset, traj, index):
        R = []
        D = []
        for i in range(index-31,index):
            d, r = self._load_transform(offset, traj, i)  # translation and rotation
            D.append(d)   ### d = 1*3 tensor
            R.append(r)
        
        return torch.stack(R, 0), torch.stack(D, 0)

    def __getitem__(self, index):
        ind_bool = [index in i for i in self.data_dict]
        print("index:", index)
        ind = ind_bool.index(True)  # ind表示该index属于第ind个视频
        ind_frame_in_mocap = index - self.data_dict[ind][0] + self.data_sync[ind][1] #确定index所对应的mocap中的帧数
        #ind_frame_in_video = ind_frame_in_mocap - 4
        ind_frame_in_video = ind_frame_in_mocap + self.data_sync[ind][0]
        print("ind_frame_in_video: ", ind_frame_in_video)
        print("ind_frame_in_mocap: ", ind_frame_in_mocap)
        if(ind_frame_in_video < 32):
            print("index out of bounds")
            return
        dir = self.data_list[ind]
        image_dir = os.path.join(self.dataset_path, "fpv_frames", dir)
        traj_file = dir + ".npy"
        traj_path = "data/" + traj_file
        bvh_file = dir + ".bvh"
        bvh_path = os.path.join(self.dataset_path, "traj", bvh_file)
        with open(bvh_path) as f:
            mocap = Bvh(f.read())
        traj = np.load(traj_path)[:-1]
        image = self._load_image(image_dir, ind_frame_in_video)

        rotation = self._load_rotation(traj, ind_frame_in_mocap)
        offset = self._load_offset(mocap)
        keypoints = self._load_keypoint_positon(rotation, offset)
        f, u = self._load_f_u(rotation, offset)
        label = torch.cat([f, u, keypoints], dim=1)
        R, d = self._build_motion(offset, traj, ind_frame_in_mocap)
        return self.transform(image), label, R, d

    def __len__(self):
        len = 0 
        for i in self.data_sync:
            temp = i[2] - i[1]
            len += temp
        return len