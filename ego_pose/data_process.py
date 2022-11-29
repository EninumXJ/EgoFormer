import os
import yaml
from torch.utils.data import Dataset
import torch
from PIL import Image
from mocap.pose import load_bvh_file, interpolated_traj
from bvh import Bvh
import pickle
import math
import torch.nn.functional as F

subject_config = "meta_subject_01.yaml"

def get_rotation_matrix(z_deg, x_deg, y_deg):
    Rz = torch.tensor([[math.cos(z_deg/180.*math.pi), -math.sin(z_deg/180.*math.pi), 0],
                       [math.sin(z_deg/180.*math.pi), math.cos(z_deg/180.*math.pi), 0],
                       [0, 0, 1]])
    Rx = torch.tensor([[1, 0, 0],
                       [0, math.cos(x_deg/180.*math.pi), -math.sin(x_deg/180.*math.pi)],
                       [0, math.sin(x_deg/180.*math.pi), math.cos(x_deg/180.*math.pi)]])
    Ry = torch.tensor([[math.cos(y_deg/180.*math.pi), 0, math.sin(y_deg/180.*math.pi)],
                       [0, 1, 0],
                       [-math.sin(y_deg/180.*math.pi), 0, math.cos(y_deg/180.*math.pi)]])
    return Rz@Rx@Ry

def load_joint_offset(mocap, joint):
    joint_offset = mocap.joint_offset(joint)
    return torch.Tensor([[joint_offset[0], joint_offset[1], joint_offset[2]]])

# traj:将要读取的traj文件; index: 该关节在数组中的索引位置; idx:某一帧的帧数
def load_joint_rotation(traj, index, idx):
    print("index: ",index)
    print("idx: ",idx)
    if(index[1]-index[0] == 3):
        joint_rotation = traj[idx][index[0]:index[1]]
        return get_rotation_matrix(joint_rotation[0], joint_rotation[1], joint_rotation[2])
    else:
        joint_rotation = traj[idx][index[0]]
        return get_rotation_matrix(0, joint_rotation, 0)

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
    def _load_offset(self, bvh):
        # Hips->Spine->Spine1->Spine2->Spine3->Neck->Head
        with open(bvh) as f:
            mocap = Bvh(f.read())
        hips_offset = mocap.joint_offset('Hips')
        hips = torch.Tensor([[hips_offset[0], hips_offset[1], hips_offset[2]]])
        spine_offset = mocap.joint_offset('Spine')
        spine = torch.Tensor([[spine_offset[0], spine_offset[1], spine_offset[2]]])
        spine1_offset = mocap.joint_offset('Spine1')
        spine1 = torch.Tensor([[spine1_offset[0], spine1_offset[1], spine1_offset[2]]])
        spine2_offset = mocap.joint_offset('Spine2')
        spine2 = torch.Tensor([[spine2_offset[0], spine2_offset[1], spine2_offset[2]]])
        spine3_offset = mocap.joint_offset('Spine3')
        spine3 = torch.Tensor([[spine3_offset[0], spine3_offset[1], spine3_offset[2]]])
        neck_offset = mocap.joint_offset('Neck')
        neck = torch.Tensor([[neck_offset[0], neck_offset[1], neck_offset[2]]])
        head_offset = mocap.joint_offset('Head')
        head = torch.Tensor([[head_offset[0], head_offset[1], head_offset[2]]])
        return {"Hips":hips, "Spine":spine, "Spine1":spine1, 
                "Spine2":spine2, "Spine3":spine3, "Neck":neck, "Head":head}
            
    def _load_rotation(self, traj, idx):
        # bvh rotation: Z axis, X axis, Y axis
        # Hip has 6 channels: [translation, rotation]
        # idx: current frame in a bvh file
        # traj = pickle.load(open(directory, 'rb'))
        
        hips_rotation = traj[idx][3:6]
        R_hips = get_rotation_matrix(hips_rotation[0], hips_rotation[1], hips_rotation[2]) 
        spine_rotation = traj[idx][6:9]
        R_spine = get_rotation_matrix(spine_rotation[0], spine_rotation[1], spine_rotation[2]) 
        spine1_rotation = traj[idx][9:12]
        R_spine1 = get_rotation_matrix(spine1_rotation[0], spine1_rotation[1], spine1_rotation[2])
        spine2_rotation = traj[idx][12:15]
        R_spine2 = get_rotation_matrix(spine2_rotation[0], spine2_rotation[1], spine2_rotation[2])
        spine3_rotation = traj[idx][15:18]
        R_spine3 = get_rotation_matrix(spine3_rotation[0], spine3_rotation[1], spine3_rotation[2])
        neck_rotation = traj[idx][18:21]
        R_neck = get_rotation_matrix(neck_rotation[0], neck_rotation[1], neck_rotation[2])
        head_rotation = traj[idx][21:24]  # 31*3
        R_head = get_rotation_matrix(head_rotation[0], head_rotation[1], head_rotation[2])
        #rotation = R_hips @ R_spine @ R_spine1 @ R_spine2 @ R_spine3 @ R_neck @ R_head
        return {"Hips":R_hips, "Spine":R_spine, "Spine1":R_spine1, 
                "Spine2":R_spine2, "Spine3":R_spine3, "Neck":R_neck, "Head":R_head}

    # traj: .p文件名称
    # idx: 当前的视频帧 
    def _load_transform(self, traj, bvh_file, idx):
        #bvh_file = os.path.join(traj[:-5], '.bvh')  ## '0213_take_01_traj.p'->'0213_take_01'+'.bvh'
        rotation = {}
        offset = {}
        offset = self._load_offset(bvh_file)
        print("idx: ", idx)
        rotation = self._load_rotation(traj, idx)
        Translation = offset["Hips"]@rotation["Hips"] + offset["Spine"]@rotation["Spine"] + offset["Spine1"]@rotation["Spine1"] + \
                      offset["Spine2"]@rotation["Spine2"] + offset["Spine3"]@rotation["Spine3"] + offset["Neck"]@rotation["Neck"] + offset["Head"]@rotation["Head"]
        Rotation = rotation["Hips"]@rotation["Spine"]@rotation["Spine1"]@rotation["Spine2"]@rotation["Spine3"]@rotation["Neck"]@rotation["Head"]
        return Translation, Rotation
    
    def _load_f_u(self, traj, mocap, bvh_file, idx):
        # bvh_file = os.path.join(traj_file[:-5], '.bvh')  ## '0213_take_01_traj.p'->'0213_take_01'+'.bvh'
        rotation = {}
        offset = {}
        offset = self._load_offset(bvh_file)
        rotation = self._load_rotation(traj, idx)
        neck = load_joint_offset(mocap, 'Spine')@load_joint_rotation(traj, (6,9), idx) \
             + load_joint_offset(mocap, 'Spine1')@load_joint_rotation(traj, (9,12), idx) \
             + load_joint_offset(mocap, 'Spine2')@load_joint_rotation(traj, (12,15), idx) \
             + load_joint_offset(mocap, 'Spine3')@load_joint_rotation(traj, (15,18), idx) \
             + load_joint_offset(mocap, 'Neck')@load_joint_rotation(traj, (18,21), idx)
        f = neck + offset["Neck"]@rotation["Neck"]
        F.normalize(f, dim=1)
        # 我们假设u向量是f向量绕y轴顺时针旋转90度
        u = f@get_rotation_matrix(0, 0, -90)
        return f, u

    # 读取身体各个关节在本地坐标系中的坐标位置 一个包含51个元素的tensor
    def _load_keypoint_positon(self, traj, mocap, idx):
        hips = torch.Tensor([[0, 0, 0]])  # 1*3
        neck = load_joint_offset(mocap, 'Spine')@load_joint_rotation(traj, (6,9), idx) \
             + load_joint_offset(mocap, 'Spine1')@load_joint_rotation(traj, (9,12), idx) \
             + load_joint_offset(mocap, 'Spine2')@load_joint_rotation(traj, (12,15), idx) \
             + load_joint_offset(mocap, 'Spine3')@load_joint_rotation(traj, (15,18), idx) \
             + load_joint_offset(mocap, 'Neck')@load_joint_rotation(traj, (18,21), idx)
        head = neck + load_joint_offset(mocap, 'Head')@load_joint_rotation(traj, (21,24), idx)
        RightShoulder = neck + load_joint_offset(mocap, 'RightShoulder')@load_joint_rotation(traj, (24,27), idx)
        RightArm = RightShoulder + load_joint_offset(mocap, 'RightArm')@load_joint_rotation(traj, (27,30), idx)
        RightHand = RightArm + load_joint_offset(mocap, 'RightForeArm')@load_joint_rotation(traj, (30, 31), idx) \
                   + load_joint_offset(mocap, 'RightHand')@load_joint_rotation(traj, (31,34), idx)
        LeftShoulder = neck + load_joint_offset(mocap, 'LeftShoulder')@load_joint_rotation(traj, (34,37), idx)
        LeftArm = LeftShoulder + load_joint_offset(mocap, 'LeftArm')@load_joint_rotation(traj, (37,40), idx)
        LeftHand = LeftArm + load_joint_offset(mocap, 'LeftForeArm')@load_joint_rotation(traj, (40, 41), idx) \
                  + load_joint_offset(mocap, 'LeftHand')@load_joint_rotation(traj, (41,44), idx)
        RightUpLeg = hips + load_joint_offset(mocap, 'RightUpLeg')@load_joint_rotation(traj, (44,47), idx)
        RightLeg = RightUpLeg + load_joint_offset(mocap, 'RightLeg')@load_joint_rotation(traj, (47,48), idx)
        RightFoot = RightLeg + load_joint_offset(mocap, 'RightFoot')@load_joint_rotation(traj, (48,51), idx)
        LeftUpLeg = hips + load_joint_offset(mocap, 'LeftUpLeg')@load_joint_rotation(traj, (51,54), idx)
        LeftLeg = LeftUpLeg + load_joint_offset(mocap, 'LeftLeg')@load_joint_rotation(traj, (54,55), idx)
        LeftFoot = LeftLeg + load_joint_offset(mocap, 'LeftFoot')@load_joint_rotation(traj, (55,58), idx)
        return torch.cat([hips, neck, head, RightShoulder, RightArm, RightHand, LeftShoulder, LeftArm, LeftHand,
                          RightUpLeg, RightLeg, RightFoot, LeftUpLeg, LeftLeg, LeftFoot], dim=-1)

    def _build_motion(self, traj, bvh_file, index):
        R = []
        D = []
        for i in range(index-31,index):
            d, r = self._load_transform(traj, bvh_file, i)  # translation and rotation
            D.append(d)
            R.append(r)
        
        return torch.stack(R, 0), torch.stack(D, 0)

    def __getitem__(self, index):
        ind_bool = [index in i for i in self.data_dict]
        print("index:", index)
        ind = ind_bool.index(True)  # ind表示该index属于第ind个视频
        ind_frame_in_mocap = index - self.data_dict[ind][0] + self.data_sync[ind][1] #确定index所对应的mocap中的帧数
        ind_frame_in_video = ind_frame_in_mocap - 4
        if(ind_frame_in_video < 32):
            print("index out of bounds")
            return
        dir = self.data_list[ind]
        image_dir = os.path.join(self.dataset_path, "fpv_frames", dir)
        traj_file = dir + "_traj.p"
        traj_path = os.path.join(self.dataset_path, "traj", traj_file)
        bvh_file = dir + ".bvh"
        bvh_path = os.path.join(self.dataset_path, "traj", bvh_file)
        with open(bvh_path) as f:
            mocap = Bvh(f.read())
        traj = pickle.load(open(traj_path, 'rb'))
        image = self._load_image(image_dir, ind_frame_in_video)
        keypoints = self._load_keypoint_positon(traj, mocap, ind_frame_in_mocap)
        f, u = self._load_f_u(traj, mocap, bvh_path, ind_frame_in_mocap)
        label = torch.cat([f, u, keypoints], dim=1)
        R, d = self._build_motion(traj, bvh_path, ind_frame_in_mocap)
        print(image)
        return self.transform(image), label, R, d

    def __len__(self):
        len = 0 
        for i in self.data_sync:
            temp = i[2] - i[1]
            len += temp
        return len