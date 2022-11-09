import os
import yaml
from torch.utils.data import Dataset
import torch
from PIL import Image
from mocap.pose import load_bvh_file, interpolated_traj
from bvh import Bvh
import pickle
import math

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

class MoCapDataset(Dataset):
    def __init__(self, dataset_path, config_path, mocap_fr, image_tmpl, image_transform=None, test_mode=False):
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

    def _get_video_ind(self):
        self.data_dict = []
        len = 0
        for i in range(len(self.data_sync)):
            self.data_dict.append(range(len, len + self.data_sync[i][2] - self.data_sync[i][1]))
            len += self.data_sync[i][2] - self.data_sync[i][1]

    def _load_image(self, directory, idx):
        return [Image.open(os.path.join(directory, self.image_tmpl.format(idx))).convert('RGB')]

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
            
    def _load_rotation(self, directory, idx):
        # bvh rotation: Z axis, X axis, Y axis
        # Hip has 6 channels: [translation, rotation]
        # idx: current frame in a bvh file
        traj = pickle.load(open(directory, 'rb'))
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
        head_rotation = traj[idx][21:24]
        R_head = get_rotation_matrix(head_rotation[0], head_rotation[1], head_rotation[2])
        #rotation = R_hips @ R_spine @ R_spine1 @ R_spine2 @ R_spine3 @ R_neck @ R_head
        return {"Hips":R_hips, "Spine":R_spine, "Spine1":R_spine1, 
                "Spine2":R_spine2, "Spine3":R_spine3, "Neck":R_neck, "Head":R_head}

    # traj: .p文件名称
    # idx: 当前的视频帧 
    def _load_transform(self, traj, idx):
        bvh_file = os.path.join(traj[:-5], '.bvh')  ## '0213_take_01_traj.p'->'0213_take_01'+'.bvh'
        rotation = {}
        offset = {}
        offset = self._load_offset(bvh_file)
        rotation = self._load_rotation(traj, idx)
        Translation = offset["Hips"]@rotation["Hips"] + offset["Spine"]@rotation["Spine"] + offset["Spine1"]@rotation["Spine1"] + \
                      offset["Spine2"]@rotation["Spine2"] + offset["Spine3"]@rotation["Spine3"] + offset["Neck"]@rotation["Neck"] + offset["Head"]@rotation["Head"]
        Rotation = rotation["Hips"]@rotation["Spine"]@rotation["Spine1"]@rotation["Spine2"]@rotation["Spine3"]@rotation["Neck"]@rotation["Head"]
        return Translation, Rotation

    def _load_mocap(self, traj_file, idx):
        traj = pickle.load(open(traj_file, 'rb'))
        return torch.Tensor(traj[:][idx]).unsqueeze(0)

    def __getitem__(self, index):
        ind_bool = [index in i for i in self.data_dict]
        ind = ind_bool.index("True")  # ind表示该index属于第ind个视频
        ind_frame_in_mocap = ind - self.data_dict[ind][0] + self.data_sync[i][1] #确定index所对应的mocap中的帧数
        ind_frame_in_video = ind_frame_in_mocap - 4
        dir = self.data_dict[ind]
        image_dir = os.path.join(self.dataset_path, "fpv_frames", dir)
        traj_file = dir + "_traj.p"
        traj = os.path.join(self.dataset_path, "traj", traj_file)
        image = self._load_image(image_dir, ind_frame_in_video)
        label = self._load_mocap(traj, ind_frame_in_mocap)
        transformation = self._load_transform(traj_file, ind_frame_in_mocap)
        return self.transform(image), transformation, label
    
    def __len__(self):
        len = 0 
        for i in self.data_sync:
            temp = i[2] - i[1]
            len += temp
        return len