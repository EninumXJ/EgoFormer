import os
import yaml
from torch.utils.data import Dataset
from PIL import Image
from mocap.pose import load_bvh_file, interpolated_traj

subject_config = "meta_subject_01.yaml"

class MoCapDataset(Dataset):
    def __init__(self, dataset_path, config_path, mocap_fr):
        with open(config_path, 'r') as f:
            config = yaml.load(f.read(),Loader=yaml.FullLoader)
        self.dataset_path = dataset_path
        self.capture = config['capture']  # frame rate of video
        self.train_split = config['train']
        self.test_split = config['test']
        self.train_sync = [config['video_mocap_sync'][i] for i in self.train_split]
        self.test_sync = [config['video_mocap_sync'][i] for i in self.test_split]
        self.mocap_fr = mocap_fr
        
    def _load_image(self, directory, idx):
        return [Image.open(os.path.join(directory, self.image_tmpl.format(idx))).convert('RGB')]

    def _load_bvh(self, directory):
        
    def 
    def __getitem__(self, index):
        for i in 