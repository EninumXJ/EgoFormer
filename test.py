import torch
from torch.utils.data import DataLoader
from ego_pose.data_process import MoCapDataset
from ego_pose.transforms import *
from ego_pose.model import *
from ego_pose.loss import *
import shutil
from opts import parser
import torch.optim
import torch.nn.parallel
from torch.nn.utils import clip_grad_norm
import os
import time
from tqdm import tqdm
from train import build_foreground, build_motion_history
from utils.visualize import DrawSkeleton

exp_name = 'train01'
path = os.getcwd()
save_path = os.path.join(path, 'results', exp_name)
if not os.path.exists(save_path):
    os.makedirs(save_path) 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = EgoNet()

dataset_path = '/data1/lty/dataset/egopose_dataset/datasets'
config_path = '/data1/lty/dataset/egopose_dataset/datasets/meta/meta_subject_01.yml'
### load checkpoints if exist

resume = 'logs/train01/baseline_stage1_checkpoint.pth.tar'
# checkpoint = torch.load(resume)
# model.load_state_dict(checkpoint['state_dict'])
# model = nn.DataParallel(model, device_ids=[0,1]).cuda()
model.load_state_dict({k.replace('module.',''):v for k,v in torch.load(resume)['state_dict'].items()})   
model = model.to(device)
val_data = MoCapDataset(dataset_path=dataset_path, 
                              config_path=config_path, 
                              image_tmpl="{:05d}.png", 
                              image_transform=torchvision.transforms.Compose([
                                        Scale(256),
                                        ToTorchFormatTensor(),
                                        GroupNormalize(
                                            mean=[.485, .456, .406],
                                            std=[.229, .224, .225])
                                        ]), test_mode=True)

val_loader = DataLoader(dataset=val_data, batch_size=1, 
                        shuffle=False, num_workers=1, pin_memory=True)

for i, (image, label, R, d) in tqdm(enumerate(val_loader), total=len(val_loader)):
    with torch.no_grad():
        label = label.to(device)
        foreground = build_foreground(image)
        foreground = foreground.to(device)
        motion_input = build_motion_history(R, d)
        motion_input = motion_input.to(device)
        print(foreground.shape)
        print(motion_input.shape)
        keypoint, head1, head2 = model(foreground, motion_input)
        
        image_name = 'Skeleton'+'_'+str(i)+'.jpg'
        image_path = os.path.join(save_path, image_name)
        label_name = 'Label'+'_'+str(i)+'.jpg'
        label_path = os.path.join(save_path, label_name)
        keypoint = keypoint.cpu()
        head1 = head1.cpu()
        head2 = head2.cpu()
        label = label.cpu()
        print(label)
        
        DrawSkeleton(label.squeeze()[6:], label.squeeze()[0:3], label.squeeze()[3:6], label_path)
        DrawSkeleton(keypoint[0], head1[0], head2[0], image_path)