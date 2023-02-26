import torch
from torch.utils.data import DataLoader
from ego_pose.data_process import MoCapDataset
from ego_pose.transforms import *
from ego_pose.transformer import *
from ego_pose.loss import *
import shutil
from opts import parser
import torch.optim
import torch.nn.parallel
from torch.nn.utils import clip_grad_norm
import os
import time
from tqdm import tqdm
from utils.visualize import DrawSkeleton45

exp_name = 'train03'
path = os.getcwd()
save_path = os.path.join(path, 'results', exp_name)
if not os.path.exists(save_path):
    os.makedirs(save_path) 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = EgoViT(N=6, d_model=120, d_ff=720, pose_dim=51, h=5, dropout=0.1)

dataset_path = '/home/liumin/litianyi/workspace/data/datasets'
config_path = '/home/liumin/litianyi/workspace/data/datasets/meta/meta_subject_01.yml'
### load checkpoints if exist

resume = 'logs/train03/transformer_model_best.pth.tar'
# checkpoint = torch.load(resume)
# model.load_state_dict(checkpoint['state_dict'])
# model = nn.DataParallel(model, device_ids=[0,1]).cuda()
# model.load_state_dict({k.replace('module.',''):v for k,v in torch.load(resume)['state_dict'].items()})   
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
                        shuffle=True, num_workers=1, pin_memory=True)

model.eval()
for i, (motion, label) in tqdm(enumerate(val_loader), total=10):
    label = label.to(device)
    tgt = label
    src = motion.to(device)
    # src shape:(batch,length,feature_dim)
    src_mask = (src.sum(axis=-1) != 0).squeeze(-1).unsqueeze(-2)
    # src_mask shape:(batch,1,length)
    tgt_mask = (tgt.sum(axis=-1) != 0).squeeze(-1).unsqueeze(-2)
    # tgt_mask shape:(batch,1,length)
    mask_ = torch.tensor(subsequent_mask(tgt.size(-2)).type_as(tgt_mask.data))
    # mask_ shape:(1,length,length)
    tgt_mask = tgt_mask & mask_
    # tgt_mask shape:(batch,length,length)
    output = model(src, tgt, src_mask, tgt_mask)
    # output shape:(batch,length,pose_dim)label = label.to(device)
    tgt = label
    src = motion.to(device)
    # src shape:(batch,length,feature_dim)
    src_mask = (src.sum(axis=-1) != 0).squeeze(-1).unsqueeze(-2)
    # src_mask shape:(batch,1,length)
    tgt_mask = (tgt.sum(axis=-1) != 0).squeeze(-1).unsqueeze(-2)
    # tgt_mask shape:(batch,1,length)
    mask_ = torch.tensor(subsequent_mask(tgt.size(-2)).type_as(tgt_mask.data))
    # mask_ shape:(1,length,length)
    tgt_mask = tgt_mask & mask_
    # tgt_mask shape:(batch,length,length)
    output = model(src, tgt, src_mask, tgt_mask)
    # output shape:(batch,length,pose_dim)
    # print(output)
    output = output.cpu().detach().numpy()
    label = label.cpu().detach().numpy()
    print("label shape: ", label.shape)
    for j in range(output.shape[1]):
        image_name = 'Skeleton'+'_'+str(i)+'_'+str(j)+'.jpg'
        image_path = os.path.join(save_path, image_name)
        label_name = 'Label'+'_'+str(i)+'_'+str(j)+'.jpg'
        label_path = os.path.join(save_path, label_name)
        keypoint = output[:, j, 6:]
        head1 = output[:, j, 0:3]
        head2 = output[:, j, 3:6]
        print("head1 shape: ", head1.shape)
        label_ = label[:, j, :]
        print("label shape: ", label.shape)
        print(label_.squeeze(0)[6:].shape)
        DrawSkeleton45(label_.squeeze(0)[6:], label_.squeeze(0)[0:3], label_.squeeze(0)[3:6], label_path)
        DrawSkeleton45(keypoint[0], head1[0], head2[0], image_path)