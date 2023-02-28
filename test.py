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

def inference(model, src, src_mask):
    memory = model.model.encode(src, src_mask)
    ys = torch.zeros(1,1).type_as(src)
    L = 20  # 序列长度
    for i in range(L-1):
         out = model.model.decode(
            memory, src_mask, ys, subsequent_mask(ys.size(1)).type_as(src.data)
        )
    return ys
         
exp_name = 'train09'
length = 20
path = os.getcwd()
save_path = os.path.join(path, 'results', exp_name)
if not os.path.exists(save_path):
    os.makedirs(save_path) 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
Model = EgoViT(N=16, d_model=120, d_ff=1440, pose_dim=45, h=10, dropout=0.1)
dataset_path = '/home/liumin/litianyi/workspace/data/datasets'
config_path = '/home/liumin/litianyi/workspace/data/datasets/meta/meta_subject_01.yml'
### load checkpoints if exist

resume = 'logs/train09/transformer_model_best.pth.tar'
# checkpoint = torch.load(resume)
# model.load_state_dict(checkpoint['state_dict'])
# model = nn.DataParallel(model, device_ids=[0,1]).cuda()
# model.load_state_dict({k.replace('module.',''):v for k,v in torch.load(resume)['state_dict'].items()})   
Model = Model.to(device)
val_data = MoCapDataset(dataset_path=dataset_path, 
                              config_path=config_path, 
                              image_tmpl="{:05d}.png", 
                              image_transform=torchvision.transforms.Compose([
                                        Scale(256),
                                        ToTorchFormatTensor(),
                                        GroupNormalize(
                                            mean=[.485, .456, .406],
                                            std=[.229, .224, .225])
                                        ]), 
                              L=length,
                              test_mode=True)

val_loader = DataLoader(dataset=val_data, batch_size=1, 
                        shuffle=True, num_workers=1, pin_memory=True)
keypoints_intro = torch.tensor(np.load("/home/liumin/litianyi/workspace/data/datasets/keypoints/0213_take_01.npy")).to(device)
print(keypoints_intro.shape)
d_max = torch.max(keypoints_intro, dim=1)[0].unsqueeze(1)
# d_max shape: (batch)->(batch, 1)
d_min = torch.min(keypoints_intro, dim=1)[0].unsqueeze(1)
# print("d_min shape: ", d_min.shape)
dst = d_max - d_min
keypoints_intro = ((keypoints_intro - d_min) / dst - 0.5) / 0.5
Model.eval()
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
    # output = model(src, tgt, src_mask, tgt_mask)
    # output shape:(batch,length,pose_dim)label = label.to(device)
    memory = Model.model.encode(src, src_mask)
    # memory shape: (batch, length, feature_dim)
    # ys = torch.zeros(1, 1, 45).fill_(0).type_as(src.data)
    ys = keypoints_intro[1, 6:].reshape(1, 1, 45)
    for k in range(length - 1):
        out = Model.model.decode(
            memory, src_mask, ys, subsequent_mask(ys.size(1)).type_as(src.data)
        )
        pose = Model.model.generator(out[:, -1]).unsqueeze(0)
        # pose shape: (1, pose_dim)->(1, 1, pose_dim)
        ys = torch.cat([ys, pose], dim=1)
    # ys shape: (1, length, pose_dim)
    ys = ys.cpu().detach().numpy()
    label = label.cpu().detach().numpy()
    print("label shape: ", label.shape)
    for j in range(ys.shape[1]):
        image_name = 'Skeleton'+'_'+str(i)+'_'+str(j)+'.jpg'
        image_path = os.path.join(save_path, image_name)
        label_name = 'Label'+'_'+str(i)+'_'+str(j)+'.jpg'
        label_path = os.path.join(save_path, label_name)
        keypoint = ys[:, j, :]
        # head1 = ys[:, j, 0:3]
        # head2 = ys[:, j, 3:6]
        label_ = label[:, j, :]
        DrawSkeleton45(label_.squeeze(0)[:], head1=None, head2=None, image_name=label_path)
        DrawSkeleton45(keypoint[0], head1=None, head2=None, image_name=image_path)