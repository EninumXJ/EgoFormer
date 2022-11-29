import torch
from torch.utils.data import DataLoader
from ego_pose.data_process import MoCapDataset
from ego_pose.transforms import *
from ego_pose.model import *

def train():
    dataset_path = '/data1/lty/dataset/egopose_dataset/datasets'
    config_path = '/data1/lty/dataset/egopose_dataset/datasets/meta/meta_subject_01.yml'
    img_tmpl = "{:05d}.png"
    transform = torchvision.transforms.Compose([
                    Scale(256),
                    ToTorchFormatTensor(),
                    GroupNormalize(
                        mean=[.485, .456, .406],
                        std=[.229, .224, .225]
                    )
                ])
    batch_size = 2

    train_data = MoCapDataset(dataset_path=dataset_path, config_path=config_path, image_tmpl=img_tmpl, image_transform=transform, test_mode=False)
    train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=False)

    model = EgoNet()

    for step, (image, label, R, d) in enumerate(train_loader):
        print(image.shape)
        print(label.shape)
        
        keypoint, head1, head2 = model(image, R, d)
        print(keypoint.shape)
        print(head1.shape)
        print(head2.shape)
        if step >= 1:
            break

train()