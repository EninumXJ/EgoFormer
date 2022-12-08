from torch import nn
from ego_pose.transforms import *
from torch.nn.init import normal, constant
import torch.nn.functional as F

class ResBlockB(nn.Module):
    def __init__(self, channels_in, channels_out):
        super(ResBlockB, self).__init__()
        self.channels_in = channels_in
        self.channels_out = channels_out
        self.conv1 = nn.Sequential(
                        nn.Conv2d(self.channels_in, self.channels_out, 3, 1, 1),
                        nn.BatchNorm2d(self.channels_out),
                        nn.ReLU()
                    )
        self.conv2 = nn.Sequential(
                        nn.Conv2d(self.channels_out, self.channels_out, 3, 1, 1),
                        nn.BatchNorm2d(self.channels_out)
                    )
        self.conv3 = nn.Sequential(
                        nn.Conv2d(self.channels_out, self.channels_out, 3, 1, 1),
                        nn.BatchNorm2d(self.channels_out),
                        nn.ReLU()
                    )
        self.conv4 = nn.Sequential(
                        nn.Conv2d(self.channels_out, self.channels_out, 3, 1, 1),
                        nn.BatchNorm2d(self.channels_out)
                    )
    def forward(self, x):
        y = F.relu(x + self.conv2(self.conv1(x)))
        return F.relu(y + self.conv4(self.conv3(y)))

class ResBlockC(nn.Module):
    def __init__(self, channels_in, channels_out):
        super(ResBlockC, self).__init__()
        self.channels_in = channels_in
        self.channels_out = channels_out
        self.conv1 = nn.Sequential(
                        nn.Conv2d(self.channels_in, self.channels_out, 3, 2, 1),
                        nn.BatchNorm2d(self.channels_out),
                        nn.ReLU(),
                        nn.Conv2d(self.channels_out, self.channels_out, 3, 1, 1),
                        nn.BatchNorm2d(self.channels_out)
                    )
        self.conv2 = nn.Sequential(
                        nn.Conv2d(self.channels_in, self.channels_out, 3, 2, 1),
                        nn.BatchNorm2d(self.channels_out),
                    )
        self.conv3 = nn.Sequential(
                        nn.Conv2d(self.channels_out, self.channels_out, 3, 1, 1),
                        nn.BatchNorm2d(self.channels_out),
                        nn.ReLU(),
                        nn.Conv2d(self.channels_out, self.channels_out, 3, 1, 1),
                        nn.BatchNorm2d(self.channels_out)
                    )

    def forward(self, x):
        y = F.relu(self.conv1(x) + self.conv2(x))
        return F.relu(y + self.conv3(y))

class shape_net(nn.Module):
    def __init__(self, img_height, img_width):
        super(shape_net, self).__init__()
        self.height = img_height
        self.width = img_width
        self.conv1 = nn.Sequential(
                        nn.Conv2d(5, 64, 7, 1, 3),
                        nn.BatchNorm2d(64),
                        nn.MaxPool2d(3, 1, 1)
                    )
        self.conv2 = nn.Sequential(
                        nn.Conv2d(25, 2, 3, 1, 1),
                        nn.Softmax2d()
                    )
        self.ResBlockb = ResBlockB(64, 64)
        self.ResBlockc1 = ResBlockC(64, 128)
        self.ResBlockc2 = ResBlockC(128, 256)
        self.ResBlockc3 = ResBlockC(256, 512)
        self.branch_0 = nn.Sequential(nn.Upsample((256,256), mode='bilinear'),
                                      nn.Conv2d(64, 5, 3, 1, 1))
        self.branch_1 = nn.Sequential(nn.Upsample((256,256), mode='bilinear'),
                                      nn.Conv2d(64, 5, 3, 1, 1))
        self.branch_2 = nn.Sequential(nn.Upsample((256,256), mode='bilinear'),
                                      nn.Conv2d(128, 5, 3, 1, 1))
        self.branch_3 = nn.Sequential(nn.Upsample((256,256), mode='bilinear'),
                                      nn.Conv2d(256, 5, 3, 1, 1))
        self.branch_4 = nn.Sequential(nn.Upsample((256,256), mode='bilinear'),
                                      nn.Conv2d(512, 5, 3, 1, 1))
    def forward(self, x):
        x_0 = self.conv1(x)
        x_1 = self.ResBlockb(x_0)
        x_2 = self.ResBlockc1(x_1)
        x_3 = self.ResBlockc2(x_2)
        x_4 = self.ResBlockc3(x_3)
        y_0 = self.branch_0(x_0)
        y_1 = self.branch_1(x_1)
        y_2 = self.branch_2(x_2)
        y_3 = self.branch_3(x_3)
        y_4 = self.branch_4(x_4)
        y = torch.cat((y_0, y_1, y_2, y_3, y_4), dim=1)  # concat along the channel dimension
        return self.conv2(y)

class EgoNet(nn.Module):
    def __init__(self,):
        super(EgoNet, self).__init__()
        self.motion_net = nn.Sequential(
                            nn.Conv2d(1, 64, 7, 1, 3),
                            nn.BatchNorm2d(64),
                            nn.MaxPool2d(3, 1, 1),
                            ResBlockB(64, 64),
                            ResBlockC(64, 128),
                            ResBlockC(128, 256),
                            ResBlockC(256, 512),
                            nn.AdaptiveAvgPool2d((1,1))
                        )
        self.shape_net = shape_net(256, 256)
        self.shape_feature_net = nn.Sequential(
                                    nn.Conv2d(1, 64, kernel_size=3, stride=4, padding=1, dilation=2),
                                    nn.BatchNorm2d(64),
                                    nn.ReLU(),
                                    nn.MaxPool2d(3, stride=4, padding=1, dilation=2),
                                    nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, dilation=2),
                                    nn.BatchNorm2d(128),
                                    nn.ReLU(),
                                    nn.MaxPool2d(kernel_size=3, stride=1, padding=1, dilation=2),
                                    nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, dilation=2),
                                    nn.BatchNorm2d(256),
                                    nn.ReLU(),
                                    nn.MaxPool2d(kernel_size=3, stride=1, padding=1, dilation=2),
                                    nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, dilation=2),
                                    nn.BatchNorm2d(512),
                                    nn.ReLU(),
                                    nn.MaxPool2d(kernel_size=3, stride=1, padding=1, dilation=2),
                                )
        self.fc = nn.Linear(8192, 16)
        self.fc1 = nn.Linear(528, 45)
        self.fc2 = nn.Linear(528, 3)
        self.fc3 = nn.Linear(528, 3)

    def forward(self, foreground, motion):
        motion_feature = self.motion_net(motion)         # (b,512,1,1)
        if motion_feature.shape[0] == 1:
            motion_feature = self.motion_net(motion).squeeze().unsqueeze(0)
        else:
            motion_feature = self.motion_net(motion).squeeze()
        result = self.shape_net(foreground)        # (b,2,256,256)
        mask = result[:,1,:,:].unsqueeze(1)
        segmentation = result[:,0,:,:].unsqueeze(1)
        
        threshold = 0.5
        a = torch.zeros_like(mask)
        b = torch.ones_like(mask)
        # threshold  if segmentation < threshold, then = 0
        mask = torch.where(mask < threshold, a, b)
        segmentation = segmentation * mask
       
        shape_feature = self.shape_feature_net(segmentation).reshape(-1,8192)
        shape_feature = self.fc(shape_feature)
       
        # cat
        print("shape_feature: ", shape_feature.shape)
        print("motion_feature: ", motion_feature.shape)
        fusion_feature = torch.cat((motion_feature, shape_feature), dim=1)
        keypoint = self.fc1(fusion_feature)
        head1 = self.fc2(fusion_feature)
        head2 = self.fc3(fusion_feature)
        return keypoint, head1, head2
