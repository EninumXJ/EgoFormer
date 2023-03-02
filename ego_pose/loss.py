import torch

def BoneLength(joint1, joint2):
    length = torch.sqrt(torch.sum((joint1-joint2)**2, dim=-1)).sum()
    return length

def ComputeLoss(keypoints, label, L=15):
    # label = label.squeeze()
    f = label[..., 0:3]
    u = label[..., 3:6]
    f_g = keypoints[..., 0:3]
    u_g = keypoints[..., 3:6]
    keypoints_g = label[..., 6:]
    loss_d = torch.sum(torch.abs(f_g-f)) + torch.sum(torch.abs(u_g-u)) + torch.sum(torch.abs(keypoints[..., 6:]- keypoints_g))
    # print("loss_d: ", loss_d.shape)
    loss_o = torch.sum(torch.abs(f*u)) + \
             torch.sum(torch.abs(torch.norm(f)-1)) + \
             torch.sum(torch.abs(torch.norm(u)-1))
    # print("loss_o: ", loss_o.shape)
    RightShoulder = keypoints[..., 9:12]
    RightArm = keypoints[..., 12:15]
    RightHand = keypoints[..., 15:18]
    LeftShoulder = keypoints[..., 18:21]
    LeftArm = keypoints[..., 21:24]
    LeftHand = keypoints[..., 24:27]
    RightUpLeg = keypoints[..., 27:30]
    RightLeg = keypoints[..., 30:33]
    RightFoot = keypoints[..., 33:36]
    LeftUpLeg = keypoints[..., 36:39] 
    LeftLeg = keypoints[..., 39:42]
    LeftFoot = keypoints[..., 42:45]
    RightBone1 = BoneLength(RightShoulder, RightArm)
    RightBone2 = BoneLength(RightArm, RightHand)
    RightBone3 = BoneLength(RightUpLeg, RightLeg)
    RightBone4 = BoneLength(RightLeg, RightFoot)
    LeftBone1 = BoneLength(LeftShoulder, LeftArm)
    LeftBone2 = BoneLength(LeftArm, LeftHand)
    LeftBone3 = BoneLength(LeftUpLeg, LeftLeg)
    LeftBone4 = BoneLength(LeftLeg, LeftFoot)
    loss_s = torch.abs(RightBone1-LeftBone1) + torch.abs(RightBone2-LeftBone2) + \
             torch.abs(RightBone3-LeftBone3) + torch.abs(RightBone4-LeftBone4)
    # print("loss_s: ", loss_s.shape)
    return (loss_s + loss_d + loss_o)/L

###
# order = 'xyz'时：label和keypoints的排序方式为：(xyz,xyz,xyz)
# order = 'xxx'时：label和keypoints的排序方式为：(xxx,yyy,zzz)
def ComputeLoss_nohead(keypoints, label, L=15, order='xyz'):
    # label = label.squeeze()
    keypoints_g = label
    loss_d = torch.sum(torch.abs(keypoints - keypoints_g))
    # print("loss_d: ", loss_d.shape)
    # print("loss_o: ", loss_o.shape)
    if order == 'xyz':
        RightShoulder = keypoints[..., 9:12]
        RightArm = keypoints[..., 12:15]
        RightHand = keypoints[..., 15:18]
        LeftShoulder = keypoints[..., 18:21]
        LeftArm = keypoints[..., 21:24]
        LeftHand = keypoints[..., 24:27]
        RightUpLeg = keypoints[..., 27:30]
        RightLeg = keypoints[..., 30:33]
        RightFoot = keypoints[..., 33:36]
        LeftUpLeg = keypoints[..., 36:39] 
        LeftLeg = keypoints[..., 39:42]
        LeftFoot = keypoints[..., 42:45]
    if order == 'xxx':
        RightShoulder = torch.cat(keypoints[..., 3], keypoints[..., 18], keypoints[..., 33], dim=-1)
        RightArm = torch.cat(keypoints[..., 4], keypoints[..., 19], keypoints[..., 34], dim=-1)
        RightHand = torch.cat(keypoints[..., 5], keypoints[..., 20], keypoints[..., 35], dim=-1)
        LeftShoulder = torch.cat(keypoints[..., 6], keypoints[..., 21], keypoints[..., 36], dim=-1)
        LeftArm = torch.cat(keypoints[..., 7], keypoints[..., 22], keypoints[..., 37], dim=-1)
        LeftHand = torch.cat(keypoints[..., 8], keypoints[..., 23], keypoints[..., 38], dim=-1)
        RightUpLeg = torch.cat(keypoints[..., 9], keypoints[..., 24], keypoints[..., 39], dim=-1)
        RightLeg = torch.cat(keypoints[..., 10], keypoints[..., 25], keypoints[..., 40], dim=-1)
        RightFoot = torch.cat(keypoints[..., 11], keypoints[..., 26], keypoints[..., 41], dim=-1)
        LeftUpLeg = torch.cat(keypoints[..., 12], keypoints[..., 27], keypoints[..., 42], dim=-1)
        LeftLeg = torch.cat(keypoints[..., 13], keypoints[..., 28], keypoints[..., 43], dim=-1)
        LeftFoot = torch.cat(keypoints[..., 14], keypoints[..., 29], keypoints[..., 44], dim=-1)

    RightBone1 = BoneLength(RightShoulder, RightArm)
    RightBone2 = BoneLength(RightArm, RightHand)
    RightBone3 = BoneLength(RightUpLeg, RightLeg)
    RightBone4 = BoneLength(RightLeg, RightFoot)
    LeftBone1 = BoneLength(LeftShoulder, LeftArm)
    LeftBone2 = BoneLength(LeftArm, LeftHand)
    LeftBone3 = BoneLength(LeftUpLeg, LeftLeg)
    LeftBone4 = BoneLength(LeftLeg, LeftFoot)
    loss_s = torch.abs(RightBone1-LeftBone1) + torch.abs(RightBone2-LeftBone2) + \
            torch.abs(RightBone3-LeftBone3) + torch.abs(RightBone4-LeftBone4)

    # print("loss_s: ", loss_s.shape)
    return (loss_s + loss_d)/L

if __name__=='__main__':
    label = torch.Tensor([[-2.3714e-02, -7.5240e-01,  4.2565e+00, -2.3714e-02, -4.2565e+00,
                -7.5240e-01,  0.0000e+00,  0.0000e+00,  0.0000e+00,  3.0252e-02,
                1.2480e+00,  1.9981e+01,  6.5384e-03,  4.9558e-01,  2.4238e+01,
                -1.0441e+00,  6.6327e-01,  1.7591e+01, -6.4229e+00,  7.9788e-01,
                1.7664e+01, -2.4902e+01,  1.3036e+00,  1.7849e+01,  1.0831e+00,
                6.1483e-01,  1.7592e+01,  6.4619e+00,  4.9831e-01,  1.7686e+01,
                2.4944e+01,  1.6264e-01,  1.7918e+01, -3.2153e+00,  1.6674e-02,
                -9.3478e-01, -3.2356e+00,  1.1299e-01, -1.6593e+01, -3.2514e+00,
                2.5413e-01, -3.0662e+01,  3.2123e+00, -2.5620e-03, -9.4536e-01,
                3.1813e+00, -9.2077e-03, -1.6604e+01,  3.1584e+00, -4.3270e-03,
                -3.0673e+01],
                [-2.3714e-02, -7.5240e-01,  4.2565e+00, -2.3714e-02, -4.2565e+00,
                -7.5240e-01,  0.0000e+00,  0.0000e+00,  0.0000e+00,  3.0252e-02,
                1.2480e+00,  1.9981e+01,  6.5384e-03,  4.9558e-01,  2.4238e+01,
                -1.0441e+00,  6.6327e-01,  1.7591e+01, -6.4229e+00,  7.9788e-01,
                1.7664e+01, -2.4902e+01,  1.3036e+00,  1.7849e+01,  1.0831e+00,
                6.1483e-01,  1.7592e+01,  6.4619e+00,  4.9831e-01,  1.7686e+01,
                2.4944e+01,  1.6264e-01,  1.7918e+01, -3.2153e+00,  1.6674e-02,
                -9.3478e-01, -3.2356e+00,  1.1299e-01, -1.6593e+01, -3.2514e+00,
                2.5413e-01, -3.0662e+01,  3.2123e+00, -2.5620e-03, -9.4536e-01,
                3.1813e+00, -9.2077e-03, -1.6604e+01,  3.1584e+00, -4.3270e-03,
                -3.0673e+01]])
    keypoints = torch.ones(2, 45)
    head1 = torch.ones(2, 3)
    head2 = torch.zeros(2, 3)
    print(keypoints.shape)
    print(label.shape)
    loss = ComputeLoss(keypoints, head1, head2, label)
    print(loss)