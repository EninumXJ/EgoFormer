import matplotlib
import numpy as np

matplotlib.use('Agg')
import matplotlib.pyplot as plt

# keypoints: ndarray [1, 51]
def DrawSkeleton(keypoints, head1=None, head2=None, image_name='Skeleton.jpg'):
    # pos_x = keypoints[0:45:3]
    # pos_y = keypoints[1:45:3]
    # pos_z = keypoints[2:45:3]
    # head = keypoints[6:9]
    pos_x = keypoints[0:len(keypoints):3]
    pos_y = keypoints[1:len(keypoints):3]
    pos_z = keypoints[2:len(keypoints):3]
    head = keypoints[6:9]

    xp = pos_x.T
    yp = pos_y.T
    zp = pos_z.T
    ax = plt.axes(projection='3d')
    if head1==None and head2==None:
        pass
    else:
        f = head1
        f = f/np.sqrt((f[0]**2 + f[1]**2 + f[2]**2))
        print(f)
        u = head2
        u = u/np.sqrt((u[0]**2 + u[1]**2 + u[2]**2))
        print(u)
        #画起点为head,终点为f_end的向量
        ax.quiver(head[0], head[1], head[2], f[0]*10, f[1]*10, f[2]*10, color='green', arrow_length_ratio=0.2)
        #画起点为head,终点为u_end的向量
        ax.quiver(head[0], head[1], head[2], u[0]*10, u[1]*10, u[2]*10, color='blue', arrow_length_ratio=0.2)
    radius = 1
    ax.set_xlim3d([-radius, radius])
    ax.set_zlim3d([-radius, radius])
    ax.set_ylim3d([-radius, radius])
    ax.view_init(elev=15., azim=70)
    
    ax.dist = 7.5

    # 3D scatter
    ax.scatter3D(xp, yp, zp, cmap='Greens')
    
    # hips, neck, head, node [0, 1, 2]
    ax.plot(xp[0:3], yp[0:3], zp[0:3], ls='-', color='gray')
    
    # RightShoulder, RightArm, RightForeArm, RightHand
    ax.plot(xp[3:7], yp[3:7], zp[3:7], ls='-', color='blue')
    # LeftShoulder, LeftArm, LeftForeArm, LeftHand
    ax.plot(xp[7:11], yp[7:11], zp[7:11], ls='-', color='red')
    # RightUpLeg, RightLeg, RightFoot
    ax.plot(xp[11:14], yp[11:14], zp[11:14], ls='-', color='blue')
    
    # LeftUpLeg, LeftLeg, LeftFoot
    ax.plot(xp[14:17], yp[14:17], zp[14:17], ls='-', color='red')

    plt.savefig(image_name, dpi=300)

# keypoints: ndarray [1, 51]
def DrawSkeleton45(keypoints, head1=None, head2=None, image_name='Skeleton.jpg'):
    pos_x = keypoints[0:len(keypoints):3]
    pos_y = keypoints[1:len(keypoints):3]
    pos_z = keypoints[2:len(keypoints):3]
    head = keypoints[6:9]

    xp = pos_x.T
    yp = pos_y.T
    zp = pos_z.T
    ax = plt.axes(projection='3d')
    if head1 == None and head2 == None:
        pass
    else:
        f = head1
        f = f/np.sqrt((f[0]**2 + f[1]**2 + f[2]**2))
        print(f)
        u = head2
        u = u/np.sqrt((u[0]**2 + u[1]**2 + u[2]**2))
        print(u)
        #画起点为head,终点为f_end的向量
        ax.quiver(head[0], head[1], head[2], f[0]*10, f[1]*10, f[2]*10, color='green', arrow_length_ratio=0.2)
        #画起点为head,终点为u_end的向量
        ax.quiver(head[0], head[1], head[2], u[0]*10, u[1]*10, u[2]*10, color='blue', arrow_length_ratio=0.2)
    radius = 1
    ax.set_xlim3d([-radius, radius])
    ax.set_zlim3d([-radius, radius])
    ax.set_ylim3d([-radius, radius])
    ax.view_init(elev=15., azim=70)
    
    ax.dist = 7.5

    # 3D scatter
    ax.scatter3D(xp, yp, zp, cmap='Greens')
    
    # hips, neck, head, node [0, 1, 2]
    ax.plot(xp[0:3], yp[0:3], zp[0:3], ls='-', color='gray')
    
    # RightShoulder, RightArm, RightHand
    ax.plot(xp[3:6], yp[3:6], zp[3:6], ls='-', color='blue')
    # LeftShoulder, LeftArm, LeftHand
    ax.plot(xp[6:9], yp[6:9], zp[6:9], ls='-', color='red')
    # RightUpLeg, RightLeg, RightFoot
    ax.plot(xp[9:12], yp[9:12], zp[9:12], ls='-', color='blue')
    
    # LeftUpLeg, LeftLeg, LeftFoot
    ax.plot(xp[12:15], yp[12:15], zp[12:15], ls='-', color='red')

    plt.savefig(image_name, dpi=300)

if __name__=='__main__':
    keypoint = np.array([[ 0.0945,  0.0125,  0.1152,  0.1040,  0.0359,  0.8434,  0.1053,  0.0562,
          1.0000,  0.2202,  0.0418,  0.7179,  0.2825,  0.0251,  0.5322,  0.5138,
          0.1713, -0.0814,  0.0886,  0.1202,  0.7150,  0.0126,  0.1453,  0.5355,
          0.1286,  0.5460,  0.0444,  0.1807, -0.0630,  0.0725,  0.0708, -0.1156,
         -0.4864,  0.0570, -0.1081, -1.0000, -0.0198,  0.0471,  0.0886, -0.0831,
         -0.0822, -0.4649, -0.0998, -0.1110, -0.9777]])
    print(keypoint[0, :].shape)
    # DrawSkeleton(keypoint[0, 6:], keypoint[0, 0:3], keypoint[0, 3:6])
    DrawSkeleton45(keypoint[0, :])