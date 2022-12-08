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
    pos_x = keypoints[0:51:3]
    pos_y = keypoints[1:51:3]
    pos_z = keypoints[2:51:3]
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
    radius = 30
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

if __name__=='__main__':
    keypoint = np.array([[-2.3714e-02, -7.5240e-01,  4.2565e+00, -2.3714e-02, -4.2565e+00,
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

    DrawSkeleton(keypoint[0, 6:], keypoint[0, 0:3], keypoint[0, 3:6])