from bvh import Bvh
import pickle
from mocap.pose import load_bvh_file
from mocap.skeleton import Skeleton

fname = '/data1/lty/dataset/egopose_dataset/datasets/traj/1205_take_15.bvh'
with open(fname) as f:
    mocap = Bvh(f.read())
print("Number of joints: ",len(mocap.get_joints_names()))
print("Number of frames: ", mocap.nframes)
print(mocap.get_joints_names())
pname = '/data1/lty/dataset/egopose_dataset/datasets/traj/1205_take_15_traj.p'
traj = pickle.load(open(pname, 'rb'))
print(traj[0].shape)

skeleton = Skeleton()
exclude_bones = {'Thumb', 'Index', 'Middle', 'Ring', 'Pinky', 'End', 'Toe'}
spec_channels = {'LeftForeArm': ['Zrotation'], 'RightForeArm': ['Zrotation'],
                 'LeftLeg': ['Xrotation'], 'RightLeg': ['Xrotation']}
skeleton.load_from_bvh(fname, exclude_bones, spec_channels)
poses, bone_addr = load_bvh_file(fname, skeleton)
print("poses: ", poses)
print("bone_addr: ", bone_addr)