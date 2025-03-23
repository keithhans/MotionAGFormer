import numpy as np
from scipy.spatial.transform import Rotation as R


import sys
import os

# sys.path.append(os.getcwd())
from lib.utils import camera_to_world


def calculate_joint_angles(poses):
    """
    将3D关键点转换为机器人关节角度
    poses: [F, 17, 3] 数组，F是帧数，17是关键点数，3是xyz坐标
    """
    frames = poses.shape[0]
    joint_angles = []
    
    for frame in range(frames):
        pose = poses[frame]
        angles = {}
        
        # 计算躯干角度（相对于垂直线）
        spine_vec = pose[8] - pose[7]  # thorax - spine
        spine_angle = np.arctan2(spine_vec[0], spine_vec[1])
        angles['spine'] = np.degrees(spine_angle)
        
        # 计算左臂角度
        left_upper_arm = pose[12] - pose[11]  # left_elbow - left_shoulder
        left_forearm = pose[13] - pose[12]    # left_wrist - left_elbow
        
        # 计算左肩角度（相对于躯干）
        left_shoulder_angle = np.arctan2(left_upper_arm[2], left_upper_arm[1])
        angles['left_shoulder'] = np.degrees(left_shoulder_angle)
        
        # 计算左肘角度
        left_elbow_angle = calculate_angle(left_upper_arm, left_forearm)
        angles['left_elbow'] = left_elbow_angle
        
        # 计算右臂角度（类似左臂）
        right_upper_arm = pose[15] - pose[14]
        right_forearm = pose[16] - pose[15]
        
        right_shoulder_angle = np.arctan2(right_upper_arm[2], right_upper_arm[1])
        angles['right_shoulder'] = np.degrees(right_shoulder_angle)
        
        right_elbow_angle = calculate_angle(right_upper_arm, right_forearm)
        angles['right_elbow'] = right_elbow_angle
        
        # 计算腿部角度
        left_thigh = pose[5] - pose[4]    # left_knee - left_hip
        left_calf = pose[6] - pose[5]     # left_foot - left_knee
        
        left_hip_angle = np.arctan2(left_thigh[2], left_thigh[1])
        angles['left_hip'] = np.degrees(left_hip_angle)
        
        left_knee_angle = calculate_angle(left_thigh, left_calf)
        angles['left_knee'] = left_knee_angle
        
        # 右腿同理
        right_thigh = pose[2] - pose[1]
        right_calf = pose[3] - pose[2]
        
        right_hip_angle = np.arctan2(right_thigh[2], right_thigh[1])
        angles['right_hip'] = np.degrees(right_hip_angle)
        
        right_knee_angle = calculate_angle(right_thigh, right_calf)
        angles['right_knee'] = right_knee_angle
        
        joint_angles.append(angles)
    
    return joint_angles

def calculate_angle(vec1, vec2):
    """计算两个向量之间的角度"""
    cos_angle = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
    return np.degrees(angle)

if __name__ == "__main__":
    # 读取3D姿态数据
    poses = np.load('3d_poses.npy')
    
    new_poses = []

    for i, pose in enumerate(poses):
        rot =  [0.1407056450843811, -0.1500701755285263, -0.755240797996521, 0.6223280429840088]
        rot = np.array(rot, dtype='float32')
        post_out = camera_to_world(pose, R=rot, t=0)
        post_out[:, 2] -= np.min(post_out[:, 2])
        max_value = np.max(post_out)
        post_out /= max_value
        new_poses.append(post_out)

        if i == 0:
            print(post_out)

    poses = np.array(new_poses)

    # 计算关节角度
    joint_angles = calculate_joint_angles(poses)
    
    # 保存关节角度数据
    np.save('joint_angles.npy', joint_angles)