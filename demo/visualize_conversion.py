import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from convert import calculate_joint_angles
from calculate_arm_angles import calculate_left_arm_angles, calculate_right_arm_angles

import sys
import os

sys.path.append(os.getcwd())
from lib.utils import camera_to_world


def plot_skeleton(ax, pose, title):
    # 定义骨骼连接
    connections = [
        (0,1), (1,2), (2,3),    # 右腿
        (0,4), (4,5), (5,6),    # 左腿
        (0,7), (7,8),           # 脊椎
        (8,11), (11,12), (12,13),  # 左臂
        (8,14), (14,15), (15,16),  # 右臂
        (8,9), (9,10)           # 头部
    ]
    
    # 绘制骨骼
    for start, end in connections:
        x = [pose[start,0], pose[end,0]]
        y = [pose[start,1], pose[end,1]]
        z = [pose[start,2], pose[end,2]]
        ax.plot(x, y, z, 'b-')
    
    # 绘制关节点
    ax.scatter(pose[:,0], pose[:,1], pose[:,2], c='r', marker='o')
    
    # 设置坐标轴比例一致
    ax.set_box_aspect([1,1,1])  # 设置三个轴的比例相同
    
    # 确保显示范围一致
    max_range = np.array([
        pose[:,0].max()-pose[:,0].min(),
        pose[:,1].max()-pose[:,1].min(),
        pose[:,2].max()-pose[:,2].min()
    ]).max() / 2.0
    
    mid_x = (pose[:,0].max()+pose[:,0].min()) * 0.5
    mid_y = (pose[:,1].max()+pose[:,1].min()) * 0.5
    mid_z = (pose[:,2].max()+pose[:,2].min()) * 0.5
    
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    ax.set_title(title)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

def visualize_angles(ax, angles):
    # 创建一个简单的示意图显示关节角度
    # fig, ax = plt.subplots(figsize=(10, 6))
    y_pos = np.arange(len(angles))
    
    ax.barh(y_pos, list(angles.values()))
    ax.set_yticks(y_pos)
    ax.set_yticklabels(angles.keys())
    ax.set_xlabel('Angle (degrees)')
    ax.set_title('Joint Angles')

if __name__ == "__main__":
    # 读取数据
    poses = np.load('3d_poses.npy')
    
    # 获取第一帧
    first_frame = poses[50]

    rot =  [0.1407056450843811, -0.1500701755285263, -0.755240797996521, 0.6223280429840088]
    rot = np.array(rot, dtype='float32')
    pose_out = camera_to_world(first_frame, R=rot, t=0)
    pose_out[:, 2] -= np.min(pose_out[:, 2])
    max_value = np.max(pose_out)
    pose_out /= max_value

    print(pose_out)
    
    # 计算左臂角度
    left_arm_pitch, left_arm_roll = calculate_left_arm_angles(
        pose_out[0],  # hip
        pose_out[11], # left_shoulder
        pose_out[14], # right_shoulder
        pose_out[12]  # left_elbow
    )

    print(f"左臂 Pitch (前后摆动): {left_arm_pitch:.2f}度")
    print(f"左臂 Roll (左右摆动): {left_arm_roll:.2f}度")

    # 计算右臂角度
    right_arm_pitch, right_arm_roll = calculate_right_arm_angles(
        pose_out[0],  # hip
        pose_out[11], # left_shoulder
        pose_out[14], # right_shoulder
        pose_out[15]  # right_elbow
    )

    print(f"右臂 Pitch (前后摆动): {right_arm_pitch:.2f}度")
    print(f"右臂 Roll (左右摆动): {right_arm_roll:.2f}度")

    # 计算关节角度
    angles = calculate_joint_angles(poses[0:1])[0]
    
    # 创建图形
    fig = plt.figure(figsize=(8, 8))
    
    # 3D骨架图
    ax = fig.add_subplot(111, projection='3d')
    plot_skeleton(ax, pose_out, '3D Skeleton')
    
    plt.tight_layout()
    plt.show()