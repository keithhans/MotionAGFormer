import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from convert import calculate_joint_angles

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
    first_frame = poses[0]

    rot =  [0.1407056450843811, -0.1500701755285263, -0.755240797996521, 0.6223280429840088]
    rot = np.array(rot, dtype='float32')
    pose_out = camera_to_world(first_frame, R=rot, t=0)
    pose_out[:, 2] -= np.min(pose_out[:, 2])
    max_value = np.max(pose_out)
    pose_out /= max_value

    
    # 计算关节角度
    angles = calculate_joint_angles(poses[0:1])[0]
    
    # 创建图形
    fig = plt.figure(figsize=(15, 5))
    
    # 3D骨架图
    ax1 = fig.add_subplot(121, projection='3d')
    plot_skeleton(ax1, pose_out, '3D Skeleton')
    
    # 关节角度图
    ax2 = fig.add_subplot(122)
    visualize_angles(ax2, angles)
    
    plt.tight_layout()
    plt.show()