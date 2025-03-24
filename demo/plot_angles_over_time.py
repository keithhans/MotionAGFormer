import numpy as np
import matplotlib.pyplot as plt

joint_angles = np.load('joint_angles.npy', allow_pickle=True)
frames = len(joint_angles)
time = np.arange(frames)

# 准备数据
head_pitch_data = []
head_yaw_data = []
left_arm_pitch_data = []
left_arm_roll_data = []
right_arm_pitch_data = []
right_arm_roll_data = []

for i, frame in enumerate(joint_angles):
    angles = frame['angles']
    
    # 限制角度范围
    head_pitch = np.clip(angles['head_pitch'], -50, 50)
    head_yaw = np.clip(angles['head_yaw'], -90, 90)
    left_arm_pitch = np.clip(angles['left_arm_pitch'], -180, 180)
    left_arm_roll = np.clip(angles['left_arm_roll'], -180, 180)
    right_arm_pitch = np.clip(angles['right_arm_pitch'], -180, 180)
    right_arm_roll = np.clip(angles['right_arm_roll'], -180, 180)

    print(f"Angle: {i} {head_pitch:.1f} {head_yaw:.1f} {left_arm_pitch:.1f} {left_arm_roll:.1f} {right_arm_pitch:.1f} {right_arm_roll:.1f}")

    # 收集数据
    head_pitch_data.append(head_pitch)
    head_yaw_data.append(head_yaw)
    left_arm_pitch_data.append(left_arm_pitch)
    left_arm_roll_data.append(left_arm_roll)
    right_arm_pitch_data.append(right_arm_pitch)
    right_arm_roll_data.append(right_arm_roll)

# 创建图表
plt.figure(figsize=(12, 8))

# 绘制头部角度
plt.subplot(2, 1, 1)
plt.plot(time, head_pitch_data, label='Head Pitch', color='blue')
plt.plot(time, head_yaw_data, label='Head Yaw', color='red')
plt.title('Head Angles over Time')
plt.xlabel('Frame')
plt.ylabel('Angle (degrees)')
plt.legend()
plt.grid(True)

# 绘制手臂角度
plt.subplot(2, 1, 2)
plt.plot(time, left_arm_pitch_data, label='Left Arm Pitch', color='green')
plt.plot(time, left_arm_roll_data, label='Left Arm Roll', color='purple')
plt.plot(time, right_arm_pitch_data, label='Right Arm Pitch', color='orange')
plt.plot(time, right_arm_roll_data, label='Right Arm Roll', color='brown')
plt.title('Arm Angles over Time')
plt.xlabel('Frame')
plt.ylabel('Angle (degrees)')
plt.legend()
plt.grid(True)

# 调整子图间距
plt.tight_layout()

# 显示图表
plt.show()