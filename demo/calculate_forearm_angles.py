import numpy as np


def calculate_left_forearm_angle(left_shoulder, left_elbow, left_wrist):
    """
    计算左前臂相对于左上臂的旋转角度
    所有输入都是shape为(3,)的numpy数组，表示3D坐标点
    返回旋转角度（单位：度）
    """
    # 计算上臂和前臂的向量
    upper_arm = left_elbow - left_shoulder
    forearm = left_wrist - left_elbow
    
    # 标准化向量
    upper_arm = upper_arm / np.linalg.norm(upper_arm)
    forearm = forearm / np.linalg.norm(forearm)
    
    # 计算夹角
    cos_angle = np.dot(upper_arm, forearm)
    angle = np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))
    
    return angle


def calculate_right_forearm_angle(right_shoulder, right_elbow, right_wrist):
    """
    计算右前臂相对于右上臂的旋转角度
    所有输入都是shape为(3,)的numpy数组，表示3D坐标点
    返回旋转角度（单位：度）
    """
    # 计算上臂和前臂的向量
    upper_arm = right_elbow - right_shoulder
    forearm = right_wrist - right_elbow
    
    # 标准化向量
    upper_arm = upper_arm / np.linalg.norm(upper_arm)
    forearm = forearm / np.linalg.norm(forearm)
    
    # 计算夹角
    cos_angle = np.dot(upper_arm, forearm)
    angle = np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))
    
    return angle


# 测试代码
if __name__ == "__main__":
    # 示例坐标点
    left_shoulder = np.array([-0.09552623, -0.02224409, 0.84772354])
    left_elbow = np.array([-0.23032123,  0.06436861,  0.88997406])
    left_wrist = np.array([-0.39214453,  0.1720497,   0.9211942])
    # left_shoulder = np.array([-1, 0, 1])
    # left_elbow = np.array([-2, 1, 1])
    # left_wrist = np.array([-3, 2, 1])
    left_angle = calculate_left_forearm_angle(left_shoulder, left_elbow, left_wrist)
    print(f"左肘关节角度: {left_angle:.2f}度")
    
    # 示例坐标点 - 右臂（对称于左臂）
    right_shoulder =  np.array([-0.1008563, -0.04602984,  0.8566299])
    right_elbow = np.array([-0.25813976,  0.02717768,  0.8320716])
    right_wrist = np.array([-0.25698468,  0.09457579,   0.92279565 ])

    # right_shoulder = np.array([1, 0, 1])
    # right_elbow = np.array([2, 1, 1])
    # right_wrist = np.array([3, 2, 1])
    right_angle = calculate_right_forearm_angle(right_shoulder, right_elbow, right_wrist)
    print(f"右肘关节角度: {right_angle:.2f}度")