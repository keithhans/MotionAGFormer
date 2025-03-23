import numpy as np

def calculate_left_arm_angles(hip, left_shoulder, right_shoulder, left_elbow):
    """
    计算左上臂相对身体的pitch和roll角度
    所有输入都是shape为(3,)的numpy数组，表示3D坐标点
    返回pitch和roll角度（单位：度）
    """
    # 1. 计算身体的前向向量和上向向量
    body_up = (left_shoulder + right_shoulder) / 2 - hip  # 身体向上的向量
    shoulder_vec = right_shoulder - left_shoulder  # 肩部向量
    body_forward = np.cross(body_up, shoulder_vec)  # 身体前向的向量（上向量叉乘肩部向量得到前向量）
    body_right = np.cross(body_forward, body_up)  # 身体右向的向量（前向量叉乘上向量得到右向量）
        
    # 2. 标准化这些向量
    body_up = body_up / np.linalg.norm(body_up)
    body_forward = body_forward / np.linalg.norm(body_forward)
    body_right = body_right / np.linalg.norm(body_right)
    
    # 3. 构建身体坐标系的旋转矩阵
    body_rotation = np.array([body_right, body_forward, body_up]).T
    
    # 4. 计算上臂向量在身体坐标系中的表示
    upper_arm = left_elbow - left_shoulder
    upper_arm = upper_arm / np.linalg.norm(upper_arm)
    
    # 5. 将上臂向量转换到身体坐标系
    upper_arm_local = np.dot(body_rotation.T, upper_arm)
    
    # 6. 计算pitch和roll角度
    pitch = np.degrees(np.arctan2(-upper_arm_local[1], -upper_arm_local[2]))  # 前后摆动
    roll = np.degrees(np.arctan2(upper_arm_local[0], -upper_arm_local[2]))    # 左右摆动
    
    return pitch, roll

def calculate_right_arm_angles(hip, left_shoulder, right_shoulder, right_elbow):
    """
    计算右上臂相对身体的pitch和roll角度
    所有输入都是shape为(3,)的numpy数组，表示3D坐标点
    返回pitch和roll角度（单位：度）
    """
    # 1. 计算身体的前向向量和上向向量
    body_up = (left_shoulder + right_shoulder) / 2 - hip  # 身体向上的向量
    shoulder_vec = right_shoulder - left_shoulder  # 肩部向量
    body_forward = np.cross(body_up, shoulder_vec)  # 身体前向的向量（上向量叉乘肩部向量得到前向量）
    body_right = np.cross(body_forward, body_up)  # 身体右向的向量（前向量叉乘上向量得到右向量）
    
    # 2. 标准化这些向量
    body_up = body_up / np.linalg.norm(body_up)
    body_forward = body_forward / np.linalg.norm(body_forward)
    body_right = body_right / np.linalg.norm(body_right)
    
    # 3. 构建身体坐标系的旋转矩阵
    body_rotation = np.array([body_right, body_forward, body_up]).T
    
    # 4. 计算上臂向量在身体坐标系中的表示
    upper_arm = right_elbow - right_shoulder
    upper_arm = upper_arm / np.linalg.norm(upper_arm)
    
    # 5. 将上臂向量转换到身体坐标系
    upper_arm_local = np.dot(body_rotation.T, upper_arm)
    
    # 6. 计算pitch和roll角度
    pitch = np.degrees(np.arctan2(-upper_arm_local[1], -upper_arm_local[2]))  # 前后摆动
    roll = np.degrees(np.arctan2(upper_arm_local[0], -upper_arm_local[2]))    # 左右摆动
    
    return pitch, roll

# 测试代码
if __name__ == "__main__":
    # 示例坐标点（来自notes.txt）
    # hip = np.array([0, 0, 0.5154449])
    # left_shoulder = np.array([-0.09552623, -0.02224409, 0.84772354])
    # right_shoulder = np.array([0.1008563, -0.04602984, 0.8566299])
    # left_elbow = np.array([-0.23032123, 0.06436861, 0.88997406])
    # right_elbow = np.array([0.25813976, -0.02717768, 0.8320716])    

    # 示例坐标点（简化版）
    hip = np.array([0, 0, 0])  # 原点
    left_shoulder = np.array([-1, 0, 2])  # 左肩在xz平面上
    right_shoulder = np.array([1, 0, 2])  # 右肩在xz平面上，与左肩对称
    left_elbow = np.array([-2, 1, 1])  # 左肘向前抬起
    right_elbow = np.array([2, 1, 1])  # 右肘向前抬起
    
    # 计算左臂角度
    left_pitch, left_roll = calculate_left_arm_angles(hip, left_shoulder, right_shoulder, left_elbow)
    print(f"左臂 Pitch (前后摆动): {left_pitch:.2f}度")
    print(f"左臂 Roll (左右摆动): {left_roll:.2f}度")
    
    # 计算右臂角度
    right_pitch, right_roll = calculate_right_arm_angles(hip, left_shoulder, right_shoulder, right_elbow)
    print(f"右臂 Pitch (前后摆动): {right_pitch:.2f}度")
    print(f"右臂 Roll (左右摆动): {right_roll:.2f}度")
