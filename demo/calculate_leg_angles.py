import numpy as np

def calculate_leg_angles(left_hip, right_hip, spine, left_knee, right_knee):
    """
    计算左右大腿相对身体的pitch和roll角度
    所有输入都是shape为(3,)的numpy数组，表示3D坐标点
    返回左右腿的pitch和roll角度（单位：度）
    """
    # 1. 计算身体的坐标系
    body_up = spine - (left_hip + right_hip) / 2  # 身体向上的向量
    hip_vec = right_hip - left_hip  # 髋部向量
    body_forward = np.cross(body_up, hip_vec)  # 身体前向的向量
    body_right = np.cross(body_forward, body_up)  # 身体右向的向量
    
    # 标准化向量
    body_up = body_up / np.linalg.norm(body_up)
    body_forward = body_forward / np.linalg.norm(body_forward)
    body_right = body_right / np.linalg.norm(body_right)
    
    # 2. 构建身体坐标系的旋转矩阵
    body_rotation = np.array([body_right, body_forward, body_up]).T
    
    # 3. 计算左右大腿向量
    left_thigh = left_knee - left_hip
    right_thigh = right_knee - right_hip
    
    # 标准化大腿向量
    left_thigh = left_thigh / np.linalg.norm(left_thigh)
    right_thigh = right_thigh / np.linalg.norm(right_thigh)
    
    # 4. 将大腿向量转换到身体坐标系
    left_thigh_local = np.dot(body_rotation.T, left_thigh)
    right_thigh_local = np.dot(body_rotation.T, right_thigh)
    
    # 5. 计算左腿的pitch和roll角度
    left_pitch = np.degrees(np.arctan2(-left_thigh_local[1], -left_thigh_local[2]))  # 前后摆动
    left_roll = np.degrees(np.arctan2(left_thigh_local[0], -left_thigh_local[2]))    # 左右摆动
    
    # 6. 计算右腿的pitch和roll角度
    right_pitch = np.degrees(np.arctan2(-right_thigh_local[1], -right_thigh_local[2]))  # 前后摆动
    right_roll = np.degrees(np.arctan2(right_thigh_local[0], -right_thigh_local[2]))    # 左右摆动
    
    return left_pitch, left_roll, right_pitch, right_roll

# 测试代码
if __name__ == "__main__":
    # 示例坐标点
    left_hip = np.array([-0.5, 0, 1])
    right_hip = np.array([0.5, 0, 1])
    spine = np.array([0, 0, 2])
    left_knee = np.array([-0.5, 0.2, 0.5])   # 左膝盖稍微向前
    right_knee = np.array([0.5, 0.2, 0.5])   # 右膝盖稍微向前
    
    left_pitch, left_roll, right_pitch, right_roll = calculate_leg_angles(
        left_hip, right_hip, spine, left_knee, right_knee)
    
    print(f"左腿 Pitch (前后摆动): {left_pitch:.2f}度")
    print(f"左腿 Roll (左右摆动): {left_roll:.2f}度")
    print(f"右腿 Pitch (前后摆动): {right_pitch:.2f}度")
    print(f"右腿 Roll (左右摆动): {right_roll:.2f}度")