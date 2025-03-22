import numpy as np

def calculate_head_angles(hip, left_shoulder, right_shoulder, thorax, nose, head):
    """
    计算头部相对于身体的pitch和yaw角度
    所有输入都是shape为(3,)的numpy数组，表示3D坐标点
    返回pitch和yaw角度（单位：度）
    """
    # 1. 计算身体的坐标系
    body_up = thorax - hip  # 身体向上的向量
    shoulder_vec = right_shoulder - left_shoulder  # 肩部向量
    body_forward = np.cross(shoulder_vec, body_up)  # 身体前向的向量
    
    # 标准化向量
    body_up = body_up / np.linalg.norm(body_up)
    body_forward = body_forward / np.linalg.norm(body_forward)
    body_right = np.cross(body_forward, body_up)
    body_right = body_right / np.linalg.norm(body_right)
    
    # 2. 构建身体坐标系的旋转矩阵
    body_rotation = np.array([body_right, body_forward, body_up]).T
    
    # 3. 计算头部方向向量
    head_vec = head - nose  # 从鼻子指向头顶的向量
    neck_vec = nose - thorax  # 从胸部指向鼻子的向量
    head_forward = np.cross(head_vec, shoulder_vec)  # 头部前向量
    
    # 标准化头部向量
    neck_vec = neck_vec / np.linalg.norm(neck_vec)
    head_forward = head_forward / np.linalg.norm(head_forward)
    
    # 4. 将头部向量转换到身体坐标系
    neck_local = np.dot(body_rotation.T, neck_vec)
    head_forward_local = np.dot(body_rotation.T, head_forward)
    
    # 5. 计算pitch和yaw角度
    # pitch: 低头抬头角度（绕x轴）
    pitch = np.degrees(np.arctan2(-neck_local[1], neck_local[2]))
    
    # yaw: 左右转头角度（绕z轴）
    yaw = np.degrees(np.arctan2(head_forward_local[1], head_forward_local[0]))
    
    return pitch, yaw

# 测试代码
if __name__ == "__main__":
    # 示例坐标点
    hip = np.array([0, 0, 0])
    left_shoulder = np.array([-1, 0, 2])
    right_shoulder = np.array([1, 0, 2])
    thorax = np.array([0, 0, 2])
    nose = np.array([0, 0.2, 3])
    head = np.array([0, 0.2, 3.5])
    
    pitch, yaw = calculate_head_angles(hip, left_shoulder, right_shoulder, thorax, nose, head)
    print(f"Pitch (低头抬头): {pitch:.2f}度")
    print(f"Yaw (左右转头): {yaw:.2f}度")