import numpy as np

def calculate_head_angles(hip, left_shoulder, right_shoulder, thorax, nose, head, verbose=False):
    if verbose:
        print("\n=== 计算头部角度 ===")
    
    # 1. 计算身体的坐标系
    body_up = thorax - hip
    shoulder_vec = right_shoulder - left_shoulder
    body_forward = np.cross(body_up, shoulder_vec)
    
    if verbose:
        print("\n1. 原始向量:")
        print(f"身体向上向量: {body_up}")
        print(f"肩部向量: {shoulder_vec}")
        print(f"身体前向量: {body_forward}")
    
    # 标准化向量
    body_up = body_up / np.linalg.norm(body_up)
    body_forward = body_forward / np.linalg.norm(body_forward)
    body_right = np.cross(body_forward, body_up)
    body_right = body_right / np.linalg.norm(body_right)
    
    if verbose:
        print("\n2. 标准化后的身体坐标系基向量:")
        print(f"右向量 (X): {body_right}")
        print(f"前向量 (Y): {body_forward}")
        print(f"上向量 (Z): {body_up}")
    
    # 构建身体坐标系的旋转矩阵
    body_rotation = np.array([body_right, body_forward, body_up]).T
    if verbose:
        print("\n3. 身体坐标系旋转矩阵:")
        print(body_rotation)
    
    # 计算头部方向向量
    head_vec = head - nose
    neck_vec = nose - thorax
    head_forward = np.cross(head_vec, shoulder_vec)
    
    if verbose:
        print("\n4. 头部向量:")
        print(f"头顶向量: {head_vec}")
        print(f"颈部向量: {neck_vec}")
        print(f"头部前向量: {head_forward}")
    
    # 标准化头部向量
    neck_vec = neck_vec / np.linalg.norm(neck_vec)
    head_forward = head_forward / np.linalg.norm(head_forward)
    
    if verbose:
        print("\n5. 标准化后的头部向量:")
        print(f"颈部向量: {neck_vec}")
        print(f"头部前向量: {head_forward}")
    
    # 将头部向量转换到身体坐标系
    neck_local = np.dot(body_rotation.T, neck_vec)
    head_forward_local = np.dot(body_rotation.T, head_forward)
    
    if verbose:
        print("\n6. 局部坐标系中的头部向量:")
        print(f"局部颈部向量: {neck_local}")
        print(f"局部头部前向量: {head_forward_local}")
    
    # 计算pitch和yaw角度
    pitch = np.degrees(np.arctan2(-neck_local[1], neck_local[2]))
    yaw = np.degrees(np.arctan2(neck_local[0], neck_local[1]))
    
    if verbose:
        print("\n7. 最终角度:")
        print(f"Pitch (低头抬头): {pitch:.2f}度")
        print(f"Yaw (左右转头): {yaw:.2f}度")
    
    return pitch, yaw

# 测试代码
if __name__ == "__main__":
    # 示例坐标点
    # hip = np.array([0, 0, 0])
    # left_shoulder = np.array([-1, 0, 2])
    # right_shoulder = np.array([1, 0, 2])
    # thorax = np.array([0, 0, 2])
    # nose = np.array([0.1, 0.2, 3])
    # head = np.array([0.1, 0.2, 3.5])

    hip = np.array([0, 0, 0.5154449])
    left_shoulder = np.array([-0.09552623, -0.02224409,  0.84772354])
    right_shoulder = np.array([0.1008563,  -0.04602984,  0.8566299 ])
    thorax = np.array([-0.01926025, -0.0249161,   0.8698806])
    nose = np.array([-0.03845415,  0.00920062,  0.9364334])
    head = np.array([-0.03470165, -0.01818796,  1.        ])


    pitch, yaw = calculate_head_angles(hip, left_shoulder, right_shoulder, thorax, nose, head, verbose=True)
    print(f"Pitch (低头抬头): {pitch:.2f}度")
    print(f"Yaw (左右转头): {yaw:.2f}度")