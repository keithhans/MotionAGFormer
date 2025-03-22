import numpy as np

def calculate_calf_angles(left_hip, right_hip, left_knee, right_knee, left_foot, right_foot):
    """
    计算左右小腿相对大腿的夹角
    所有输入都是shape为(3,)的numpy数组，表示3D坐标点
    返回左右小腿与大腿的夹角（单位：度）
    """
    # 计算大腿和小腿的向量
    left_thigh = left_knee - left_hip
    right_thigh = right_knee - right_hip
    left_calf = left_foot - left_knee
    right_calf = right_foot - right_knee
    
    # 计算夹角（使用点积）
    left_cos = np.dot(left_thigh, left_calf) / (np.linalg.norm(left_thigh) * np.linalg.norm(left_calf))
    right_cos = np.dot(right_thigh, right_calf) / (np.linalg.norm(right_thigh) * np.linalg.norm(right_calf))
    
    # 限制cos值在[-1, 1]范围内
    left_cos = np.clip(left_cos, -1.0, 1.0)
    right_cos = np.clip(right_cos, -1.0, 1.0)
    
    # 计算角度
    left_angle = np.degrees(np.arccos(left_cos))
    right_angle = np.degrees(np.arccos(right_cos))
    
    return left_angle, right_angle

# 测试代码
if __name__ == "__main__":
    # 示例坐标点
    left_hip = np.array([-0.5, 0, 1])
    right_hip = np.array([0.5, 0, 1])
    left_knee = np.array([-0.6, 0.2, 0.5])
    right_knee = np.array([0.6, 0.2, 0.5])
    left_foot = np.array([-0.7, 0.3, 0])
    right_foot = np.array([0.7, 0.3, 0])
    
    left_angle, right_angle = calculate_calf_angles(
        left_hip, right_hip, left_knee, right_knee, left_foot, right_foot)
    
    print(f"左腿弯曲角度: {left_angle:.2f}度")
    print(f"右腿弯曲角度: {right_angle:.2f}度")