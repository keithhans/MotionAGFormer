import json
import numpy as np

def process_json(json_path):
    # 读取JSON文件
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # 获取总帧数
    total_frames = len(data)
    
    # 获取人数（假设每帧的人数相同）
    num_persons = len(data[0]['instances'])
    
    # 初始化数组
    keypoints = np.zeros((num_persons, total_frames, 17, 2))
    scores = np.zeros((num_persons, total_frames, 17))
    
    # 遍历每一帧
    for frame_idx, frame_data in enumerate(data):
        # 遍历每个人
        for person_idx, instance in enumerate(frame_data['instances']):
            # 提取关键点坐标
            kpts = np.array(instance['keypoints'])  # shape: (17, 2)
            keypoints[person_idx, frame_idx] = kpts
            
            # 提取置信度分数
            conf = np.array(instance['keypoint_scores'])  # shape: (17,)
            scores[person_idx, frame_idx] = conf
    
    return keypoints, scores

def main():
    json_path = '/Users/keith/Downloads/src/MotionAGFormer/demo/execise1s.json'
    keypoints, scores = process_json(json_path)
    
    print(f"Keypoints shape: {keypoints.shape}")  # 应该是 (N, T, 17, 2)
    print(f"Scores shape: {scores.shape}")        # 应该是 (N, T, 17)
    
    # 验证数据是否正确读取
    print("\n数据验证:")
    print(f"第1个人第1帧的第1个关键点坐标: {keypoints[0, 0, 0]}")
    print(f"第1个人第1帧的第1个关键点置信度: {scores[0, 0, 0]}")

if __name__ == "__main__":
    main()