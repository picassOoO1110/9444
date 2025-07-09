import json
import cv2
import os
import numpy as np
from pathlib import Path

def rotate_points(points, angle, image_shape):
    """旋转点坐标"""
    center = (image_shape[1] // 2, image_shape[0] // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    points = np.array(points).reshape(-1, 2)
    ones = np.ones(shape=(len(points), 1))
    points_ones = np.hstack([points, ones])
    
    transformed_points = matrix.dot(points_ones.T).T
    
    return transformed_points.flatten().tolist()

def augment_dataset(input_dir, label_dir, output_dir):
    # 创建输出目录
    image_output_dir = os.path.join(output_dir, 'images')
    label_output_dir = os.path.join(output_dir, 'labels')
    os.makedirs(image_output_dir, exist_ok=True)
    os.makedirs(label_output_dir, exist_ok=True)
    
    # 获取所有图片文件
    image_files = list(Path(input_dir).glob('*.png'))
    
    for img_path in image_files:
        try:
            # 读取图片
            image = cv2.imread(str(img_path))
            if image is None:
                print(f"Warning: Could not read image {img_path}")
                continue
                
            # 构造对应的JSON文件路径
            json_path = Path(label_dir) / img_path.with_suffix('.json').name
            if not json_path.exists():
                print(f"Warning: No JSON file found for {img_path}")
                continue
                
            with open(json_path, 'r', encoding='utf-8') as f:
                try:
                    label_data = json.load(f)
                except json.JSONDecodeError:
                    print(f"Warning: Invalid JSON file {json_path}")
                    continue
                    
            base_name = img_path.stem
            
            # 原始图片和标注
            cv2.imwrite(os.path.join(image_output_dir, f"{base_name}_orig.png"), image)
            with open(os.path.join(label_output_dir, f"{base_name}_orig.json"), 'w', encoding='utf-8') as f:
                json.dump(label_data, f, indent=2)
            
            # 水平翻转
            flip_h = cv2.flip(image, 1)
            flip_h_label = label_data.copy()
            for shape in flip_h_label['shapes']:
                points = np.array(shape['points'])
                points[:, 0] = image.shape[1] - points[:, 0]
                shape['points'] = points.tolist()
            
            cv2.imwrite(os.path.join(image_output_dir, f"{base_name}_flip_h.png"), flip_h)
            with open(os.path.join(label_output_dir, f"{base_name}_flip_h.json"), 'w', encoding='utf-8') as f:
                json.dump(flip_h_label, f, indent=2)
            
            # 垂直翻转
            flip_v = cv2.flip(image, 0)
            flip_v_label = label_data.copy()
            for shape in flip_v_label['shapes']:
                points = np.array(shape['points'])
                points[:, 1] = image.shape[0] - points[:, 1]
                shape['points'] = points.tolist()
                
            cv2.imwrite(os.path.join(image_output_dir, f"{base_name}_flip_v.png"), flip_v)
            with open(os.path.join(label_output_dir, f"{base_name}_flip_v.json"), 'w', encoding='utf-8') as f:
                json.dump(flip_v_label, f, indent=2)
            
            # 旋转90度
            for angle in [90, 180, 270]:
                matrix = cv2.getRotationMatrix2D((image.shape[1]/2, image.shape[0]/2), angle, 1.0)
                rotated = cv2.warpAffine(image, matrix, (image.shape[1], image.shape[0]))
                
                rotated_label = label_data.copy()
                for shape in rotated_label['shapes']:
                    shape['points'] = rotate_points(shape['points'], angle, image.shape)
                
                cv2.imwrite(os.path.join(image_output_dir, f"{base_name}_rot_{angle}.png"), rotated)
                with open(os.path.join(label_output_dir, f"{base_name}_rot_{angle}.json"), 'w', encoding='utf-8') as f:
                    json.dump(rotated_label, f, indent=2)
                    
            print(f"Successfully processed {base_name}")
            
        except Exception as e:
            print(f"Error processing {img_path}: {str(e)}")
            continue

if __name__ == "__main__":
    input_directory = r"D:\DENTEX_dataset\training_data\training_data\quadrant-enumeration-disease\xrays"
    
    label_directory =r"D:\DENTEX_dataset\training_data\training_data\quadrant-enumeration-disease\xrays\train_quadrant_enumeration_disease.json" # 修改为正确的标签目录
    output_directory = r"D:\DENTEX_dataset\training_data\training_data\quadrant-enumeration-disease\augmented"

    
    print(f"Starting dataset augmentation...")
    print(f"Input directory: {input_directory}")
    print(f"Label directory: {label_directory}")
    print(f"Output directory: {output_directory}")
    
    augment_dataset(input_directory, label_directory, output_directory)
    print("Dataset augmentation completed!")