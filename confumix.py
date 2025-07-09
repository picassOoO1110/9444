import os
import numpy as np
from pathlib import Path
import torch
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
from ultralytics import YOLO

def compute_confusion_matrix(model, val_dir, conf_thres=0.25):
    """
    计算混淆矩阵
    Args:
        model: YOLO模型
        val_dir: 验证集目录路径
        conf_thres: 置信度阈值
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 获取模型的类别数量和名称
    nc = model.names
    num_classes = len(nc)
    class_names = list(nc.values())
    
    # 初始化混淆矩阵
    confusion_matrix = np.zeros((num_classes, num_classes))
    
    # 在验证集上运行预测
    results = model.predict(val_dir, conf=conf_thres, device=device)
    
    for result in tqdm(results, desc="Processing predictions"):
        # 获取预测结果
        boxes = result.boxes
        cls = boxes.cls.cpu().numpy()
        conf = boxes.conf.cpu().numpy()
        
        # 获取真实标签
        labels = result.boxes.cls.cpu().numpy()  # 获取图片的真实标签
        
        # 更新混淆矩阵
        for pred_cls, pred_conf in zip(cls, conf):
            if pred_conf >= conf_thres:
                for true_cls in labels:
                    confusion_matrix[int(true_cls)][int(pred_cls)] += 1
    
    return confusion_matrix, class_names

def plot_confusion_matrix(confusion_matrix, class_names, output_path='confusion_matrix.png'):
    """
    绘制混淆矩阵
    """
    plt.figure(figsize=(12, 10))
    sns.heatmap(confusion_matrix, annot=True, fmt='.1f', 
                xticklabels=class_names, 
                yticklabels=class_names,
                cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def print_metrics(confusion_matrix, class_names):
    """
    打印每个类别的评估指标
    """
    print("\nPer-class metrics:")
    for i, class_name in enumerate(class_names):
        tp = confusion_matrix[i, i]
        fp = confusion_matrix[:, i].sum() - tp
        fn = confusion_matrix[i, :].sum() - tp
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        print(f"\n{class_name}:")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-Score: {f1:.4f}")

def main():
    # 配置参数
    model_path = '/data1/hbx/YOLO/yolov5s.pt'  # 训练好的模型路径
    val_dir = '/data1/hbx/YOLO/dataset/validation_data/quadrant_enumeration_disease/xrays'  # 验证集目录路径
    conf_thres = 0.25  # 置信度阈值
    
    print("Loading model...")
    model = YOLO(model_path)
    
    print("Computing confusion matrix...")
    confusion_matrix, class_names = compute_confusion_matrix(
        model, 
        val_dir,
        conf_thres=conf_thres
    )
    
    # 打印评估指标
    print_metrics(confusion_matrix, class_names)
    
    # 绘制混淆矩阵
    print("\nPlotting confusion matrix...")
    plot_confusion_matrix(confusion_matrix, class_names)
    print("Done! Confusion matrix has been saved as 'confusion_matrix.png'")

if __name__ == "__main__":
    main()