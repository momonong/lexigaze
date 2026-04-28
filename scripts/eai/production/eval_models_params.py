import torch
import torch.nn as nn
from torchvision import models

# 1. 定義模型架構 (必須跟訓練時一模一樣)
class L2CS_ResNet50(nn.Module):
    def __init__(self, num_bins=90):
        super(L2CS_ResNet50, self).__init__()
        # Teacher: ResNet50
        self.model = models.resnet50(weights=None)
        self.model.fc = nn.Linear(2048, num_bins * 2)

    def forward(self, x):
        x = self.model(x)
        return x[:, :90], x[:, 90:]

class L2CS_MobileNetV3(nn.Module):
    def __init__(self, num_bins=90):
        super(L2CS_MobileNetV3, self).__init__()
        # Student: MobileNetV3-Large
        self.backbone = models.mobilenet_v3_large(weights=None)
        in_features = self.backbone.classifier[3].in_features
        self.backbone.classifier[3] = nn.Linear(in_features, num_bins * 2)

    def forward(self, x):
        x = self.backbone(x)
        return x[:, :90], x[:, 90:]

def count_parameters(model):
    # 計算所有需要梯度的參數總數
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def main():
    print("🧮 正在計算模型參數量...\n")

    # 實例化模型
    teacher = L2CS_ResNet50()
    student = L2CS_MobileNetV3()

    # 計算
    teacher_params = count_parameters(teacher)
    student_params = count_parameters(student)

    # 輸出結果 (已修正格式化錯誤)
    print(f"{'模型':<25} | {'參數量 (Count)':<15} | {'百萬單位 (M)':<10}")
    print("-" * 60)
    # 🔥 修正點：將 ,:<15 改為 :<15, (先對齊寬度，再加逗號)
    print(f"{'Teacher (ResNet50)':<25} | {teacher_params:<15,} | {teacher_params/1e6:.2f} M")
    print(f"{'Student (MobileNetV3)':<25} | {student_params:<15,} | {student_params/1e6:.2f} M")
    
    print("-" * 60)
    print(f"📉 參數壓縮倍率: {teacher_params / student_params:.2f} 倍")

if __name__ == "__main__":
    main()