import torch
import torch.nn as nn
import onnxruntime as ort
import time
import os
import numpy as np
from torchvision import models

# ================= ⚙️ CPU 測試設定 =================
DEVICE = torch.device('cpu')  # 🔥 強制使用 CPU
ITERATIONS = 100               # CPU 較慢，跑 100 次取平均
MODELS_DIR = 'models'
DUMMY_INPUT = torch.randn(1, 3, 224, 224)
# =================================================

class L2CS_ResNet50(nn.Module):
    def __init__(self, num_bins=90):
        super().__init__()
        self.model = models.resnet50(weights=None)
        self.model.fc = nn.Linear(2048, num_bins * 2)
    def forward(self, x):
        x = self.model(x)
        return x[:, :90], x[:, 90:]

class L2CS_MobileNetV3(nn.Module):
    def __init__(self, num_bins=90):
        super().__init__()
        self.backbone = models.mobilenet_v3_large(weights=None)
        in_features = self.backbone.classifier[3].in_features
        self.backbone.classifier[3] = nn.Linear(in_features, num_bins * 2)
    def forward(self, x):
        x = self.backbone(x)
        return x[:, :90], x[:, 90:]

def benchmark_pytorch_cpu(model, name):
    model.eval()
    model.to(DEVICE)
    input_tensor = DUMMY_INPUT.to(DEVICE)
    for _ in range(5): model(input_tensor) # 熱身
    
    start = time.time()
    with torch.no_grad():
        for _ in range(ITERATIONS):
            _ = model(input_tensor)
    end = time.time()
    
    avg_latency = (end - start) / ITERATIONS * 1000
    fps = 1000 / avg_latency
    return avg_latency, fps

def benchmark_onnx_cpu(path):
    # 強制使用 CPUExecutionProvider
    sess = ort.InferenceSession(path, providers=['CPUExecutionProvider'])
    input_name = sess.get_inputs()[0].name
    input_data = DUMMY_INPUT.numpy()
    for _ in range(5): sess.run(None, {input_name: input_data}) # 熱身
    
    start = time.time()
    for _ in range(ITERATIONS):
        _ = sess.run(None, {input_name: input_data})
    end = time.time()
    
    avg_latency = (end - start) / ITERATIONS * 1000
    fps = 1000 / avg_latency
    return avg_latency, fps

def main():
    model_configs = [
        ("Teacher (ResNet50)", "L2CSNet_gaze360.pkl", "torch_pkl"),
        ("Student (FP32 PTH)", "student_mobilenet_3people_9k.pth", "torch_pth"),
        ("Student (QAT INT8)", "student_mobilenet_qat.pth", "torch_qat"),
        ("Student (FP32 ONNX)", "litegaze_student_fp32.onnx", "onnx"),
        ("Student (INT8 ONNX)", "litegaze_student_int8.onnx", "onnx"),
    ]

    print(f"🚀 開始實時 CPU 性能分析...")
    print(f"{'模型名稱':<20} | {'體積(MB)':<10} | {'延遲(ms)':<10} | {'FPS':<10}")
    print("-" * 60)

    for label, filename, mtype in model_configs:
        path = os.path.join(MODELS_DIR, filename)
        if not os.path.exists(path): continue
        size = os.path.getsize(path) / (1024 * 1024)
        
        try:
            if mtype == "torch_pkl":
                model = L2CS_ResNet50()
                model.load_state_dict(torch.load(path, map_location=DEVICE), strict=False)
                lat, fps = benchmark_pytorch_cpu(model, label)
            elif mtype in ["torch_pth", "torch_qat"]:
                model = L2CS_MobileNetV3()
                model.load_state_dict(torch.load(path, map_location=DEVICE), strict=False)
                lat, fps = benchmark_pytorch_cpu(model, label)
            elif mtype == "onnx":
                lat, fps = benchmark_onnx_cpu(path)
            print(f"{label:<20} | {size:>8.2f}MB | {lat:>8.2f}ms | {fps:>8.1f}")
        except Exception as e:
            print(f"{label:<20} | 測試失敗")

if __name__ == "__main__":
    main()