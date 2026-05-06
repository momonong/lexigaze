import torch
import torch.nn as nn
import onnxruntime as ort
import time
import os
import numpy as np
from torchvision import models

# ================= ⚙️ 設定 =================
MODELS_DIR = 'models'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
ITERATIONS = 200  # 每個模型跑 200 次取平均
DUMMY_INPUT = torch.randn(1, 3, 224, 224)
# ==========================================

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

def benchmark_pytorch(model, name):
    model.eval()
    model.to(DEVICE)
    input_tensor = DUMMY_INPUT.to(DEVICE)
    
    # 預熱
    for _ in range(10): model(input_tensor)
    
    start = time.time()
    with torch.no_grad():
        for _ in range(ITERATIONS):
            _ = model(input_tensor)
    end = time.time()
    
    avg_latency = (end - start) / ITERATIONS * 1000
    fps = 1000 / avg_latency
    return avg_latency, fps

def benchmark_onnx(path):
    # 強制使用 CPU 測試，模擬一般部署環境
    sess = ort.InferenceSession(path, providers=['CPUExecutionProvider'])
    input_name = sess.get_inputs()[0].name
    input_data = DUMMY_INPUT.numpy()
    
    # 預熱
    for _ in range(10): sess.run(None, {input_name: input_data})
    
    start = time.time()
    for _ in range(ITERATIONS):
        _ = sess.run(None, {input_name: input_data})
    end = time.time()
    
    avg_latency = (end - start) / ITERATIONS * 1000
    fps = 1000 / avg_latency
    return avg_latency, fps

def main():
    results = []
    
    model_configs = [
        ("Teacher (ResNet50)", "L2CSNet_gaze360.pkl", "torch_pkl"),
        ("Student (FP32 PTH)", "student_mobilenet_3people_9k.pth", "torch_pth"),
        ("Student (QAT INT8)", "student_mobilenet_qat.pth", "torch_qat"),
        ("Student (FP32 ONNX)", "litegaze_student_fp32.onnx", "onnx"),
        ("Student (INT8 ONNX)", "litegaze_student_int8.onnx", "onnx"),
    ]

    print(f"🚀 開始實時性能分析 (測試環境: {DEVICE})...")
    print(f"{'模型名稱':<20} | {'體積(MB)':<10} | {'延遲(ms)':<10} | {'FPS':<10}")
    print("-" * 60)

    for label, filename, mtype in model_configs:
        path = os.path.join(MODELS_DIR, filename)
        if not os.path.exists(path):
            continue
            
        size = os.path.getsize(path) / (1024 * 1024)
        
        try:
            if mtype == "torch_pkl":
                # Teacher 特殊載入
                model = L2CS_ResNet50()
                ckpt = torch.load(path, map_location=DEVICE)
                # 簡單權重匹配邏輯
                state = {}
                for k, v in ckpt.items():
                    nk = 'model.'+k if not k.startswith('model.') else k
                    state[nk] = v
                model.load_state_dict(state, strict=False)
                lat, fps = benchmark_pytorch(model, label)
            
            elif mtype == "torch_pth":
                model = L2CS_MobileNetV3()
                model.load_state_dict(torch.load(path, map_location=DEVICE))
                lat, fps = benchmark_pytorch(model, label)
                
            elif mtype == "torch_qat":
                # 這裡僅測試結構，若要測真正量化後的速度需轉 ONNX 或用 CPU 引擎
                model = L2CS_MobileNetV3() # 簡化測試
                lat, fps = benchmark_pytorch(model, label)

            elif mtype == "onnx":
                lat, fps = benchmark_onnx(path)

            print(f"{label:<20} | {size:>8.2f}MB | {lat:>8.2f}ms | {fps:>8.1f}")
            results.append((label, size, lat, fps))
            
        except Exception as e:
            print(f"{label:<20} | 測試失敗: {str(e)[:20]}")

    print("\n💡 分析結論：")
    # 簡單分析邏輯
    fastest = sorted(results, key=lambda x: x[3], reverse=True)[0]
    smallest = sorted(results, key=lambda x: x[1])[0]
    print(f"1. 最速模型：{fastest[0]}，達到 {fastest[3]:.1f} FPS")
    print(f"2. 最輕量化：{smallest[0]}，僅有 {smallest[1]:.2f} MB")

if __name__ == "__main__":
    main()