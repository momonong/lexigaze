# LiteGaze - 輕量化視線追蹤系統

一個基於知識蒸餾的高效能視線追蹤解決方案，將 ResNet50 教師模型的知識轉移到 MobileNetV3 學生模型，實現實時視線追蹤。

## 🎯 專案目標

解決傳統視線追蹤模型過於龐大、無法在邊緣設備實時運行的問題，通過以下技術實現輕量化：

- **知識蒸餾 (Knowledge Distillation)**: 使用 L2CS ResNet50 作為教師模型指導 MobileNetV3 學生模型學習
- **量化優化 (Quantization)**: 支援 QAT (量化感知訓練) 和 ONNX INT8 量化
- **自製數據增強**: 收集個人化數據提升模型泛化能力
- **多格式部署**: 支援 PyTorch、ONNX (FP32/INT8)、TensorFlow Lite 等格式

## 🚀 核心技術方案

### 1. 教師-學生架構
- **教師模型**: L2CS ResNet50 (預訓練於 Gaze360 數據集)
- **學生模型**: MobileNetV3-Large (ImageNet 預訓練 + 知識蒸餾)
- **蒸餾策略**: 溫度縮放 (T=5.0) + KL 散度損失

### 2. 數據處理流程
- 使用 MediaPipe 進行實時人臉檢測
- 自製數據收集與標註系統
- 強化數據增強防止過擬合 (ColorJitter, GaussianBlur, RandomGrayscale)

### 3. 模型優化技術
- **QAT (Quantization Aware Training)**: 訓練時模擬量化噪聲
- **ONNX 轉換**: 支援跨平台部署
- **靜態量化**: INT8 量化實現 4x 模型壓縮

## 📊 性能表現

| 模型版本 | 大小 | 延遲 (CPU) | FPS | 精度保持 |
|---------|------|-----------|-----|---------|
| Teacher (ResNet50) | ~100MB | ~50ms | ~20 | 基準 |
| Student (FP32) | ~21MB | ~15ms | ~67 | 95%+ |
| Student (INT8 ONNX) | ~5.3MB | ~8ms | ~125 | 90%+ |

## 🛠️ 安裝與使用

### 環境設置
```bash
# 安裝依賴 (生產環境)
pip install -r requirements.txt

# 開發環境 (如需訓練)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install onnx onnxruntime-gpu
```

### 快速開始
```bash
# 運行最終 Demo (需要攝像頭)
python scripts/production/demo_final.py

# 評估所有模型性能
python scripts/production/eval_models.py
```

## 📁 專案結構

```
├── models/                     # 訓練好的模型檔案
│   ├── L2CSNet_gaze360.pkl    # 教師模型 (ResNet50)
│   ├── student_mobilenet_3people_9k.pth  # 學生模型 (FP32)
│   ├── litegaze_student_fp32.onnx        # ONNX FP32 版本
│   └── litegaze_student_int8.onnx        # ONNX INT8 版本
├── scripts/production/         # 🔥 最終版本程式碼
│   ├── demo_final.py          # 實時視線追蹤 Demo
│   ├── train_with_selfmade.py # 知識蒸餾訓練
│   ├── export_onnx.py         # PyTorch → ONNX 轉換
│   ├── quantize_onnx.py       # ONNX 靜態量化
│   ├── train_qat.py           # 量化感知訓練
│   └── eval_models.py         # 模型性能評估
├── data/
│   ├── selfmade_combined/     # 自製訓練數據
│   └── teacher_224.h5         # 教師模型數據
└── requirements.txt           # 生產環境依賴
```

## 🔬 技術實現細節

### 知識蒸餾流程
1. **教師模型載入**: 載入預訓練的 L2CS ResNet50
2. **學生模型初始化**: MobileNetV3 + ImageNet 預訓練權重
3. **蒸餾訓練**: 使用溫度縮放軟化 logits，KL 散度計算蒸餾損失
4. **數據增強**: 防止過擬合到特定環境/人臉特徵

### 量化優化策略
- **QAT**: 訓練時插入 FakeQuant 節點模擬量化誤差
- **靜態量化**: 使用校準數據集進行 INT8 量化
- **算子融合**: Conv+BN+ReLU 融合提升推理效率

### 部署優化
- **ONNX Runtime**: 跨平台推理引擎
- **MediaPipe**: 高效人臉檢測
- **多線程數據載入**: 提升訓練效率

## 🎮 使用說明

### 訓練自己的模型
```bash
# 1. 收集個人化數據
python scripts/production/collect_data_official.py

# 2. 知識蒸餾訓練
python scripts/production/train_with_selfmade.py

# 3. 轉換為 ONNX
python scripts/production/export_onnx.py

# 4. INT8 量化
python scripts/production/quantize_onnx.py
```

### 模型評估
```bash
# 性能基準測試
python scripts/production/eval_models.py

# CPU 專用評估
python scripts/production/eval_models_cpu.py
```

## 🔧 技術棧

- **深度學習**: PyTorch, TensorFlow
- **模型優化**: ONNX, ONNX Runtime, TensorRT
- **計算機視覺**: OpenCV, MediaPipe
- **數據處理**: NumPy, PIL
- **量化技術**: PyTorch Quantization, ONNX Quantization

## 📈 創新點

1. **端到端蒸餾流程**: 從數據收集到模型部署的完整解決方案
2. **多格式支援**: 同時支援 PyTorch、ONNX、TFLite 部署
3. **自適應量化**: QAT + 靜態量化雙重優化策略
4. **實時性能**: 在普通 CPU 上達到 100+ FPS
5. **個人化數據**: 支援自製數據提升特定場景精度

---

## 🚀 快速 Demo

運行實時視線追蹤：
```bash
python scripts/production/demo_final.py
```

按 'q' 退出，享受流暢的視線追蹤體驗！