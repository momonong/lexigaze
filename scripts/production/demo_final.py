import cv2
import mediapipe as mp
import numpy as np
import onnxruntime as ort
import time

# ================= ⚙️ 最終設定 =================
# 請確認這兩個檔案都在 models 資料夾下
ONNX_MODEL_PATH = 'models/litegaze_student_fp32.onnx' 
# 如果你想秀極致壓縮，也可以換成 INT8 版本:
# ONNX_MODEL_PATH = 'models/litegaze_student_int8.onnx' 
# ===============================================

class LiteGazeDemo:
    def __init__(self, model_path):
        # [cite_start]1. 初始化 MediaPipe (極速人臉偵測) [cite: 101]
        self.mp_face_detection = mp.solutions.face_detection
        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection=0, min_detection_confidence=0.5)
        
        # [cite_start]2. 初始化 ONNX Runtime (極速視線推理) [cite: 162]
        try:
            # 優先嘗試 GPU，不行就 CPU (ONNX 的 CPU 也有 AVX-512 加速)
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            self.session = ort.InferenceSession(model_path, providers=providers)
            print(f"🚀 ONNX 模型載入成功: {model_path}")
            print(f"⚡ 執行設備: {self.session.get_providers()[0]}")
        except Exception as e:
            print(f"❌ 模型載入失敗: {e}")
            raise e

        self.input_name = self.session.get_inputs()[0].name
        
        # 3. 預熱模型 (Warmup)
        dummy = np.random.randn(1, 3, 224, 224).astype(np.float32)
        self.session.run(None, {self.input_name: dummy})

    def preprocess(self, face_img):
        # 對齊訓練時的預處理 (Resize 224 -> Normalize)
        img = cv2.resize(face_img, (224, 224))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        # ImageNet Mean & Std
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img = (img - mean) / std
        img = img.transpose(2, 0, 1) # HWC -> CHW
        img = np.expand_dims(img, axis=0) # Add batch dim
        return img

    def softmax(self, x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=1, keepdims=True)

    def get_gaze(self, prediction):
        # 將 Logits 轉為角度 (與 PyTorch 邏輯一致)
        prob = self.softmax(prediction)
        idx = np.arange(90).astype(np.float32)
        gaze = np.sum(prob * idx, axis=1) * 4 - 180
        return gaze[0]

    def draw_gaze(self, image, pitch, yaw, bbox):
        x_min, y_min, x_max, y_max = bbox
        cx, cy = int((x_min + x_max) / 2), int((y_min + y_max) / 2)
        
        # 計算箭頭長度與方向
        face_w = x_max - x_min
        length = face_w / 2.0
        
        # 座標轉換 (Pitch/Yaw -> dx/dy)
        dx = -length * np.sin(yaw) * np.cos(pitch)
        dy = -length * np.sin(pitch)
        
        # 畫箭頭 (紅色)
        cv2.arrowedLine(image, (cx, cy), 
                       (int(cx + dx), int(cy + dy)), 
                       (0, 0, 255), 4, cv2.LINE_AA, tipLength=0.2)
        
        # 畫臉框 (綠色)
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

    def run(self):
        cap = cv2.VideoCapture(0)
        # 設定為 720p 以展示效能
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        prev_time = 0
        fps_avg = 0

        print("\n✅ Demo 開始! 按 'q' 離開")

        while cap.isOpened():
            success, frame = cap.read()
            if not success: break

            # 翻轉鏡頭 (像鏡子一樣)
            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape
            
            # FPS 計算
            curr_time = time.time()
            fps = 1 / (curr_time - prev_time)
            prev_time = curr_time
            fps_avg = 0.9 * fps_avg + 0.1 * fps # 平滑化顯示

            # 1. MediaPipe 偵測人臉
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_detection.process(frame_rgb)

            if results.detections:
                for detection in results.detections:
                    # 取得 BBox
                    bboxC = detection.location_data.relative_bounding_box
                    x_min = int(bboxC.xmin * w)
                    y_min = int(bboxC.ymin * h)
                    box_w = int(bboxC.width * w)
                    box_h = int(bboxC.height * h)
                    
                    # 邊界保護
                    x_min = max(0, x_min)
                    y_min = max(0, y_min)
                    x_max = min(w, x_min + box_w)
                    y_max = min(h, y_min + box_h)

                    face_img = frame[y_min:y_max, x_min:x_max]

                    if face_img.size > 0:
                        # 2. ONNX 推理
                        input_tensor = self.preprocess(face_img)
                        
                        # Run session
                        outputs = self.session.run(None, {self.input_name: input_tensor})
                        pitch_logits, yaw_logits = outputs[0], outputs[1]
                        
                        # 3. 後處理
                        pitch = self.get_gaze(pitch_logits)
                        yaw = self.get_gaze(yaw_logits)
                        
                        # 轉弧度
                        pitch_rad = pitch * np.pi / 180
                        yaw_rad = yaw * np.pi / 180

                        # 4. 繪圖
                        self.draw_gaze(frame, pitch_rad, yaw_rad, [x_min, y_min, x_max, y_max])
                        
                        # 顯示數值
                        text = f"P: {pitch:.1f} Y: {yaw:.1f}"
                        cv2.putText(frame, text, (x_min, y_min - 10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            # 顯示 FPS (綠色大字)
            cv2.putText(frame, f"FPS: {int(fps_avg)}", (20, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

            cv2.imshow('LiteGaze Final Demo', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    demo = LiteGazeDemo(ONNX_MODEL_PATH)
    demo.run()