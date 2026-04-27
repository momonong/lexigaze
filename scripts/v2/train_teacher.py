import tensorflow as tf
import numpy as np
import os
import h5py
from .models import build_teacher_v3

# === ⚙️ A100 超級教師設定 ===
DATA_PATH = "data/teacher_224.h5"  # 這是剛剛正在做的檔案
BATCH_SIZE = 64                    # A100 記憶體大，可以開大一點 (64/128)
EPOCHS = 20                        # 大模型收斂快
LEARNING_RATE = 1e-4               # 微調建議用小一點的 LR
NUM_BINS = 90
BIN_MIN, BIN_MAX = -1.57, 1.57     # Radians

def hdf5_generator(path, batch_size, is_train=True):
    """
    HDF5 串流生成器
    就像迴轉壽司師傅，一次只拿 batch_size 盤出來
    """
    while True:
        with h5py.File(path, 'r') as hf:
            images_dset = hf['images']
            labels_dset = hf['labels']
            total_len = images_dset.shape[0]
            
            # 生成隨機索引 (如果是訓練模式)
            indices = np.arange(total_len)
            if is_train:
                np.random.shuffle(indices)
            
            # 批次讀取
            for i in range(0, total_len, batch_size):
                batch_idx = indices[i : i + batch_size]
                # 這裡要 sorted 才能讀 h5，讀完再 shuffle 回去有點麻煩
                # 為了效能，我們這裡做一個妥協：直接讀連續區塊，然後再 shuffle batch 內部
                # (對於超大數據集，通常會先隨機打散儲存，或者用更複雜的 Shuffler，這裡先求穩)
                
                # 修正：h5py 支援 list indexing 嗎？不一定。
                # 最穩的方法是 sort index
                sorted_idx = np.sort(batch_idx)
                
                batch_imgs = images_dset[sorted_idx]
                batch_lbls = labels_dset[sorted_idx]
                
                # 因為我們剛剛存的是 uint8 (0-255)，這裡要轉成 float32 並 Normalize
                batch_imgs = batch_imgs.astype(np.float32) / 255.0
                
                # 這裡需要把標籤轉成 L2CS 格式
                # Pitch/Yaw processing
                p_idx_list = []
                y_idx_list = []
                
                for label in batch_lbls:
                    pitch, yaw = label[0], label[1]
                    p_norm = (pitch - BIN_MIN) / (BIN_MAX - BIN_MIN)
                    y_norm = (yaw - BIN_MIN) / (BIN_MAX - BIN_MIN)
                    p_idx = int(np.clip(p_norm * (NUM_BINS - 1), 0, NUM_BINS - 1))
                    y_idx = int(np.clip(y_norm * (NUM_BINS - 1), 0, NUM_BINS - 1))
                    p_idx_list.append(p_idx)
                    y_idx_list.append(y_idx)
                
                yield batch_imgs, {
                    'gaze_out': batch_lbls,
                    'pitch_logits': np.array(p_idx_list),
                    'yaw_logits': np.array(y_idx_list)
                }

def create_dataset(h5_path, batch_size, is_train=True):
    # 取得資料總長度
    with h5py.File(h5_path, 'r') as hf:
        total_samples = hf['images'].shape[0]
        
    output_signature = (
        tf.TensorSpec(shape=(None, 224, 224, 3), dtype=tf.float32),
        {
            'gaze_out': tf.TensorSpec(shape=(None, 2), dtype=tf.float32),
            'pitch_logits': tf.TensorSpec(shape=(None,), dtype=tf.int32),
            'yaw_logits': tf.TensorSpec(shape=(None,), dtype=tf.int32),
        }
    )
    
    ds = tf.data.Dataset.from_generator(
        lambda: hdf5_generator(h5_path, batch_size, is_train),
        output_signature=output_signature
    )
    
    # 優化效能
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds, total_samples

def main():
    print("🚀 Initializing God Teacher Training (HDF5 Streaming Mode)...")
    
    if not os.path.exists(DATA_PATH):
        print(f"❌ Error: 找不到 {DATA_PATH}")
        return

    # 1. 建立 Dataset
    # 這裡簡化：沒有分 Train/Val，直接全部 Train (因為主要是為了蒸餾，老師看過也沒關係)
    # 或是您可以切分兩個 h5 檔。這邊先做簡單版：90% 用於訓練
    train_ds, total_samples = create_dataset(DATA_PATH, BATCH_SIZE, is_train=True)
    
    steps_per_epoch = total_samples // BATCH_SIZE
    print(f"📊 Total samples: {total_samples}, Steps per epoch: {steps_per_epoch}")

    # 2. 建立模型
    # 必須要用 MirroredStrategy 才能榨乾 A100 的多卡效能 (如果是單卡也沒關係)
    strategy = tf.distribute.MirroredStrategy()
    print(f"✅ Number of devices: {strategy.num_replicas_in_sync}")

    with strategy.scope():
        model = build_teacher_v3(input_shape=(224, 224, 3))
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE, clipnorm=1.0),
            loss={
                'gaze_out': 'mse', 
                'pitch_logits': 'sparse_categorical_crossentropy', 
                'yaw_logits': 'sparse_categorical_crossentropy'
            },
            loss_weights={'gaze_out': 1.0, 'pitch_logits': 1.0, 'yaw_logits': 1.0},
            metrics={'gaze_out': 'mae'}
        )
    
    model.summary()

    # 3. 訓練
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint("models/teacher_v3_best.h5", save_best_only=True, monitor='loss', mode='min'),
        # 如果沒有 Validation Set，我們就 Monitor 'loss'
        tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=2, mode='min')
    ]

    print("🔥 Start Training on High-Performance GPU...")
    model.fit(
        train_ds,
        epochs=EPOCHS,
        steps_per_epoch=steps_per_epoch,
        callbacks=callbacks
    )
    
    print("🎉 God Teacher Trained & Saved!")

if __name__ == "__main__":
    main()