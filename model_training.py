import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, Input
from config import MODELS_PATH, FEATURES_PATH, PARAMS_MODEL, DATASET_PATH

def train_and_evaluate(dataset, mel_features, log):
    log.info('--- Huấn luyện mô hình ---')

    # Đường dẫn lưu mô hình
    json_file = os.path.join(MODELS_PATH, 'model.json')
    weights_file = os.path.join(MODELS_PATH, 'model_weights.weights.h5')  # Đảm bảo tên file đúng định dạng

    # Kiểm tra nếu mô hình đã tồn tại
    if os.path.exists(json_file) and os.path.exists(weights_file):
        log.info('Tải mô hình từ file JSON và weights...')
        with open(json_file, 'r') as f:
            model_json = f.read()
        model = tf.keras.models.model_from_json(model_json)
        model.load_weights(weights_file)
    else:
        log.info('Huấn luyện mô hình mới...')
        # Định nghĩa input_shape
        time_steps = 128  # Điều chỉnh nếu cần
        input_shape = (PARAMS_MODEL['n_mels'], time_steps, 1)

        # Tạo mô hình CNN
        model = Sequential([
            Input(shape=input_shape),
            Conv2D(32, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),
            Conv2D(64, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),
            Flatten(),
            Dense(128, activation='relu'),
            Dropout(0.5),
            Dense(50, activation='softmax')  # 50 lớp cho ESC-50
        ])

        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        # Đọc file esc50.csv
        meta_path = os.path.join(DATASET_PATH, 'meta', 'esc50.csv')
        meta_df = pd.read_csv(meta_path)

        # Huấn luyện qua các fold
        for fold in range(1, 3):  # Ví dụ: 2 fold
            log.info(f"Huấn luyện fold {fold}/2...")
            try:
                # Chia dữ liệu theo fold
                train_files = meta_df[meta_df['fold'] != fold]['filename'].tolist()
                val_files = meta_df[meta_df['fold'] == fold]['filename'].tolist()

                X_train, y_train = [], []
                X_val, y_val = [], []

                # Tải đặc trưng Mel cho tập train
                for filename in train_files:
                    file_path = os.path.join(DATASET_PATH, 'audio', filename)
                    label = meta_df[meta_df['filename'] == filename]['target'].values[0]
                    mel_data = mel_features.load_feature(file_path, FEATURES_PATH)
                    mel_data = mel_data[:, :time_steps] if mel_data.shape[1] >= time_steps else np.pad(mel_data, ((0, 0), (0, time_steps - mel_data.shape[1])), 'constant')
                    mel_data = np.expand_dims(mel_data, axis=-1)
                    X_train.append(mel_data)
                    y_train.append(label)

                # Tải đặc trưng Mel cho tập val
                for filename in val_files:
                    file_path = os.path.join(DATASET_PATH, 'audio', filename)
                    label = meta_df[meta_df['filename'] == filename]['target'].values[0]
                    mel_data = mel_features.load_feature(file_path, FEATURES_PATH)
                    mel_data = mel_data[:, :time_steps] if mel_data.shape[1] >= time_steps else np.pad(mel_data, ((0, 0), (0, time_steps - mel_data.shape[1])), 'constant')
                    mel_data = np.expand_dims(mel_data, axis=-1)
                    X_val.append(mel_data)
                    y_val.append(label)

                # Chuyển thành numpy array
                X_train = np.array(X_train)
                y_train = np.array(y_train)
                X_val = np.array(X_val)
                y_val = np.array(y_val)

                # Huấn luyện mô hình
                model.fit(
                    X_train, y_train,
                    validation_data=(X_val, y_val),
                    epochs=PARAMS_MODEL['epochs'],
                    batch_size=PARAMS_MODEL['batch_size'],
                    verbose=1
                )

                # Đánh giá
                metrics = model.evaluate(X_val, y_val, verbose=0)
                log.info(f"Fold {fold} - Accuracy: {metrics[1]:.4f}")
            except Exception as e:
                log.error(f"Lỗi khi huấn luyện fold {fold}: {e}")
                continue

        # Lưu mô hình
        model_json = model.to_json()
        with open(json_file, 'w') as f:
            f.write(model_json)
        model.save_weights(weights_file)
        log.info(f"Đã lưu mô hình vào {json_file} và {weights_file}")

    return model