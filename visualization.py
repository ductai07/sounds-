import os
import librosa
import numpy as np
import dcase_util
from config import FEATURES_PATH, PARAMS_FEATURES, RESULTS_PATH
import matplotlib.pyplot as plt
import librosa.display

def predict_and_visualize(file_path, dataset, model, mel_features, log):
    log.info('--- Demo dự đoán ---')
    
    # Kiểm tra file tồn tại
    if not os.path.exists(file_path):
        log.error(f"File không tồn tại: {file_path}")
        return None, "Unknown"
    
    # Trích xuất đặc trưng
    try:
        # Trích xuất mel spectrogram
        mel_data = mel_features.extract_file(file_path, FEATURES_PATH)
        log.info(f'Đã trích xuất đặc trưng cho: {os.path.basename(file_path)}')
        
        # Kiểm tra mel_data không phải None
        if mel_data is None:
            log.error("Trích xuất mel spectrogram thất bại, trả về None")
            return None, "Unknown"
        
        # Kiểm tra shape
        log.info(f"Shape của mel_data trước khi xử lý: {mel_data.shape}")
        
        # QUAN TRỌNG: Chuẩn hóa kích thước đầu vào cho model
        # Model mong đợi shape (None, 64, 128, 1)
        time_steps = 128  # Số time steps yêu cầu
        
        # Đảm bảo mel_data có chiều cao 64 (n_mels)
        if mel_data.shape[0] != PARAMS_FEATURES['n_mels']:
            log.warning(f"Điều chỉnh số kênh mel từ {mel_data.shape[0]} thành {PARAMS_FEATURES['n_mels']}")
            mel_data = np.resize(mel_data, (PARAMS_FEATURES['n_mels'], mel_data.shape[1]))
        
        # Điều chỉnh chiều rộng (time steps)
        if mel_data.shape[1] > time_steps:
            # Cắt bớt nếu dài quá
            mel_data = mel_data[:, :time_steps]
        else:
            # Pad nếu ngắn quá
            mel_data = np.pad(mel_data, ((0, 0), (0, time_steps - mel_data.shape[1])), 'constant')
        
        # Thêm batch dimension và channel dimension
        mel_data = np.expand_dims(mel_data, axis=0)  # Batch dimension
        mel_data = np.expand_dims(mel_data, axis=-1)  # Channel dimension
        
        log.info(f"Kích thước input cho model: {mel_data.shape}")
        
        # Dự đoán - với xử lý lỗi
        if model is None:
            log.error("Model là None, không thể dự đoán")
            return None, "Unknown"
        
        try:
            # Kiểm tra xem model là KerasModelContainer hay Sequential
            if hasattr(model, 'predict') and callable(model.predict):
                # Đây là mô hình Keras trực tiếp (Sequential)
                prediction = model.predict(mel_data)
                log.info(f"Đã dự đoán trực tiếp với model.predict()")
            elif hasattr(model, 'model') and hasattr(model.model, 'predict'):
                # Đây là KerasModelContainer
                prediction = model.model.predict(mel_data)
                log.info(f"Đã dự đoán với model.model.predict()")
            else:
                log.error("Không thể tìm thấy phương thức predict phù hợp")
                return None, "Unknown"
                
            log.info(f"Prediction shape: {prediction.shape if hasattr(prediction, 'shape') else 'unknown'}")
            
        except Exception as e:
            log.error(f"Lỗi khi dự đoán với model.predict: {str(e)}")
            # Thử phương pháp dự đoán khác
            try:
                # Đây là fallback, thử gọi model như một hàm
                raw_output = model(mel_data)
                if hasattr(raw_output, 'numpy'):
                    prediction = np.argmax(raw_output.numpy(), axis=1)
                else:
                    prediction = np.argmax(raw_output, axis=1)
                log.info(f"Đã dự đoán với model call: {prediction}")
            except Exception as inner_e:
                log.error(f"Lỗi khi dự đoán với model call: {str(inner_e)}")
                return None, "Unknown"
        
        # Xử lý kết quả dự đoán
        if isinstance(prediction, np.ndarray):
            if prediction.ndim > 1 and prediction.shape[1] > 1:
                # Đầu ra là probabilities
                predicted_class = np.argmax(prediction[0])
            else:
                # Đầu ra là class index
                predicted_class = prediction[0] if len(prediction) > 0 else 0
        else:
            predicted_class = prediction if prediction is not None else 0
        
        # Lấy danh sách classes từ dataset
        class_names = None
        if hasattr(dataset, 'get_classes') and callable(dataset.get_classes):
            class_names = dataset.get_classes()
        elif hasattr(dataset, 'classes'):
            class_names = dataset.classes
        elif hasattr(dataset, 'class_names'):
            class_names = dataset.class_names
        else:
            # Tạo danh sách lớp mặc định nếu không tìm thấy
            log.warning("Không tìm thấy danh sách lớp từ dataset, sử dụng danh sách mặc định")
            class_names = [f"Class {i}" for i in range(50)]  # ESC-50 có 50 lớp
            
        # Kiểm tra giá trị predicted_class hợp lệ
        num_classes = len(class_names)
        if predicted_class >= num_classes:
            log.warning(f"Lớp dự đoán ({predicted_class}) vượt quá số lớp ({num_classes})")
            predicted_class = 0  # Mặc định về lớp đầu tiên
            
        class_name = class_names[predicted_class]
        log.info(f"Lớp dự đoán: {class_name} (index: {predicted_class})")
        
        # Visualize kết quả
        plt.figure(figsize=(10, 6))
        
        # Hiển thị mel spectrogram
        plt.subplot(2, 1, 1)
        librosa.display.specshow(
            mel_data[0, :, :, 0],  # Lấy ra mel spectrogram từ tensor 4D
            sr=PARAMS_FEATURES['sr'],
            hop_length=PARAMS_FEATURES['hop_length'],
            x_axis='time',
            y_axis='mel',
            fmin=PARAMS_FEATURES['fmin'],
            fmax=PARAMS_FEATURES['fmax']
        )
        plt.colorbar(format='%+2.0f dB')
        plt.title(f'Mel Spectrogram - Predicted: {class_name}')
        
        # Hiển thị waveform
        plt.subplot(2, 1, 2)
        y, sr = librosa.load(file_path, sr=PARAMS_FEATURES['sr'])
        plt.plot(np.linspace(0, len(y)/sr, len(y)), y)
        plt.title('Waveform')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        
        plt.tight_layout()
        prediction_img_path = os.path.join(RESULTS_PATH, 'prediction.png')
        plt.savefig(prediction_img_path)
        plt.close()
        
        log.info(f"Đã lưu kết quả dự đoán tại: {prediction_img_path}")
        
        return predicted_class, class_name
    
    except Exception as e:
        log.error(f"Lỗi khi dự đoán: {str(e)}")
        import traceback
        log.error(traceback.format_exc())
        return None, "Unknown"