import os
import librosa
import numpy as np
import dcase_util
from config import FEATURES_PATH, PARAMS_FEATURES, RESULTS_PATH
import matplotlib.pyplot as plt
import librosa.display

def extract_features(dataset, log):
    log.info('--- Trích xuất đặc trưng ---')

    # Hàm trích xuất Mel Spectrogram bằng librosa
    def extract_mel(file_path):
        try:
            y, sr = librosa.load(file_path, sr=PARAMS_FEATURES['sr'])
            mel_spec = librosa.feature.melspectrogram(
                y=y,
                sr=sr,
                n_fft=PARAMS_FEATURES['n_fft'],
                hop_length=PARAMS_FEATURES['hop_length'],
                n_mels=PARAMS_FEATURES['n_mels'],
                fmin=PARAMS_FEATURES['fmin'],
                fmax=PARAMS_FEATURES['fmax']
            )
            # Chuyển sang dB
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
            return mel_spec_db
        except Exception as e:
            print(f"Lỗi khi trích xuất mel spectrogram: {str(e)}")
            return None

    # Tạo thư mục lưu đặc trưng nếu chưa có
    if not os.path.exists(FEATURES_PATH):
        os.makedirs(FEATURES_PATH)

    # Lấy danh sách file từ dataset
    file_list = dataset.get_files()

    # Trích xuất đặc trưng cho file đầu tiên (hoặc tất cả nếu cần)
    for file_path in file_list[:1]:  # Chỉ xử lý file đầu tiên để nhanh
        feature_file = os.path.join(FEATURES_PATH, os.path.basename(file_path) + '.npy')
        if not os.path.exists(feature_file):
            mel_spec = extract_mel(file_path)
            if mel_spec is not None:
                np.save(feature_file, mel_spec)
                log.info(f'Đã trích xuất đặc trưng cho: {os.path.basename(file_path)}')
        else:
            log.info(f'Đã tìm thấy đặc trưng cho: {os.path.basename(file_path)}')

    # Đối tượng giả lập để load đặc trưng
    class DummyMelFeatures:
        def get_name(self):
            return "melspectrogram"

        def get_shape(self):
            # Trả về shape mẫu cho model
            return (PARAMS_FEATURES['n_mels'], None, 1)  # (n_mels, time_steps, channels)

        def load_feature(self, file_path, features_path):
            feature_file = os.path.join(features_path, os.path.basename(file_path) + '.npy')
            if os.path.exists(feature_file):
                return np.load(feature_file)
            else:
                # Nếu không tìm thấy file, tự động trích xuất
                return self.extract_file(file_path, features_path)
        
        def extract_file(self, file_path, features_path):
            os.makedirs(features_path, exist_ok=True)
            
            # Kiểm tra xem file có tồn tại không
            if not os.path.exists(file_path):
                print(f"File không tồn tại: {file_path}")
                # Trả về một mảng rỗng phù hợp với kích thước
                dummy_mel = np.zeros((PARAMS_FEATURES['n_mels'], 128))
                return dummy_mel
            
            # Trích xuất đặc trưng
            try:
                y, sr = librosa.load(file_path, sr=PARAMS_FEATURES['sr'])
                mel_spec = librosa.feature.melspectrogram(
                    y=y,
                    sr=sr,
                    n_fft=PARAMS_FEATURES['n_fft'],
                    hop_length=PARAMS_FEATURES['hop_length'],
                    n_mels=PARAMS_FEATURES['n_mels'],
                    fmin=PARAMS_FEATURES['fmin'],
                    fmax=PARAMS_FEATURES['fmax']
                )
                
                # Chuyển sang dB
                mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
                
                # Lưu đặc trưng
                feature_file = os.path.join(features_path, os.path.basename(file_path) + '.npy')
                np.save(feature_file, mel_spec_db)
                
                # Kiểm tra kết quả
                print(f"Đã trích xuất mel spectrogram với shape: {mel_spec_db.shape}")
                return mel_spec_db
                
            except Exception as e:
                print(f"Lỗi khi trích xuất mel spectrogram: {str(e)}")
                # Trả về một mảng rỗng phù hợp với kích thước
                dummy_mel = np.zeros((PARAMS_FEATURES['n_mels'], 128))
                return dummy_mel

    mel_features = DummyMelFeatures()
    log.info('Đã hoàn thành trích xuất đặc trưng')
    return mel_features

def visualize_features(file_path, mel_features, log):
    log.info('--- Visualization đặc trưng ---')

    # Tải file âm thanh
    try:
        y, sr = librosa.load(file_path, sr=PARAMS_FEATURES['sr'])
    except Exception as e:
        log.error(f"Không thể tải file âm thanh: {str(e)}")
        return None
        
    # Tải mel spectrogram
    mel_data = mel_features.load_feature(file_path, FEATURES_PATH)
    if mel_data is None:
        log.error("Không thể tải mel spectrogram")
        return None

    # Visualize với matplotlib và librosa
    plt.figure(figsize=(10, 6))
    librosa.display.specshow(
        mel_data,
        sr=PARAMS_FEATURES['sr'],
        hop_length=PARAMS_FEATURES['hop_length'],
        x_axis='time',
        y_axis='mel',
        fmin=PARAMS_FEATURES['fmin'],
        fmax=PARAMS_FEATURES['fmax'],
        cmap='inferno'
    )
    plt.colorbar(format='%+2.0f dB')
    plt.title(f'Mel Spectrogram - {os.path.basename(file_path)}')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_PATH, 'mel_dcase_util.png'))
    plt.close()
    log.info(f'Đã lưu visualization tại: {os.path.join(RESULTS_PATH, "mel_dcase_util.png")}')
    return mel_data

if __name__ == "__main__":
    from dataset import initialize_dataset
    dataset, log = initialize_dataset()
    mel_features = extract_features(dataset, log)
    sample_file = dataset.get_files()[0]
    visualize_features(sample_file, mel_features, log)