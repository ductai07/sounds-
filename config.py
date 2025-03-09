import os

# Định nghĩa đường dẫn
DATASET_PATH = 'ESC-50'
WORKSPACE = 'workspace'
FEATURES_PATH = os.path.join(WORKSPACE, 'features')
MODELS_PATH = os.path.join(WORKSPACE, 'models')
LOG_FOLDER = os.path.join(WORKSPACE, 'log')
RESULTS_PATH = os.path.join(WORKSPACE, 'results')
os.makedirs(MODELS_PATH, exist_ok=True)
# Tham số trích xuất đặc trưng
PARAMS_FEATURES = {
    'sr': 44100,
    'n_fft': 2048,
    'hop_length': 1024,
    'n_mels': 64,
    'fmin': 0,  # Đảm bảo dùng fmin thay vì f_min
    'fmax': 22050  # Đảm bảo dùng fmax thay vì f_max
}

# Tham số mô hình
PARAMS_MODEL = {
    'seed': 42,
    'batch_size': 32,  # Giảm batch_size để chạy nhanh
    'epochs': 23,    # Giảm số epoch để chạy nhanh
    'n_mels': 64,   # Số Mel bands, phải khớp với PARAMS_FEATURES['n_mels']
    'sequence_hop': 0.5,
    'sequence_length': 1.0,
    'preprocessing': 'normalization',
    'augmentation': None,
    'mixup': None
}

# Tạo các thư mục cần thiết
for directory in [WORKSPACE, FEATURES_PATH, MODELS_PATH, LOG_FOLDER, RESULTS_PATH]:
    os.makedirs(directory, exist_ok=True)