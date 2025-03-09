import dcase_util
from dcase_models.data.datasets import ESC50
from config import DATASET_PATH, LOG_FOLDER
import logging
import os
import pandas as pd

# Thiết lập logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(LOG_FOLDER, 'analysis.log'), encoding='utf-8'),
        logging.StreamHandler()
    ]
)

def initialize_dataset():
    # Khởi tạo logger
    log = logging.getLogger('ESC50_Analysis')
    log.info('Phân tích âm thanh tích hợp với ESC-50')
    log.info('--- Khởi tạo ---')

    # Khởi tạo dataset
    log.info('Kiểm tra ESC-50 dataset...')
    dataset = ESC50(DATASET_PATH)
    if not os.path.exists(os.path.join(DATASET_PATH, 'audio')):
        log.warning('Dataset chưa có, tải thủ công từ https://github.com/karolpiczak/ESC-50')
        raise FileNotFoundError("ESC-50 dataset không tồn tại. Vui lòng tải và giải nén trước.")
    else:
        log.info('Dataset đã sẵn sàng.')

    # Đọc metadata từ file esc50.csv
    meta_path = os.path.join(DATASET_PATH, 'meta', 'esc50.csv')
    if not os.path.exists(meta_path):
        log.error("File meta/esc50.csv không tồn tại.")
        raise FileNotFoundError("File meta/esc50.csv không tồn tại.")
    
    meta_df = pd.read_csv(meta_path)
    total_samples = len(meta_df)
    classes = sorted(meta_df['category'].unique())

    # Tạo danh sách file âm thanh
    audio_dir = os.path.join(DATASET_PATH, 'audio')
    file_list = [os.path.join(audio_dir, filename) for filename in meta_df['filename']]

    # Gán danh sách file vào dataset
    dataset.file_list = file_list

    # Gán phương thức get_files không cần đối số
    dataset.get_files = lambda: file_list  # Chỉ trả về file_list, không cần tham số

    log.info(f'Tổng số mẫu: {total_samples}')
    log.info(f'Số lớp: {len(classes)}')
    log.info(f'Các lớp: {classes}')

    return dataset, log

if __name__ == "__main__":
    dataset, log = initialize_dataset()