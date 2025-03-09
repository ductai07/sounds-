from dataset import initialize_dataset
from feature_extraction import extract_features, visualize_features
from model_training import train_and_evaluate
from visualization import predict_and_visualize
import os
from config import RESULTS_PATH

def main():
    # Tạo thư mục kết quả nếu chưa tồn tại
    os.makedirs(RESULTS_PATH, exist_ok=True)
    
    dataset, log = initialize_dataset()
    mel_features = extract_features(dataset, log)
    
    # Visualize một file mẫu
    sample_file = dataset.get_files()[0]
    visualize_features(sample_file, mel_features, log)
    
    # Huấn luyện mô hình
    model = train_and_evaluate(dataset, mel_features, log)
    
    # Dự đoán trên một file khác
    try:
        predict_file = dataset.get_files()[10]  # Sử dụng file thứ 11
        if os.path.exists(predict_file):
            log.info(f"Dự đoán với file: {os.path.basename(predict_file)}")
            predicted_class, class_name = predict_and_visualize(predict_file, dataset, model, mel_features, log)
            log.info(f"Kết quả dự đoán: {class_name}")
        else:
            log.error(f"File dự đoán không tồn tại: {predict_file}")
    except Exception as e:
        log.error(f"Lỗi khi dự đoán: {str(e)}")
    
    log.info('--- Kết luận ---')
    log.info('Đã hoàn thành phân tích âm thanh tích hợp với ESC-50')

if __name__ == "__main__":
    main()