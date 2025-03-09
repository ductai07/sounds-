# Dự án Phân Tích Âm Thanh ESC-50

Dự án này được xây dựng nhằm thực hiện phân tích âm thanh trên tập dữ liệu ESC-50 sử dụng các kỹ thuật deep learning. Dự án tự động trích xuất đặc trưng âm thanh (Mel Spectrogram), huấn luyện mô hình CNN (ví dụ: SB-CNN) để phân loại 50 loại âm thanh và cung cấp các công cụ visualization cho quá trình trích xuất đặc trưng và dự đoán.

## Mục Lục

- [Giới thiệu](#giới-thiệu)
- [Tính năng](#tính-năng)
- [Cài đặt](#cài-đặt)
- [Cấu hình](#cấu-hình)
- [Cách sử dụng](#cách-sử-dụng)
- [Cấu trúc dự án](#cấu-trúc-dự-án)
- [Đóng góp](#đóng-góp)
- [Giấy phép](#giấy-phép)

## Giới thiệu

ESC-50 là tập dữ liệu bao gồm 2000 đoạn âm thanh môi trường được dán nhãn vào 50 lớp khác nhau. Dự án này trình bày cách:

- Trích xuất đặc trưng Mel Spectrogram từ âm thanh bằng thư viện Librosa.
- Huấn luyện mô hình CNN (SB-CNN) để phân loại âm thanh với thư viện [dcase-models](https://github.com/karolpiczak/dcase-models).
- Hiển thị trực quan kết quả dự đoán và đặc trưng âm thanh.
- Đánh giá và so sánh hiệu năng của các phương pháp trích xuất đặc trưng và mô hình phân loại.

## Tính năng

- **Trích xuất đặc trưng:** Tự động chuyển đổi đoạn âm thanh thành Mel Spectrogram.
- **Huấn luyện mô hình:** Sử dụng kiến trúc SB-CNN để phân loại âm thanh với Keras.
- **Dự đoán & Visualization:** Cho phép dự đoán trên file âm thanh mới và hiển thị kết quả thông qua biểu đồ Mel Spectrogram và waveform.
- **Cấu hình dễ dàng:** Tập trung các tham số cấu hình (đường dẫn, tham số xử lý âm thanh, tham số huấn luyện) trong một file cấu hình duy nhất.

## Cài đặt

1. **Clone Repository:**

   Mở Terminal (CMD hoặc PowerShell) và chạy lệnh sau:

   ```bash
   git clone https://github.com/yourusername/your-repo.git
   cd your-repo
   ```

2. **Cài đặt các thư viện cần thiết:**

   Chạy các lệnh sau trong terminal:

   ```bash
   pip install librosa
   pip install kapre==0.3.7
   pip install tensorflow==2.6.0
   pip install h5py==3.1.0
   pip install dcase-models
   pip install sed_eval
   pip install dcase_util
   pip install sed_vis
   ```

3. **Clone tập dữ liệu ESC-50:**

   ```bash
   git clone https://github.com/karolpiczak/ESC-50.git
   ```

## Cấu hình

Tất cả các thiết lập cấu hình (đường dẫn, tham số trích xuất đặc trưng, tham số huấn luyện,...) được định nghĩa trong file [`config.py`](config.py). Ví dụ:

```python
# filepath: config.py

# Đường dẫn lưu kết quả, đặc trưng và mô hình
RESULTS_PATH = "workspace/results"
FEATURES_PATH = "workspace/features"
MODELS_PATH = "workspace/models"

# Tham số trích xuất đặc trưng âm thanh
PARAMS_FEATURES = {
    'sr': 44100,
    'n_fft': 2048,
    'hop_length': 1024,
    'n_mels': 64,
    'fmin': 0,
    'fmax': 22050
}

# Tham số huấn luyện mô hình
PARAMS_MODEL = {
    'seed': 42,
    'batch_size': 64,
    'epochs': 30,  # Số epoch có thể thay đổi tùy theo yêu cầu
    'sequence_hop': 0.5,
    'sequence_length': 1.0,
    'preprocessing': 'normalization',
    'augmentation': None,
    'mixup': None
}
```

## Cách sử dụng

Dự án được chia thành một số module:

1. **Khởi tạo Dataset:**
   - File [`dataset.py`](dataset.py) cung cấp hàm `initialize_dataset()` để tải dữ liệu ESC-50 và đọc metadata (file âm thanh, nhãn).

2. **Trích xuất và Visualization đặc trưng:**
   - File [`feature_extraction.py`](feature_extraction.py) chứa các hàm `extract_features()` và `visualize_features()` dùng để trích xuất Mel Spectrogram và hiển thị kết quả.

3. **Huấn luyện mô hình:**
   - File [`model_training.py`](model_training.py) chứa hàm `train_and_evaluate()` dùng để huấn luyện mô hình phân loại (SB-CNN) trên tập dữ liệu.

4. **Dự đoán và Visualization kết quả:**
   - File [`visualization.py`](visualization.py) chứa hàm `predict_and_visualize()` để dự đoán lớp âm thanh từ file mới và hiển thị kết quả trực quan.

5. **Tích hợp quy trình:**
   - File [`main.py`](main.py) tích hợp các bước trên:
     - Tạo thư mục kết quả.
     - Khởi tạo dataset.
     - Trích xuất và visual hóa đặc trưng cho file mẫu.
     - Huấn luyện mô hình.
     - Dự đoán trên một file âm thanh khác.

Để chạy toàn bộ quy trình, chỉ cần chạy lệnh:

```bash
python main.py
```

## Cấu trúc dự án

```
tuanh/
├── config.py                # File cấu hình: đường dẫn, tham số đặc trưng & tham số huấn luyện.
├── dataset.py               # Khởi tạo dataset ESC-50 và xử lý metadata.
├── feature_extraction.py    # Trích xuất Mel Spectrogram và visualization đặc trưng.
├── model_training.py        # Huấn luyện mô hình phân loại (SB-CNN).
├── visualization.py         # Dự đoán và hiển thị kết quả dự đoán.
├── main.py                  # File chạy chính tích hợp toàn bộ quy trình.
└── README.md                # File hướng dẫn (bản này).
```

## Đóng góp

Các đóng góp và cải tiến cho dự án đều được hoan nghênh!  
Nếu bạn có ý kiến đóng góp, bạn có thể tạo một Pull Request hoặc mở issue để thảo luận.

## Giấy phép

Dự án này được cấp phép theo [MIT License](LICENSE).

---

Dự án này giúp bạn:
- Tự động trích xuất đặc trưng âm thanh từ file gốc.
- Huấn luyện mô hình CNN để phân loại âm thanh.
- Dự đoán nhãn của file âm thanh mới và trực quan hoá kết quả.
- Là nền tảng để so sánh hiệu năng của các phương pháp trích xuất đặc trưng và mô hình phân loại.
