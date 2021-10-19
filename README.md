# Text Classification
Classifying text content

# Resources
Dataset: <br> 
https://drive.google.com/drive/folders/1r8mHXQoz9JF5aqHBF2HnYfu2SzJzqEpn?usp=sharing <br>
Word vectors:<br> 
https://github.com/Kyubyong/wordvectors

# Tasks
* Làm sạch dữ liệu, tách từ, chuẩn hóa từ, loại bỏ stopword.
* Phân loại dữ liệu văn bản.
* Xây dựng mô hình (sử dụng ít nhất 3 giải thuật)
* Đánh giá mô hình.
* Xây dựng Restful API phân loại.
* Xây dựng website/ứng dụng trên di động.

# Work flow
* Các thư viện cần thiết: `pip install -r setup.txt`
* Tải các resources cần thiết và lưu vào thư mục project
* Chạy `python build.py` hoặc `build.py` để build data
* Chạy `python train.py` hoặc `train.py` để kiểm tra quá trình huấn luyện các mô hình
* Chạy `python predict.py` hoặc `predict.py` để tiến hành dự đoán kết quả phân loại văn bản.