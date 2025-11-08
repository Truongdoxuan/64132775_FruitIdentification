import cv2
import numpy as np
import sklearn
import os

# --- Chỉ để in phiên bản, kiểm tra xem thư viện đã cài đúng chưa ---
print("--- Kiem tra thu vien ---")
print(f"OpenCV version: {cv2.__version__}")
print(f"Numpy version: {np.__version__}")
print(f"Scikit-learn version: {sklearn.__version__}")
print("-------------------------\n")


# --- Thay đổi đường dẫn này thành 1 ảnh bất kỳ bạn vừa tải về ---
image_path = "dataset/apple/tao_01.jpg" # HOẶC "dataset/chuoi/chuoi_01.jpg", v.v.

# --- Bắt đầu kiểm tra OpenCV ---
if not os.path.exists(image_path):
    print(f"LỖI: Không tìm thấy file ảnh tại: {image_path}")
    print("Mẹo: Hãy đảm bảo bạn đã tạo thư mục 'dataset/tao' và có ảnh 'tao_01.jpg' bên trong.")
else:
    # 1. Đọc ảnh
    img = cv2.imread(image_path)

    if img is None:
        print(f"LỖI: OpenCV không thể đọc được ảnh tại: {image_path}")
    else:
        print(f"THÀNH CÔNG! Đã đọc được ảnh: {image_path}")
        print(f"Kích thước ảnh (Cao, Rộng, Kênh màu): {img.shape}")
        
        # 2. Hiển thị ảnh
        cv2.imshow("Anh kiem tra - Nhan phim bat ky de thoat", img)
        
        print("\nMột cửa sổ ảnh vừa bật lên. Hãy nhấn phím bất kỳ (ví dụ 'q') để đóng nó.")
        
        # 3. Chờ người dùng nhấn phím
        cv2.waitKey(0)
        
        # 4. Đóng cửa sổ
        cv2.destroyAllWindows()