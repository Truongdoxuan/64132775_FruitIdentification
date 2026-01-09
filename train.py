import cv2
import numpy as np
import os
import joblib # Dùng để lưu mô hình
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


def extract_color_histogram(image):
    histSize = [8, 8, 8]
    
    ranges = [0, 256, 0, 256, 0, 256]
    
    hist = cv2.calcHist([image], [0, 1, 2], None, histSize, ranges)
    
    cv2.normalize(hist, hist)
    
    return hist.flatten()


def extract_hu_moments(image):
    # Chuyển sang ảnh xám
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Phân ngưỡng (threshold) để tách vật thể khỏi nền
    # (Giả định vật thể sáng hơn nền một chút)
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    
    # Tính 7 giá trị Hu Moments
    moments = cv2.moments(thresh)
    huMoments = cv2.HuMoments(moments)
    
    # Scale lại Hu moments (dùng log) để ổn định giá trị
    for i in range(0, 7):
        # Dùng abs để tránh log(số âm) và 1e-5 để tránh log(0)
        huMoments[i] = -1 * np.sign(huMoments[i]) * np.log10(abs(huMoments[i] + 1e-5))
        
    # Làm phẳng thành vector 1 chiều (7 chiều)
    return huMoments.flatten()


# TẢI DỮ LIỆU VÀ HUẤN LUYỆN
print("Bat dau qua trinh huan luyen...")

dataset_path = "dataset/"

label_map = {"apple": 0, "banana": 1, "orange": 2}

data = []
labels = []

RESIZE_DIM = (100, 100)

for folder_name in os.listdir(dataset_path):
    folder_path = os.path.join(dataset_path, folder_name)
    
    if not os.path.isdir(folder_path):
        continue
        
    label = label_map.get(folder_name)
    if label is None:
        print(f"Canh bao: Bo qua thu muc la {folder_name}")
        continue
        
    print(f"Dang xu ly thu muc: {folder_name} (Nhan: {label})")
    
    for image_name in os.listdir(folder_path):
        image_path = os.path.join(folder_path, image_name)
        
        img = cv2.imread(image_path)
        
        if img is None:
            print(f"  > Loi: Khong doc duoc anh {image_name}")
            continue
        
        img_resized = cv2.resize(img, RESIZE_DIM)
        
        color_features = extract_color_histogram(img_resized)
        shape_features = extract_hu_moments(img_resized)
        
        # --- Kết hợp 2 loại đặc trưng lại ---
        # (1024 + 7 = 1031 chiều)
        combined_features = np.concatenate([color_features, shape_features])
        
        # Thêm vector đặc trưng và nhãn vào danh sách
        data.append(combined_features)
        labels.append(label)

print("\nDa trich xuat dac trung thanh cong cho toan bo dataset!")
print(f"Tong so luong anh duoc xu ly: {len(data)}")

# CHUẨN BỊ DỮ LIỆU ĐỂ HUẤN LUYỆN 
# Chuyển danh sách (list) sang mảng NumPy
X = np.array(data)
y = np.array(labels)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Kich thuoc tap huan luyen: {len(X_train)} mau")
print(f"Kich thuoc tap kiem thu: {len(X_test)} mau")

#HUẤN LUYỆN MÔ HÌNH 
print("\nBat dau huan luyen mo hinh (KNN)...")

model = KNeighborsClassifier(n_neighbors=5) # Sử dụng 5 hàng xóm

model.fit(X_train, y_train)

print("Huan luyen thanh cong!")

#ĐÁNH GIÁ MÔ HÌNH 
print("\nDanh gia mo hinh...")

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Do chinh xac (Accuracy): {accuracy * 100:.2f}%")

#LƯU MÔ HÌNH
model_filename = "fruit_model.pkl"
joblib.dump(model, model_filename)

print(f"\nDa luu mo hinh thanh cong vao file: {model_filename}")