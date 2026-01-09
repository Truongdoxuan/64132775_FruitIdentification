import cv2
import numpy as np
import joblib # Dùng để ĐỌC mô hình
import os



def extract_color_histogram(image):
    histSize = [8, 8, 8]
    ranges = [0, 256, 0, 256, 0, 256]
    hist = cv2.calcHist([image], [0, 1, 2], None, histSize, ranges)
    cv2.normalize(hist, hist)
    return hist.flatten()

def extract_hu_moments(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    moments = cv2.moments(thresh)
    huMoments = cv2.HuMoments(moments)
    for i in range(0, 7):
        huMoments[i] = -1 * np.sign(huMoments[i]) * np.log10(abs(huMoments[i] + 1e-5))
    return huMoments.flatten()

#CHƯƠNG TRÌNH NHẬN DIỆN
print("Dang tai mo hinh...")
model = joblib.load("fruit_model.pkl") 
print("Da tai xong!")

label_name = {0: "Tao (Apple)", 1: "Chuoi (Banana)", 2: "Cam (Orange)"}

test_image_path = "img_test/cam.jpg" 

RESIZE_DIM = (100, 100)

if not os.path.exists(test_image_path):
    print(f"LOI: Khong tim thay file anh: {test_image_path}")
    print("Hay copy mot tam anh bat ky vao thu muc va doi ten thanh 'test_image.jpg' de thu.")
else:
    # Đọc ảnh gốc
    original_img = cv2.imread(test_image_path)
    
    # Resize để xử lý
    img_process = cv2.resize(original_img, RESIZE_DIM)
    
    color_features = extract_color_histogram(img_process)
    shape_features = extract_hu_moments(img_process)
    combined_features = np.concatenate([color_features, shape_features])
    
    # --- Dự đoán ---
    # Máy yêu cầu dữ liệu dạng mảng 2 chiều, nên cần reshape(1, -1)
    prediction = model.predict(combined_features.reshape(1, -1))
    
    # Lấy kết quả (số 0, 1 hoặc 2)
    result_number = prediction[0]
    result_text = label_name[result_number]
    
    print(f"\nKET QUA DU DOAN: {result_text}")
    
    # --- Hiển thị kết quả lên ảnh ---
    # Viết chữ lên ảnh (Màu xanh lá, độ dày 2)
    cv2.putText(original_img, f"Ket qua: {result_text}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    
    cv2.imshow("Ket qua Nhan dien", original_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()