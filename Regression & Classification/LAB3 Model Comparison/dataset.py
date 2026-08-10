import os
import glob
import numpy as np
from PIL import Image

def load_utkface(dataset_dir, num_samples=2000, target_size=(32, 32)):
    """
    ฟังก์ชันสำหรับดึงรูปภาพ UTKFace และแปลงเป็นข้อมูลตัวเลข
    """
    image_paths = glob.glob(os.path.join(dataset_dir, "*.jpg"))
    
    if len(image_paths) == 0:
        raise FileNotFoundError(f"ไม่พบไฟล์ .jpg ในโฟลเดอร์: {dataset_dir}")
        
    print(f"พบบนเครื่องทั้งหมด: {len(image_paths)} ภาพ")
    image_paths = image_paths[:num_samples]

    X_list = []
    y_age_list = []
    y_gender_list = []

    for path in image_paths:
        filename = os.path.basename(path)
        parts = filename.split('_')
        
        # รูปแบบไฟล์ UTKFace: [age]_[gender]_[race]_[date].jpg
        if len(parts) >= 3:
            try:
                age = int(parts[0])
                gender = int(parts[1])
                
                img = Image.open(path).convert('L')
                img = img.resize(target_size)
                
                img_array = np.array(img).flatten() / 255.0
                
                X_list.append(img_array)
                y_age_list.append(age)
                y_gender_list.append(gender)
            except ValueError:
                continue

    X_features = np.array(X_list)
    y_age = np.array(y_age_list)
    y_gender = np.array(y_gender_list)

    print(f"เตรียมข้อมูลเรียบร้อย! ดึงมาทั้งหมด {len(X_features)} ภาพ")
    return X_features, y_age, y_gender