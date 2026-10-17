import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np

def train_and_save_dcnn():
    print("กำลังโหลดชุดข้อมูล Fashion-MNIST สำหรับ DCNN...")
    # โหลดชุดข้อมูล Fashion-MNIST (ภาพเสื้อผ้า รองเท้า กระเป๋า ขนาด 28x28 พิกเซล 10 คลาส)
    fashion_mnist = tf.keras.datasets.fashion_mnist
    (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

    # Normalize ค่าพิกเซลให้อยู่ในช่วง 0 - 1 และปรับมิติข้อมูล
    X_train = X_train.astype('float32') / 255.0
    X_test = X_test.astype('float32') / 255.0
    X_train = np.expand_dims(X_train, -1)
    X_test = np.expand_dims(X_test, -1)

    print("กำลังสร้างสถาปัตยกรรม Deep Convolutional Neural Network (DCNN)...")
    # ออกแบบ DCNN ให้มีความลึกหลายเลเยอร์ (Deep Layers)
    model = models.Sequential([
        layers.Input(shape=(28, 28, 1)),
        
        # Block 1: Feature Extraction ลำดับที่ 1
        layers.Conv2D(32, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Block 2: Feature Extraction ลำดับที่ 2 (เพิ่มความลึกเป็น DCNN)
        layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Fully Connected Layers (Classifier)
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(10, activation='softmax') # 10 คลาสแฟชั่น
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    print("กำลังเทรนโมเดล DCNN (อาจใช้เวลาสักครู่)...")
    model.fit(X_train, y_train, epochs=5, batch_size=64, validation_data=(X_test, y_test))

    # บันทึกไฟล์โมเดล DCNN เป็นชื่อเดิมหรือใหม่ก็ได้ (ในที่นี้ใช้ model_cnn.h5 เพื่อให้ app.py เรียกใช้ง่าย)
    model.save("model_cnn.h5")
    print("บันทึกโมเดล DCNN เป็นไฟล์ 'model_cnn.h5' สำเร็จเรียบร้อย!")

if __name__ == "__main__":
    train_and_save_dcnn()