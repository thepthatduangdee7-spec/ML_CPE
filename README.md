#Load Dataset
2.Display Shape
3.Display Data Types
4.Display Summary Statistics
5.Display Missing Values
6.Display Duplicate Records
7.Display Class Distribution
อธิบายโค้ดทีละบรรทัด
1. import pandas as pd 
คือ นำเข้าไลบรารี Pandas ชื่อย่อเป็น pd
เพื่อให้เรียกใช้งานง่าย แทนที่จะเขียน = pandas.read_csv()
เราใช้ชื่อย่อแทน  = pd.read_csv()
2. Load Dataset
df = pd.read_csv("student.csv") คือ อ่านไฟล์ student.csv เข้ามาเป็น DataFrame แล้วเก็บไว้ในตัวแปร df
DataFrame เปรียบเสมือนตาราง Excel
3. Display First 5 Rows
print("===== First 5 Rows =====")
print(df.head()) ใช้แสดงข้อมูล 5 แถวแรกเพื่อดูว่า โหลดข้อมูลถูกไหม ชื่อคอลัมน์ถูกไหม ข้อมูลอ่านถูกไหม
                        ตัวอย่าง
                        ID Name Age Faculty
                        1  A   20  IT
                        2  B   19  IT
                        3  C   18  Business
                        4  D   20  Engineering
                        5  E   21  IT
4. Display Shape
print(df.shape) แสดง จำนวนแถว จำนวนคอลัมน์
5. Display Data Types
print(df.dtypes) ใช้ดูชนิดข้อมูลของแต่ละคอลัมน์ เช่น
                        ID          int64
                        Age         int64
                        Name       object
                        Faculty    object
                        GPA       float64
6. Summary Statistics
print(df.describe()) ใช้สรุปสถิติของคอลัมน์ตัวเลข 
                        ตัวอย่าง 
                        count    100
                        mean      20.3
                        std        1.2
                        min       18
                        25%       19
                        50%       20
                        75%       21
                        max       24
ความหมายของแต่ละค่า
count = จำนวนข้อมูล 
mean = ค่าเฉลี่ย 
std = ส่วนเบี่ยงเบนมาตรฐานยิ่งมากแสดงว่าข้อมูลกระจายมาก
min = ค่าน้อยที่สุด
25% = ควอไทล์ที่ 1
50% = มัธยฐาน (Median)
75% = ควอไทล์ที่ 3
max = ค่ามากที่สุด
7. Missing Values
print(df.isnull().sum()) ใช้ตรวจสอบข้อมูลที่หายไป (Missing Values) isnull() จะตรวจสอบแต่ละช่องว่าเป็นค่าว่างหรือไม่ และ sum() จะนับจำนวนค่าว่างในแต่ละคอลัมน์
                        ตัวอย่าง
                        ID          0
                        Name        0
                        Age         2
                        Faculty     1
ทำไมต้องตรวจ Missing Value? เพราะถ้ามีข้อมูลหายอาจต้องลบข้อมูลเติมค่าเฉลี่ยเติมค่ากลางเติมค่าที่เหมาะสมก่อนวิเคราะห์
8. Duplicate Records
print(df.duplicated().sum()) ใช้ตรวจสอบข้อมูลซ้ำ duplicated() จะคืนค่า True สำหรับแถวที่ซ้ำกับแถวก่อนหน้า และ sum() จะนับจำนวนแถวที่ซ้ำทั้งหมด 
                        เช่น
                        0 False
                        1 False
                        2 True
                        3 False
9. Faculty Distribution
print(df["Faculty"].value_counts()) ใช้นับจำนวนแต่ละคณะ 
                        เช่น
                        IT             40
                        Business       30
                        Engineering    20
                        Science        10
ทำไมโจทย์เขียนว่า "Class Distribution" ในวิชา Data Science คำว่า Class Distribution หมายถึง การนับจำนวนข้อมูลของแต่ละกลุ่มใน Dataset
                        เช่น
                        Faculty
                        Gender
                        Department
                        Class
                        Label
แล้วแต่ Dataset ว่าจะใช้คอลัมน์ไหน
ในที่นี้ใช้ Faculty จึงเขียน df["Faculty"].value_counts()




#Load Dataset UTKFace Dataset
ติดตั้ง Library ที่จำเป็น
pip install numpy pandas matplotlib pillow scikit-learn
LAB 1: Regression (Regression.py)
  เน้นการทำนาย อายุ (Age) จากรูปภาพใบหน้า
    Simple Linear Regression: ดึง Feature เพียงพิกเซลเดียวตรงกลางภาพ (1 Feature) มาทำนายอายุ
    Multiple Linear Regression: ดึง Feature ทุกพิกเซลของภาพ (1,024 Features) มาประมวลผล
    Age Prediction S: แสดงตัวอย่างตารางเปรียบเทียบผลการทำนายอายุกับอายุจริง พร้อมวัดผลด้วยค่า MAE, RMSE และ R² Score

LAB 2: Classification (Classification.py)
   เน้นการจำแนก เพศ (Gender: ชาย/หญิง) จากรูปภาพใบหน้า
    Preparing Classification Data: จัดทำ Feature Scaling (StandardScaler) และลดมิติข้อมูลด้วย PCA เหลือ 2D
    Logistic Regression: การเทรนโมเดล Logistic Regression พร้อมแสดงค่า Accuracy Score
    Gender Prediction: ตารางแสดงตัวอย่างผลการทำนายเพศเทียบกับค่าจริง
    Confusion Matrix: แสดงค่า Confusion Matrix Array และ Classification Report (Precision / Recall / F1-Score)
    Decision Boundary Visualization: พล็อต กราฟ Decision Boundary 2D ร่วมกับ Confusion Matrix Display

    📊 Model Comparison (สรุปและเปรียบเทียบประสิทธิภาพโมเดล)

การเปรียบเทียบการทำงานและประสิทธิภาพของแต่ละอัลกอริทึมใน **LAB 1** และ **LAB 2**:

### 📑 ตารางเปรียบเทียบเชิงสรุป

| Model Algorithm | Task Type | Features Used | Key Metrics (โดยประมาณ) | สรุปผลและจุดเด่น/ข้อจำกัด |
| :--- | :--- | :--- | :--- | :--- |
| **Simple Linear Regression** *(LAB 1)* | Regression (ทำนายอายุ) | 1 พิกเซล (ตรงกลางภาพ) | • High MAE<br>• Low R² | **ข้อจำกัด:** พิกเซลจุดเดียวไม่เพียงพอที่จะระบุความเปลี่ยนแปลงของอายุบนใบหน้าได้ |
| **Multiple Linear Regression** *(LAB 1)* | Regression (ทำนายอายุ) | 1,024 พิกเซล (ทั้งภาพ $32 \times 32$) | • Lower MAE<br>• Higher R² | **จุดเด่น:** ประสิทธิภาพดีขึ้นอย่างชัดเจน เนื่องจากรับข้อมูลโครงสร้างใบหน้าครบทุกมิติ |
| **Logistic Regression (+ PCA 2D)** *(LAB 2)* | Classification (ทำนายเพศ) | 2 Components (PCA 2D) | • Accuracy ~70–75%<br>• F1-Score ~0.73 | **จุดเด่น:** ลดมิติข้อมูลช่วยให้ประมวลผลไว และนำไปพล็อต กราฟ **Decision Boundary 2D** ได้ง่าย |

---

### 💡 สรุปวิเคราะห์ผลการทดลอง (Key Takeaways)

1. **การเพิ่มจำนวน Features ใน Regression (LAB 1):**
   * **Simple Linear Regression** ที่ใช้เพียงพิกเซลเดียวตรงกลางภาพ ไม่สามารถสร้างความสัมพันธ์ที่ดีกับอายุจริงได้ ทำให้มีความคลาดเคลื่อน (MAE) สูง
   * **Multiple Linear Regression** สามารถพิจารณาพิกเซลทั้งหมดบนใบหน้าพร้อมกัน ส่งผลให้ค่าความคลาดเคลื่อนลดลงอย่างเห็นได้ชัด และทำนายอายุได้แม่นยำยิ่งขึ้น

2. **ผลของการทำ Dimensionality Reduction ด้วย PCA (LAB 2):**
   * การบีบอัดข้อมูลภาพจาก **1,024 พิกเซล เหลือเพียง 2 มิติ (PCA Components)** ช่วยลดภาระการคำนวณของโมเดล **Logistic Regression** ได้อย่างมหาศาล
   * แม้การลดมิติจะทำให้สูญเสียรายละเอียดบางส่วนไปบ้าง (Information Loss)

โปรเจกต์นี้เป็นการทดลองเปรียบเทียบประสิทธิภาพของโมเดล Machine Learning ในการจำแนกประเภทข้อมูลดอกไม้ Iris (Iris Dataset) โดยแบ่งออกเป็น 2 การทดลองหลัก ได้แก่ Support Vector Machine (Lab 5) และ Neural Network (Lab 6)
ความต้องการของระบบ (Requirements)
ก่อนเริ่มใช้งาน กรุณาติดตั้งไลบรารีที่จำเป็นด้วยคำสั่ง:
  pip install pandas matplotlib numpy scikit-learn
รายละเอียดการทดลองLab 5: Support Vector Machine (SVM) Classificationทดสอบประสิทธิภาพการทำงานของอัลกอริทึม SVM 
โดยเปรียบเทียบการใช้งาน Kernel Function 3 รูปแบบ:  
 1. Linear Kernel  
 2. Polynomial Kernel  
 3. Radial Basis Function (RBF) Kernel  
ขั้นตอนการทำงาน:โหลดข้อมูล Iris Dataset และแบ่งข้อมูลออกเป็น Train 70% และ Test 30%  
  1.ปรับมาตรฐานข้อมูลด้วย StandardScaler  
  2.เทรนโมเดล SVM ทั้ง 3 Kernels และคำนวณค่า Accuracy Score  
  แสดงผลลัพธ์:
    เปรียบเทียบ Accuracy Score ของแต่ละ Kernel ผ่าน Console และ กราฟแท่ง (Bar Chart)  
    เปรียบเทียบผลการทำนายจริง (Actual Target) กับ ผลการทำนายของ RBF Kernel (Predicted Target) จำนวน 15 ตัวอย่างแรก ผ่าน Console และ กราฟ Scatter Plot  

Lab 6: Neural Network (MLPClassifier) Classificationศึกษาผลกระทบของการปรับเปลี่ยนโครงสร้างซ่อน (Hidden Layers) และจำนวนรอบการเรียนรู้ (Epochs) ใน Neural Network (MLPClassifier)         เงื่อนไขการทดลอง:Architecture Configurations:
      Config A: 1 Hidden Layer (10 Neurons) 
      Config B: 2 Hidden Layers (10, 10 Neurons)  
      Config C: 2 Hidden Layers (20, 10 Neurons)  
    Epochs: 20, 50, และ 200 รอบ 
    ขั้นตอนการทำงาน:
      1.โหลดข้อมูลและสเกลข้อมูลด้วย StandardScaler 
      2.วนลูปเทรนโมเดล MLP ตาม Config และ Epochs ที่กำหนด 
      3.เก็บข้อมูล Train Loss, Validation Accuracy และ Test Accuracy ในแต่ละรอบ 
    แสดงผลลัพธ์:
      ตารางสรุปค่า Accuracy Score, Train Loss และ Validation Accuracy ของทุก Configuration  
      ตัวอย่างผลการทำนายจริงเทียบกับผลทำนายจากโมเดล 15 ตัวอย่าง  กราฟ 3 รูปแบบ:เปรียบเทียบ Accuracy vs. Epochs ของแต่ละ Config  แสดงแนวโน้มการลดลงของ Loss (Training Loss Curve) ที่ 200 Epochs  แสดงแนวโน้มการเพิ่มขึ้นของ Validation Accuracy ที่ 200 Epochs  
