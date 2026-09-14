Support Vector Machine (SVM) Classification - Iris Dataset
โปรเจกต์นี้เป็นการทดลองใช้งานอัลกอริทึม Support Vector Machine (SVM) ในการจำแนกประเภทข้อมูลดอกไอริส (Iris Dataset) เพื่อศึกษาผลกระทบของการใช้ Kernel Functions รูปแบบต่างๆ รวมถึงกระบวนการทำ Data Preprocessing อย่างถูกต้อง
ฟีเจอร์เด่นของโปรเจกต์
 * Data Preprocessing: การแบ่งข้อมูลเป็น Train/Test Set และการทำ Standardization ด้วย StandardScaler
 * Kernel Comparison: เปรียบเทียบประสิทธิภาพการจำแนกข้อมูลของ SVM ทั้งหมด 3 Kernels ได้แก่ Linear, Polynomial, และ RBF
 * Data Visualization: แสดงผลการเปรียบเทียบค่าความแม่นยำ (Accuracy) และกราฟ Scatter Plot เปรียบเทียบค่าจริง vs ค่าที่โมเดลทำนายได้
ชุดข้อมูล (Dataset)
ใช้ชุดข้อมูล Iris Dataset จากไลบรารี scikit-learn โดยไม่ต้องดาวน์โหลดไฟล์แยกเพิ่มเติม
 * จำนวนข้อมูล: 150 ตัวอย่าง
 * Features: ความกว้าง/ยาวของกลีบเลี้ยง (Sepal) และกลีบดอก (Petal)
 * Target Classes: สายพันธุ์ดอกไอริส 3 สายพันธุ์ (Setosa, Versicolor, Virginica)
ภาพรวมขั้นตอนการทำงาน
 * Load Dataset: ดึงข้อมูล Iris เข้ามาแปลงให้อยู่ในรูปแบบ Pandas DataFrame
 * Data Splitting: แบ่งข้อมูลด้วยอัตราส่วน Train 70% (105 ตัวอย่าง) และ Test 30% (45 ตัวอย่าง)
 * Data Standardization:
   * คำนวณและปรับสเกลข้อมูลชุด Train ด้วย fit_transform
   * ปรับสเกลข้อมูลชุด Test ด้วย transform เพื่อป้องกันปัญหา Data Leakage
 * Model Training & Evaluation: วนลูปเทรนโมเดล SVC ตามแต่ละ Kernel (linear, poly, rbf) และวัดค่า Accuracy Score
 * Visualization: สร้างกราฟเปรียบเทียบด้วย matplotlib
การติดตั้งและการใช้งาน
1. Requirements
ติดตั้งแพ็กเกจที่จำเป็นก่อนเริ่มใช้งาน:
pip install numpy pandas matplotlib scikit-learn

2. Running the Code
รันสคริปต์ Python ด้วยคำสั่ง:
python "Lab 5.py"

ผลลัพธ์และการแสดงผล (Output)
 * Terminal Output: แสดงค่าความแม่นยำ (%) ของแต่ละ Kernel และเปรียบเทียบคำตอบ 15 ตัวอย่างแรก
 * Graphical Output:
   * Graph 1: กราฟแท่งเปรียบเทียบ Accuracy Score ระหว่าง Linear, Poly, และ RBF Kernel
   * Graph 2: กราฟเปรียบเทียบตำแหน่งค่าจริง (Actual) และค่าทำนาย (Predicted) ของ RBF Kernel
ต้องการให้ปรับแก้ส่วนไหนเพิ่มเติม หรืออยากให้นำเสนอเป็นเวอร์ชันภาษาอังกฤษบอกได้เลยครับ

