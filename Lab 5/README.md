
Iris Dataset Classification with SVM 
โปรเจกต์นี้เป็นการเทรนและประเมินผลโมเดล Support Vector Machine (SVC) บนชุดข้อมูล Iris Dataset ด้วยการเปรียบเทียบประสิทธิภาพของ 3 Kernels (Linear, Polynomial, RBF) พร้อมแสดงผลลัพธ์ผ่าน Console และการพล็อต กราฟด้วย Matplotlib
ภาพรวมทฤษฎีและแนวคิดที่ใช้ในโค้ด
1. Support Vector Machine (SVM) & Support Vectors
โมเดลจะทำการหาเส้นหรือระนาบแบ่งที่เหมาะสมที่สุด (Optimal Hyperplane) เพื่อแยกข้อมูลแต่ละสายพันธุ์ออกจากกันโดยพยายามขยายระยะห่าง (Margin) ระหว่างระนาบแบ่งกับจุดข้อมูลที่ใกล้ที่สุด (Support Vectors) ให้มากที่สุด
![SVM Concept](ML-05-SVM (2).pdf#page=4&rect=23,109,977,910)
2. Kernel Functions (Linear, Poly, RBF)
ในกรณีที่ข้อมูลไม่สามารถแบ่งด้วยเส้นตรงได้ (Non-linear Data) โค้ดมีการทดสอบใช้ Kernel Functions ทั้ง 3 ชนิด (linear, poly, rbf) เพื่อแปลงข้อมูลไปสู่มิติที่สูงขึ้นและสร้างขอบเขตการตัดสินใจที่เหมาะสม


ขั้นตอนการทำงานของโค้ด
 * Load Dataset: โหลดข้อมูล Iris จาก sklearn.datasets แล้วแปลงข้อมูลฟีเจอร์เป็น Pandas DataFrame
 * Split Dataset: แบ่งชุดข้อมูลเป็น Train 70% (105 ตัวอย่าง) และ Test 30% (45 ตัวอย่าง) ด้วยคำสั่ง train_test_split(..., test_size=0.3, random_state=42)
 * Standardize Features: ทำการปรับสเกลข้อมูลด้วย StandardScaler
   * ใช้ fit_transform() บน X_train
   * ใช้ transform() บน X_test เพื่อป้องกันปัญหา Data Leakage
 * Train & Predict: วนลูปสร้างโมเดล SVC ตาม 3 Kernels (linear, poly, rbf) สั่ง fit กับข้อมูลชุด Train ทำการ predict ข้อมูลชุด Test และคำนวณ accuracy_score เป็นเปอร์เซ็นต์
 * Print Outputs: แสดงผล Accuracy Scores ของแต่ละ Kernel และพิมพ์ตัวอย่างผลการทำนาย 15 ตัวอย่างแรกของ RBF Kernel เทียบกับ Actual Labels
 * Data Visualization: สร้าง กราฟ 2 รูปในภาพเดียวด้วย plt.subplots(1, 2)
   * Bar Chart: แสดงการเปรียบเทียบค่า Accuracy Score (%) ของแต่ละ Kernel
   * Scatter Plot: แสดงจุดเปรียบเทียบค่า Actual Label (วงกลมสีฟ้า) และ Predicted Label ของ RBF (กากบาทสีแดง) สำหรับ 15 ตัวอย่างแรก
Requirements & Libraries
ไลบรารีที่ใช้ในโค้ดนี้ประกอบด้วย:
 * pandas
 * matplotlib
 * numpy
 * scikit-learn (load_iris, train_test_split, StandardScaler, SVC, accuracy_score)
pip install pandas matplotlib numpy scikit-learn

การรันสคริปต์
python "Lab 5.py"

สรุปโครงสร้าง Output จากโค้ด
 * Console Output:
   * แสดงค่า Accuracy Score ของ Linear, Poly, และ RBF
   * แสดงลิสต์คำตอบจริง y_test[:15] และคำตอบที่ทำนายได้ predictions['rbf'][:15]
 * Graphical Output: กราฟแสดงผล 2 กราฟควบคู่กัน (Accuracy Comparison & Actual vs Predicted Scatter Plot)

