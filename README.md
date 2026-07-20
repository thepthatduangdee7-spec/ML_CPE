# .Load Dataset
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
