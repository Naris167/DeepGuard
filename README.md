# Fall Detection System Using Deep Learning with Email Notification

## 📋 คำอธิบายโครงงาน

### ปัญหา
ญาติ หรือผู้ใกล้ชิด/ผู้ดูแลของผู้สูงอายุ อาจไม่ทราบว่าผู้สูงอายุที่อยู่ในความดูแลมีการหกล้มภายในบ้านขณะอยู่ลำพัง และเนื่องจากผู้สูงอายุส่วนใหญ่ไม่สามารถช่วยเหลือตนเองหรือขอความช่วยเหลือได้ อาจนำไปสู่อันตรายร้ายแรงในกรณีที่ไม่ได้รับการช่วยเหลือทันท่วงที

### วัตถุประสงค์
เพื่อศึกษาและพัฒนาระบบตรวจจับการล้มด้วยเทคนิคการเรียนรู้เชิงลึก (Deep Learning) พร้อมแจ้งเตือนผ่านโทรศัพท์มือถือ เพื่อให้ผู้ดูแลสามารถรับทราบเหตุการณ์การหกล้มได้อย่างรวดเร็วและทันท่วงที

### กลุ่มเป้าหมาย
ญาติ หรือผู้ใกล้ชิด/ผู้ดูแลของผู้สูงอายุ เพื่อให้สามารถได้รับแจ้งเตือนที่รวดเร็วในการรับรู้ถึงเหตุการณ์การหกล้มของผู้สูงวัย

### เทคโนโลยีที่ใช้
- **YOLO11-Pose**: สำหรับตรวจจับและติดตาม Pose Estimation (Body Keypoints) ของบุคคลในวิดีโอ
- **LSTM (Long Short-Term Memory)**: สำหรับวิเคราะห์ลำดับการเคลื่อนไหวของร่างกายเพื่อตรวจจับการล้ม
- **Deep Learning Framework**: TensorFlow/Keras สำหรับการสร้างและเทรนโมเดล

### วิธีการทำงาน
1. เปิดไฟล์วิดีโอแบบ Real-time
2. YOLO11-Pose ตรวจจับตำแหน่งของร่างกาย (17 keypoints) ในแต่ละเฟรม
3. LSTM วิเคราะห์ลำดับของ keypoints จาก 48 เฟรม (2 วินาที) เพื่อทำนายว่ามีการล้มเกิดขึ้นหรือไม่
4. เมื่อตรวจพบการล้ม ระบบจะบันทึกภาพและแจ้งเตือนผู้ดูแล

---

## 🚀 วิธีติดตั้ง

### ข้อกำหนดเบื้องต้น
- Anaconda หรือ Miniconda
- GPU (แนะนำสำหรับการเทรนโมเดล)

### ขั้นตอนการติดตั้ง

#### 1. สร้าง Environment จากไฟล์ `environment.yaml`
```bash
# ติดตั้งผ่าน Anaconda Navigator
# 1. เปิด Anaconda Navigator
# 2. ไปที่แท็บ Environments
# 3. คลิก Import
# 4. เลือกไฟล์ environment.yaml
# 5. ตั้งชื่อ environment (เช่น fall_detection)
# 6. คลิก Import

# หรือติดตั้งผ่าน Command Line
conda env create -f environment.yaml
conda activate fall_detection
```

#### 2. ดาวน์โหลดโมเดล YOLO11-Pose
```bash
# สามารถใช้ไฟล์ที่อยู่ใน ./models/yolo11l-pose.pt ได้เลย

# หรือดาวน์โหลด YOLO11l-Pose model อื่น ๆ ได้ที่
# https://github.com/ultralytics/assets/releases
# วางไฟล์ไว้ใน ./models/
```

---

## 🎯 วิธีรันโค้ดเพื่อให้ Reproducible

### ขั้นตอนที่ 1: เตรียมข้อมูล (Data Preparation)

สคริปต์นี้จะอ่านไฟล์ label video (JSON) และประมวลผลให้เป็นไฟล์ PKL เพื่อใช้ในการเทรนโมเดล
```bash
python "scripts/[LSTM] 1 data prep.py"
```

**สิ่งที่สคริปต์ทำ:**
- สแกนหาไฟล์ JSON ในโฟลเดอร์ทุกระดับ
- รวม (merge) ข้อมูลของบุคคลเดียวกันที่มี `person_id` เดียวกัน
- ตรวจสอบคุณภาพข้อมูล (Quality checks)
- Normalize keypoints ด้วย bounding box
- บันทึกเป็นไฟล์ PKL (1 ไฟล์ต่อ 1 คน)

**ผลลัพธ์:** ไฟล์ PKL ถูกบันทึกใน `./preprocessed_data/`

---

### ขั้นตอนที่ 2: เทรนโมเดล (Model Training)

สคริปต์นี้จะโหลดข้อมูล PKL และเทรนโมเดล LSTM
```bash
python "scripts/[LSTM] 2 train model.py"
```

**การตั้งค่าสำคัญ:**
- **Seed Value = 12345** (เพื่อให้ผลลัพธ์ reproducible)
- Window Size = 48 frames (2 วินาที)
- Stride = 1 frame
- LSTM Architecture: 128 → 64 units
- Batch Size = 32
- Epochs = 50 (มี Early Stopping)

**สิ่งที่สคริปต์ทำ:**
1. โหลดไฟล์ PKL ทั้งหมด
2. Forward-fill สำหรับ null frames
3. สร้าง sliding windows (48 frames)
4. Label windows (≥6 consecutive fall frames = fall)
5. Balance dataset (hybrid strategy)
6. เทรนโมเดล LSTM
7. บันทึกโมเดล, training history plot, และ confusion matrix

**ผลลัพธ์:** 
- โมเดล: `./models/fall_detection_{datetime}.h5`
- Training History: `./models/his_{datetime}.png`
- Confusion Matrix: `./models/cf_{datetime}.png`

---

### ขั้นตอนที่ 3: รัน Demo (Inference)

สคริปต์นี้จะโหลดวิดีโอและรันการตรวจจับการล้มแบบ Real-time
```bash
python "scripts/[LSTM] 3 run demo.py"
```

**วิธีใช้งาน:**
1. แก้ไข `VIDEO_PATH` ในไฟล์เพื่อเลือกวิดีโอที่ต้องการทดสอบ:
```python
   VIDEO_PATH = "path/to/your/video.mp4"
```

2. แก้ไข `LSTM_MODEL_PATH` เป็นโมเดลที่เทรนเสร็จแล้ว:
```python
   LSTM_MODEL_PATH = "./models/fall_detection_20251017_213109.h5"
```

3. รันสคริปต์

**การทำงาน:**
- ประมวลผลทีละ 1 วินาที (24 frames)
- ติดตามแต่ละบุคคลด้วย YOLO track_id
- แสดง bounding box:
  - **สีเขียว**: ไม่มีการล้ม
  - **สีแดง**: ตรวจพบการล้ม
- ใช้ Moving Average (3 predictions) เพื่อลดสัญญาณรบกวน
- บันทึก 12 ภาพเมื่อตรวจพบการล้ม

**ผลลัพธ์:**
- ภาพที่บันทึก: `./demos/{datetime}_{video_name}/{frame}_{confidence:.3f}_person_{id}.png`

**การตั้งค่าเพิ่มเติม:**
```python
SHOW_BBOX = True   # ตั้งเป็น False เพื่อซ่อน bounding box
CONFIDENCE_THRESHOLD = 0.5   # ค่า threshold สำหรับการตรวจจับ
```

## 🔬 รายละเอียดเทคนิค

### ขนาด Input Shape
- **(48, 17, 3)**: 48 เฟรม, 17 keypoints, 3 features (x, y, confidence)
- ผ่านการ normalize ด้วย bounding box เป็นค่า [0, 1]

### LSTM Architecture
```
Input (48, 17, 3)
    ↓
Reshape to (48, 51)
    ↓
LSTM(128) + Dropout(0.3)
    ↓
LSTM(64) + Dropout(0.3)
    ↓
Dense(32, relu) + Dropout(0.3)
    ↓
Dense(1, sigmoid)
    ↓
Output: Probability [0, 1]
```

### การจัดการข้อมูลที่หายไป (Null Handling)
- **Forward-fill**: คัดลอกค่าจากเฟรมล่าสุด confidence=0 (สูงสุด 48 เฟรม)
- **Fallback**: ใช้ [-1, -1, -1] เมื่อไม่มีข้อมูลก่อนหน้าหรือช่องว่างยาวเกินไป

---

## 👥 ผู้พัฒนา

1. 6814400944	พงศธร สุดลาภา
2. 6814450020	ซอฟาอ์ เตะโระ
3. 6814450046	นริศ พรจิรวิทยากุล
4. 6814450143	ชนธีร์ สุรกุล
5. 6814450151	ปิยชนม์ หรูสุวรณณกุล
6. 6817400082	ณัฐกานติ์ โตนวล


---

**หมายเหตุ**: โปรเจคนี้พัฒนาขึ้นเพื่อวัตถุประสงค์ทางการศึกษา และเพื่อช่วยเหลือผู้สูงอายุและผู้ดูแล