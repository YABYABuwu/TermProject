# S&P500 Stock Prediction

Term Project — AI Engineer ภาคเรียนที่ 2  
การวิเคราะห์ข้อมูลและฝึกโมเดล Time-Series ด้วย AutoGluon

---

## สมาชิกกลุ่ม

| ชื่อ-นามสกุล       | รหัสนักศึกษา |
|--------------------|--------------|
| นายคุณัชญ์ ทวีรัตน์ | 6810110038   |
| นายธีรวัต แซ่น่ำ | 6810110163   |
| นายชนาธิป นุ้ยสี | 6810110566   |

---

## ภาพรวม

แดชบอร์ด Plotly Dash ที่รวมข้อมูลตลาดแบบ real-time กับ AutoGluon TimeSeriesPredictor เพื่อพยากรณ์และทดสอบย้อนหลัง (Backtest) ดัชนี S&P 500 พร้อมแสดงความสัมพันธ์ระหว่างสินทรัพย์ต่าง ๆ

แดชบอร์ดประกอบด้วย:

- KPI Cards — ตัวชี้วัดความแม่นยำของโมเดล
- กราฟพยากรณ์ S&P 500 — ราคาจริง vs. AI Backtest vs. พยากรณ์ 7 วันข้างหน้า
- กราฟการเติบโตของสินทรัพย์ — Growth (%) แบบ Normalize
- Heatmap ความสัมพันธ์ — Inter-asset correlation รายวัน
- แถบราคาล่าสุด — ทองคำ น้ำมัน ก๊าซธรรมชาติ Bitcoin

---


## สินทรัพย์ที่ติดตาม

| Ticker    | ชื่อสินทรัพย์  |
|-----------|----------------|
| `^GSPC`   | S&P 500        |
| `GC=F`    | ทองคำ          |
| `CL=F`    | น้ำมันดิบ      |
| `NG=F`    | ก๊าซธรรมชาติ  |
| `BTC-USD` | Bitcoin        |

ดึงข้อมูลจาก Yahoo Finance ย้อนหลัง 150 วัน รายวัน

---

## ความต้องการของระบบ

- Python 3.9 ขึ้นไป
- แพ็กเกจ: `autogluon.timeseries`, `dash`, `plotly`, `pandas`, `numpy`, `yfinance`

> หมายเหตุ: แนะนำให้สร้าง Virtual Environment แยกต่างหากเพื่อป้องกันความขัดแย้งของแพ็กเกจ

---

## วิธีติดตั้งและรัน

1. สร้าง Virtual Environment

```bash
conda create -n termproject python=3.10
conda activate termproject
```

2. ติดตั้ง Dependencies

```bash
pip install "autogluon.timeseries>=1.5.0" dash plotly pandas numpy yfinance
```

3. รันแดชบอร์ด

```bash
python app.py
```

4. เปิดเบราว์เซอร์ที่ `http://127.0.0.1:8050`

