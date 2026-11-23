# Curriculum design (v1 · 2026-11-23)

> โครงสร้างตาม `tda-pttgc-int3`: หน้าทฤษฎี + quiz ต่อ module · lab เขียนจากต้นฉบับเดียว (percent format) → notebook ผู้เรียน (TODO + ✅/❌) และเฉลยที่รันแล้ว · checkpoint ทุกหัวข้อ
> **ยังไม่มีวิดีโอ** (ตัดสินใจ 2026-11-23) — หน้าทฤษฎีเขียนให้อ่านจบในตัว ถ้าทำวิดีโอภายหลังให้ฝังไว้หัวหน้าแต่ละ module

## ผู้เรียนและรูปแบบ

- Python พื้นฐาน (pandas เบื้องต้น) · ไม่จำเป็นต้องมีพื้น ML · ใช้เครื่องตัวเองทั้ง Windows และ macOS
- 6 ชั่วโมง = 2 ช่วง × 3 ชม. · เตรียมเครื่องก่อนวันอบรม (~45 นาที)
- ทุกอย่างทำใน **VS Code** (Python + Jupyter extension) · Python 3.12 จาก python.org + `venv` + `pip` (ไม่ใช้ Anaconda — ให้ตรงกับวิธีที่ Render ติดตั้ง)
- ผู้เรียน **fork** repo นี้ → ทำงานใน repo ของตัวเอง → Render deploy จาก fork

## เป้าหมายการเรียนรู้

1. ตั้งค่าโปรเจกต์ ML ที่ทำซ้ำได้ (venv, pinned requirements, Git/GitHub)
2. สร้าง classifier ด้วย pipeline ที่ไม่ leak และประเมินด้วย metric ที่เหมาะกับงานคัดกรอง (recall/precision/ROC-AUC/PR-AUC)
3. เลือก threshold จาก out-of-fold probability ตามต้นทุนของความผิดพลาด — ไม่ใช้ test set
4. เทียบ logistic regression กับ Random Forest อย่างเป็นธรรมด้วย CV และ tune ด้วย GridSearchCV
5. อธิบาย model ด้วย SHAP ทั้ง global และ local และรู้ข้อจำกัด
6. แปลง notebook เป็น script + app + test แล้ว deploy บน Render

## Modules

| Module | หน้าอ่าน | Lab | เวลา | เนื้อหาหลัก |
|---|---|---|---|---|
| S | `setup/windows`, `setup/macos`, `setup/github` | `00-check-setup` | ก่อนเรียน | winget / installer, PATH, PowerShell execution policy, CLT, fork/clone/venv, commit & push |
| M1 | `m1-classification` | Lab 1 | 60 นาที | 0 = missing, stratified split, Dummy baseline, pipeline, LR, confusion matrix, ROC-AUC, threshold (OOF, recall ≥ 0.80), odds ratio |
| M2 | `m2-models` | Lab 2 | 60 นาที | tree depth vs overfit, RF, `cross_validate`, `GridSearchCV`, impurity vs permutation importance, test ครั้งเดียว, joblib + metadata |
| M3 | `m3-shap` | Lab 3 | 50 นาที | TreeExplainer, additivity, bar/beeswarm/scatter/waterfall, LinearExplainer, ข้อจำกัด |
| M4 | `m4-deploy/index`, `m4-deploy/render` | Lab 4 | 60 นาที | train.py / model_utils.py / app.py, sklearn version pin, pytest, GitHub Actions, Render Web Service + Blueprint, auto-deploy, troubleshooting |
| ★ | — | Mini-challenge | 45 นาที | heart disease: ColumnTransformer + OneHotEncoder, CV, SHAP บน one-hot |

## ตัวเลขอ้างอิง (จากเฉลย, random_state = 42)

| | ค่า |
|---|---|
| Diabetes: n / prevalence | 768 / 34.9% · train 614 / test 154 |
| Missing (0 → NaN) | Insulin 374, SkinThickness 227, BloodPressure 35, BMI 11, Glucose 5 |
| LR test ROC-AUC | 0.813 · threshold 0.5 → recall 0.50 · threshold 0.281 → recall 0.83, precision 0.58 |
| CV ROC-AUC (5-fold) | LR 0.843 ± 0.019 · RF 0.834 ± 0.021 · RF tuned 0.841 (max_depth 4, min_samples_leaf 10, sqrt) |
| RF (deployed) | threshold 0.332 · test ROC-AUC 0.801, recall 0.80, precision 0.56 · model 0.18 MB |
| SHAP | base value 0.346 · อันดับ: Glucose, BMI, Age (RF) — LR เห็น Glucose, BMI ตรงกัน |
| Heart: CV ROC-AUC | LR 0.907 · RF 0.900 |

ถ้า build ใหม่แล้วตัวเลขเปลี่ยน (เช่นอัปเดต scikit-learn) ต้องแก้ตัวเลขในหน้า `course/m1-classification`, `m2-models`, `m3-shap`, `m4-deploy/index` ด้วย

## การตัดสินใจหลัก

- **Dataset:** Pima diabetes เป็นโจทย์หลัก (feature เข้าใจง่ายสำหรับ app, มี "0 = missing" ให้สอนเรื่องความหมายของข้อมูล, คลาสไม่สมดุล) · UCI Heart Disease เป็น challenge (เพิ่ม categorical feature)
- **Model ที่ deploy:** Random Forest — แม้แม่นเท่า LR — เพื่อฝึก TreeSHAP; หน้า M2 บอกตรง ๆ ว่างานจริงควรเลือก model ที่ง่ายกว่าเมื่อแม่นเท่ากัน
- **App:** Streamlit (ไม่ต้องเขียน HTML, แสดงกราฟ SHAP ง่าย) · commit ไฟล์ model (deploy สิ่งที่ test แล้ว) แทน train ตอน build
- **Render:** plan Free, region Singapore, `rootDir: app`, `PYTHON_VERSION=3.12.10`, health check `/_stcore/health`

## Walkthrough บนเว็บ

`tools/build_lab_pages.py` สร้าง `course/walkthrough/<lab>.qmd` จากต้นฉบับ + เฉลยที่รันแล้ว: โค้ด TODO แบบที่ผู้เรียนเห็น, เฉลยในกล่อง *ดูเฉลย* (ซ่อนไว้), ผลลัพธ์/ตาราง/กราฟ, ลิงก์หน้าทฤษฎีใต้ทุกหัวข้อ · ลำดับ build: `build_lab_nb.py --all` → `build_lab_pages.py` → `quarto render course`

⚠️ เว็บเป็น public — ผู้เรียนเห็นเฉลยได้ตั้งแต่ก่อนคลาส (ตั้งใจให้ใช้ทบทวน)

## สิ่งที่ยังไม่ได้ทำ / ไอเดียต่อ

- วิดีโอประกอบแต่ละ module (ใช้ pipeline แบบ INT3 ได้)
- English mirror ของเว็บคอร์ส
- dry-run การเตรียมเครื่องบน Windows จริง

**ทดสอบ deploy จริงแล้ว (2026-11-23):** Render Web Service จาก repo นี้ build+deploy 1m44s, หน้าแรกโหลดราว 45s บน plan Free, ผลทำนายตรงกับเครื่อง (0.390 สำหรับค่า default) — หน้า Lab 4 เขียนตาม UI ของ Render ณ วันนั้น
