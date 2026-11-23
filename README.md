# ML in Action: Classification → SHAP → Deploy

สื่อการสอน machine learning แบบลงมือทำ: **คัดกรองโรคเบาหวาน** ด้วย Logistic Regression และ Random Forest → อธิบายด้วย **SHAP** → ห่อเป็น **Streamlit web app** → deploy บน **Render** — เริ่มตั้งแต่เตรียมเครื่อง Windows / macOS, VS Code และ GitHub

> ⚠️ เพื่อการเรียนรู้เท่านั้น — model และ app ไม่ใช่เครื่องมือทางการแพทย์

## สำหรับผู้เรียน

1. เตรียมเครื่อง: `course/setup/windows.qmd` หรือ `course/setup/macos.qmd` (หรืออ่านบนเว็บคอร์ส)
2. **Fork** repo นี้ → clone ด้วย VS Code → สร้าง `.venv` → `pip install -r requirements.txt` (`course/setup/github.qmd`)
3. เปิด `labs/00-check-setup.ipynb` → Run All → ✅ ทุกบรรทัด
4. ทำ lab ตามลำดับใน `labs/` · เฉลยอยู่ใน `labs/solution/`
5. Deploy `app/` บน Render (`course/m4-deploy/render.qmd`)

| | หัวข้อ | ไฟล์ |
|---|---|---|
| S | เตรียมเครื่อง: Python 3.12, VS Code, Git, GitHub, venv | `course/setup/` · `labs/00-check-setup.ipynb` |
| M1 | Classification: pipeline, logistic regression, metrics, threshold | `labs/lab1-logistic.ipynb` |
| M2 | Decision tree → Random Forest, CV, GridSearchCV, importance | `labs/lab2-random-forest.ipynb` |
| M3 | SHAP: beeswarm, dependence, waterfall, TreeExplainer vs LinearExplainer | `labs/lab3-shap.ipynb` |
| M4 | notebook → `train.py` → Streamlit app → pytest → Render | `app/` · `render.yaml` |
| ★ | Mini-challenge: โรคหัวใจ (ColumnTransformer, one-hot, SHAP) | `labs/lab5-challenge-heart.ipynb` |

## โครงสร้าง

```text
labs/                     notebook ของผู้เรียน (มี TODO) + data/ + mlia.py (✅/❌, checkpoint)
  src/                    ต้นฉบับ lab (percent format) → tools/build_lab_nb.py สร้าง notebook ผู้เรียน + เฉลย
  solution/               เฉลย รันแล้ว มีผลลัพธ์ครบ
  checkpoints/            ตัวแปรของแต่ละหัวข้อ สำหรับคนที่ตามไม่ทัน
app/                      web app ที่ deploy: train.py · model_utils.py · app.py · model/ · tests/
course/                   เว็บคอร์ส (Quarto book, ภาษาไทย)
docs/curriculum.md        การออกแบบหลักสูตร (สำหรับผู้สอน)
tools/                    fetch_data.py · build_lab_nb.py · make_check_setup.py
render.yaml               Render Blueprint
.github/workflows/        รัน app/tests ทุกครั้งที่ push
```

## สำหรับผู้สอน

```bash
python tools/fetch_data.py                # ดาวน์โหลด dataset → labs/data/
```

```bash
python tools/build_lab_nb.py --all        # labs/src/*.py → labs/*.ipynb + labs/solution/*.ipynb + checkpoints
```

```bash
python tools/make_check_setup.py          # labs/00-check-setup.ipynb
```

```bash
python app/train.py && python -m pytest app/tests -q
```

```bash
quarto render course                      # เว็บคอร์ส → course/_book/
```

แก้ lab ที่ `labs/src/<lab>.py` เท่านั้น แล้ว build ใหม่ — อย่าแก้ `.ipynb` ตรง ๆ · รายละเอียดการออกแบบที่ [docs/curriculum.md](docs/curriculum.md)

## Data

- **Pima Indians Diabetes Database** — Smith et al. (1988), NIDDK, via UCI Machine Learning Repository
- **Heart Disease (Cleveland)** — Janosi, Steinbrunn, Pfisterer & Detrano (1989), UCI Machine Learning Repository, doi:10.24432/C52P4X, CC BY 4.0

ผู้สอน: ผศ.ดร.รัตน์ชัยนันท์ ธรรมสุจริต
