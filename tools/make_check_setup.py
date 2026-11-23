"""Write labs/00-check-setup.ipynb:  python tools/make_check_setup.py"""
from pathlib import Path
import nbformat as nbf

OUT = Path(__file__).resolve().parent.parent / "labs" / "00-check-setup.ipynb"
nb = nbf.v4.new_notebook()
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
nb.cells = [
    nbf.v4.new_markdown_cell(
        "# 00 · ตรวจเครื่องก่อนวันอบรม\n\n"
        "1. มุมขวาบนกด **Select Kernel** → **Python Environments** → เลือก `.venv` (Python 3.12)\n"
        "2. กด **Run All**\n"
        "3. ทุกบรรทัดต้องขึ้น ✅ — ถ้ามี ❌ ให้ capture หน้าจอส่งผู้สอน"),
    nbf.v4.new_code_cell(r'''import warnings; warnings.filterwarnings("ignore")
import sys, platform, importlib, re, shutil, subprocess
from pathlib import Path

ok = True
def check(label, good, detail=""):
    global ok
    ok &= bool(good)
    print(("✅ " if good else "❌ ") + label + (f"  ({detail})" if detail else ""))

check("Python 3.12", sys.version_info[:2] == (3, 12), platform.python_version())
check("ใช้ Python จาก .venv", ".venv" in sys.prefix, sys.prefix)

req = {}
rf = Path("../requirements.txt")
if rf.exists():
    for line in rf.read_text().splitlines():
        m = re.match(r"^([A-Za-z0-9_.-]+)==([^\s;#]+)", line)
        if m:
            req[m.group(1).lower()] = m.group(2)
check("พบ requirements.txt (เปิด notebook จากโฟลเดอร์ labs/)", rf.exists())

MODULES = {"numpy": "numpy", "pandas": "pandas", "matplotlib": "matplotlib", "seaborn": "seaborn",
           "scikit-learn": "sklearn", "shap": "shap", "joblib": "joblib", "streamlit": "streamlit"}
for pkg, mod in MODULES.items():
    try:
        v = importlib.import_module(mod).__version__
        want = req.get(pkg)
        check(f"{pkg} {v}", want is None or v == want, "" if want in (None, v) else f"ต้องการ {want}")
    except Exception as e:
        check(pkg, False, f"import ไม่ได้: {type(e).__name__}: {e}")

git = shutil.which("git")
check("git ติดตั้งแล้ว", git, subprocess.run([git, "--version"], capture_output=True, text=True).stdout.strip() if git else "ดูหน้าเตรียมเครื่อง")
if git:
    name = subprocess.run([git, "config", "user.name"], capture_output=True, text=True).stdout.strip()
    check("ตั้ง git user.name แล้ว", name, name or 'git config --global user.name "ชื่อ"')'''),
    nbf.v4.new_code_cell(r'''# smoke test: ลองใช้งานจริงแบบเล็ก ๆ
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, matplotlib.pyplot as plt
try:
    df = pd.read_csv("data/diabetes.csv")
    check("อ่าน data/diabetes.csv", df.shape == (768, 9), f"{df.shape}")
except Exception as e:
    check("อ่าน data/diabetes.csv", False, e)
try:
    from sklearn.ensemble import RandomForestClassifier
    import shap
    X = np.random.default_rng(0).normal(size=(200, 4)); y = (X[:, 0] > 0).astype(int)
    m = RandomForestClassifier(50, random_state=0).fit(X, y)
    sv = shap.TreeExplainer(m)(X[:5])
    check("scikit-learn + SHAP", sv.shape == (5, 4, 2))
except Exception as e:
    check("scikit-learn + SHAP", False, e)
try:
    fig, ax = plt.subplots(figsize=(4, 2)); ax.plot(np.sin(np.linspace(0, 6, 50))); ax.set_title("If you can see this chart, plotting works"); plt.show()
    check("matplotlib: แสดงกราฟ", True)
except Exception as e:
    check("matplotlib: แสดงกราฟ", False, e)

print("\n" + ("🎉 พร้อมเรียนแล้ว!" if ok else "⚠️ ยังมี ❌ — ส่ง screenshot ให้ผู้สอน"))'''),
]
nbf.write(nb, OUT)
print("wrote", OUT)
