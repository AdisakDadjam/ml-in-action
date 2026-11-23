"""Diabetes screening demo — Streamlit app deployed on Render.

    streamlit run app/app.py          # local, from the repo root → http://localhost:8501
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap
import streamlit as st

from model_utils import explain, load_metadata, predict

st.set_page_config(page_title="Diabetes Screening (demo)", page_icon="🩺", layout="centered")
meta = load_metadata()

st.title("🩺 Diabetes screening — demo")
st.caption("ML in Action · Random Forest + SHAP · เพื่อการเรียนรู้เท่านั้น ไม่ใช่การวินิจฉัยทางการแพทย์")

with st.sidebar:
    st.header("เกี่ยวกับ model")
    m = meta["test_metrics"]
    st.markdown(
        f"- Model: **Random Forest** (scikit-learn {meta['scikit_learn']})\n"
        f"- Threshold: **{meta['threshold']:.2f}** (เลือกให้ recall ≥ {meta['target_recall']:.0%})\n"
        f"- Test ROC-AUC: **{m['roc_auc']:.2f}** · recall **{m['recall']:.2f}** · precision **{m['precision']:.2f}**\n"
        f"- Train {meta['n_train']} / test {meta['n_test']} คน · trained {meta['trained_on']}"
    )
    st.warning("ข้อมูลฝึก: ผู้หญิงเชื้อสาย Pima อายุ ≥ 21 ปี (สหรัฐฯ) — ผลอาจไม่เหมาะกับประชากรกลุ่มอื่น")
    st.caption("Data: Smith et al. (1988), Pima Indians Diabetes Database (NIDDK), via UCI ML Repository")

with st.form("person"):
    st.subheader("ข้อมูลการตรวจสุขภาพ")
    c1, c2 = st.columns(2)
    glucose = c1.number_input("Glucose — น้ำตาล 2 ชม. หลัง OGTT (mg/dL)", 40, 250, 120)
    bmi = c2.number_input("BMI (kg/m²)", 15.0, 70.0, 30.0, step=0.1)
    age = c1.number_input("อายุ (ปี)", 21, 100, 35)
    pregnancies = c2.number_input("จำนวนครั้งที่ตั้งครรภ์", 0, 20, 1)
    bp = c1.number_input("ความดัน diastolic (mmHg)", 30, 130, 72)
    dpf = c2.number_input("Diabetes pedigree function (ประวัติครอบครัว)", 0.05, 2.5, 0.40, step=0.01,
                          help="คะแนนจากประวัติเบาหวานในเครือญาติ ค่าเฉลี่ยในข้อมูลฝึก ≈ 0.47")
    st.caption("ไม่ทราบค่า? เว้นว่างได้ — model จะเติมด้วยค่ากลาง (median) ของข้อมูลฝึก")
    insulin = c1.number_input("Insulin 2 ชม. (µU/mL)", 10, 900, value=None, placeholder="ไม่ทราบ")
    skin = c2.number_input("Triceps skinfold (mm)", 5, 100, value=None, placeholder="ไม่ทราบ")
    submitted = st.form_submit_button("ประเมินความเสี่ยง", type="primary")

if submitted:
    person = {"Pregnancies": pregnancies, "Glucose": glucose, "BloodPressure": bp, "SkinThickness": skin,
              "Insulin": insulin, "BMI": bmi, "DiabetesPedigreeFunction": dpf, "Age": age}
    result = predict(person)

    c1, c2 = st.columns(2)
    c1.metric("ความน่าจะเป็นตาม model", f"{result['probability']:.0%}")
    c2.metric("เกณฑ์ส่งตรวจ (threshold)", f"{result['threshold']:.0%}")
    if result["refer"]:
        st.error("**ควรส่งตรวจยืนยัน** (เช่น HbA1c / OGTT) — ความน่าจะเป็นสูงกว่าเกณฑ์คัดกรอง")
    else:
        st.success("**ต่ำกว่าเกณฑ์คัดกรอง** — ตรวจสุขภาพตามรอบปกติ")

    st.subheader("ทำไม model ถึงตอบแบบนี้ (SHAP)")
    st.caption("เริ่มจากค่าเฉลี่ยของ model (E[f(x)]) · แท่งแดงดันความเสี่ยงขึ้น · แท่งน้ำเงินดันลง · ปลายทางคือค่าที่ model ทำนาย (f(x))")
    plt.figure()
    shap.plots.waterfall(explain(person), show=False)
    st.pyplot(plt.gcf(), clear_figure=True)
    st.caption("SHAP อธิบายว่า *model* ใช้ข้อมูลอย่างไร ไม่ได้บอกเหตุและผลทางการแพทย์")
