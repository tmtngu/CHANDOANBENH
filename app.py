import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# --- LOAD DATA & TRAIN MODEL NHANH ---
@st.cache_data
def train_diabetes_model():
    # Giả sử file của m tên là diabetes.csv
    df = pd.read_csv("diabetes.csv")
    X = df.drop(columns=['Outcome'])
    y = df['Outcome']
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model, X.columns

model, feature_names = train_diabetes_model()

# --- GIAO DIỆN ---
st.title("🩺 AI CHUẨN ĐOÁN TIỂU ĐƯỜNG")
st.markdown("---")

# Layout chia làm các thanh ngang
st.subheader("📊 Điều chỉnh các chỉ số cơ thể")

# Tạo 6-7 thanh trượt ngang (Sliders)
# T để giá trị mặc định (value) ở mức trung bình an toàn
pregnancies = st.slider("1. Số lần mang thai", 0, 20, 1)
glucose = st.slider("2. Chỉ số Glucose (Đường huyết)", 0, 200, 120)
blood_pressure = st.slider("3. Huyết áp tâm trương (mm Hg)", 0, 130, 70)
skin_thickness = st.slider("4. Độ dày nếp gấp da (mm)", 0, 100, 20)
insulin = st.slider("5. Chỉ số Insulin (mu U/ml)", 0, 900, 80)
bmi = st.slider("6. Chỉ số BMI", 0.0, 70.0, 25.0, step=0.1)
pedigree = st.slider("7. Phả hệ tiểu đường (Pedigree Function)", 0.0, 2.5, 0.5, step=0.01)
age = st.slider("8. Tuổi", 1, 100, 30)

st.markdown("---")

# --- XỬ LÝ DỰ ĐOÁN (REAL-TIME) ---
input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, 
                        insulin, bmi, pedigree, age]])

# Lấy xác suất
prediction = model.predict(input_data)[0]
probability = model.predict_proba(input_data)[0][1] * 100

# --- HIỂN THỊ KẾT QUẢ ---
st.subheader("🎯 Kết quả dự đoán từ AI")

if prediction == 1:
    st.error(f"Cảnh báo: Có khả năng cao bị tiểu đường ({probability:.1f}%)")
    st.markdown(f"""
        <div style="background-color: #ff4b4b; padding: 20px; border-radius: 15px; text-align: center;">
            <h1 style="color: white; margin: 0;">DƯƠNG TÍNH</h1>
            <p style="color: white;">M nên đi kiểm tra tại cơ sở y tế sớm nhé!</p>
        </div>
    """, unsafe_allow_html=True)
else:
    st.success(f"Chúc mừng: Nguy cơ thấp ({probability:.1f}%)")
    st.markdown(f"""
        <div style="background-color: #28a745; padding: 20px; border-radius: 15px; text-align: center;">
            <h1 style="color: white; margin: 0;">ÂM TÍNH</h1>
            <p style="color: white;">Chỉ số hiện tại khá là ổn áp đấy!</p>
        </div>
    """, unsafe_allow_html=True)

st.info("Lưu ý: Kết quả này chỉ mang tính chất tham khảo dựa trên dữ liệu máy học.")