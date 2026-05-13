import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import Perceptron
from sklearn.preprocessing import StandardScaler

# --- CẤU HÌNH TRANG ---
st.set_page_config(page_title="Diabetes Perceptron", layout="wide")

# --- LOAD DATA & TRAIN PERCEPTRON ---
@st.cache_resource
def train_perceptron():
    df = pd.read_csv('diabetes.csv')
    
    # Logic của m: Thay 0 bằng mean
    cols_fix = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    for col in cols_fix:
        df[col] = df[col].replace(0, df[col].mean())
    
    X = df.drop('Outcome', axis=1)
    y = df['Outcome']
    
    # Perceptron BẮT BUỘC phải Scale dữ liệu thì mới hội tụ được
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Khởi tạo và huấn luyện Perceptron
    model = Perceptron(max_iter=1000, tol=1e-3, random_state=42)
    model.fit(X_scaled, y)
    
    return model, scaler

model, scaler = train_perceptron()

# --- GIAO DIỆN ---
st.markdown("<h2 style='text-align: center; color: #4A90E2;'>🧠 PERCEPTRON DIABETES PREDICTOR</h2>", unsafe_allow_html=True)
st.write("---")

# Layout: 8 thanh ngang bên trái, 1 kết quả bên phải
left_col, right_col = st.columns([3, 2], gap="large")

with left_col:
    st.markdown("### 🛠️ Điều chỉnh 8 chỉ số")
    preg = st.slider("1. Pregnancies", 0, 20, 1)
    glu = st.slider("2. Glucose", 0, 200, 117)
    bp = st.slider("3. Blood Pressure", 0, 130, 72)
    skin = st.slider("4. Skin Thickness", 0, 100, 23)
    ins = st.slider("5. Insulin", 0, 900, 30)
    bmi = st.slider("6. BMI", 0.0, 70.0, 32.0, step=0.1)
    dpf = st.slider("7. Pedigree Function", 0.0, 2.5, 0.5, step=0.01)
    age = st.slider("8. Age", 1, 100, 30)

with right_col:
    st.markdown("### 🎯 Kết quả đầu ra")
    
    # 1. Gom dữ liệu
    raw_input = np.array([[preg, glu, bp, skin, ins, bmi, dpf, age]])
    
    # 2. Scale dữ liệu đầu vào (Cực kỳ quan trọng với Perceptron)
    scaled_input = scaler.transform(raw_input)
    
    # 3. Dự đoán
    # Perceptron không có predict_proba (xác suất) mặc định như RF, 
    # nó dùng hàm quyết định (decision_function)
    prediction = model.predict(scaled_input)[0]
    score = model.decision_function(scaled_input)[0]
    
    st.write("")
    if prediction == 1:
        st.markdown(f"""
            <div style="background: linear-gradient(135deg, #FF4B2B 0%, #FF416C 100%); padding: 50px; border-radius: 25px; text-align: center; box-shadow: 0 10px 30px rgba(255, 75, 75, 0.3);">
                <h2 style="color: white; margin: 0; font-size: 20px;">AI PHÂN LOẠI:</h2>
                <h1 style="color: white; font-size: 60px; margin: 20px 0;">CÓ BỆNH</h1>
                <p style="color: white; opacity: 0.8;">(Trạng thái: Dương tính)</p>
            </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
            <div style="background: linear-gradient(135deg, #1D976C 0%, #93F9B9 100%); padding: 50px; border-radius: 25px; text-align: center; box-shadow: 0 10px 30px rgba(29, 151, 108, 0.3);">
                <h2 style="color: white; margin: 0; font-size: 20px;">AI PHÂN LOẠI:</h2>
                <h1 style="color: white; font-size: 60px; margin: 20px 0;">KHÔNG BỆNH</h1>
                <p style="color: white; opacity: 0.8;">(Trạng thái: Âm tính)</p>
            </div>
        """, unsafe_allow_html=True)

    # Giải thích thêm cho m về cái Perceptron này
    with st.expander("Perceptron hoạt động thế nào?"):
        st.write(f"Giá trị hàm quyết định: **{score:.4f}**")
        st.write("Nếu con số này > 0, Perceptron xếp vào lớp Dương tính (1), ngược lại là Âm tính (0).")