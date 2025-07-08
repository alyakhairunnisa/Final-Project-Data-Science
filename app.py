import streamlit as st
import joblib
import numpy as np
import pandas as pd
import sklearn

# ======================
# 🎯 Function load model & scaler
# ======================

@st.cache_resource
def load_model_scaler():
    try:
        scaler = joblib.load('scaler_attrition.pkl')
        model = joblib.load('model_attrition.pkl')
        return scaler, model
    except Exception as e:
        st.error(
            f"Gagal load model atau scaler: {e}\n"
            "Solusi: Pastikan versi scikit-learn SAAT TRAINING dan SAAT DEPLOY SAMA. "
            "Jika error dtype, retrain dan simpan ulang model/scaler di environment ini."
        )
        return None, None

scaler, model = load_model_scaler()

# ======================
# 🎯 Input fitur sesuai model
# GANTI list ini dengan fitur aktual modelmu
# ======================

st.title("Prediksi Attrition Karyawan")

st.write("Masukkan data karyawan untuk memprediksi apakah akan resign atau tidak.")

# Contoh fitur modelmu (GANTI SESUAI NOTEBOOK)
# Di sini hanya ilustrasi. Harus sesuai urutan X_train.columns saat training

age = st.number_input("Usia", min_value=18, max_value=60, value=30)
income = st.number_input("Pendapatan Bulanan", min_value=1000, max_value=50000, value=5000)
distance = st.number_input("Jarak dari Rumah (km)", min_value=1, max_value=50, value=10)
overtime = st.selectbox("OverTime", ["Yes", "No"])
daily_rate = st.number_input("Daily Rate", min_value=0, max_value=1000, value=100)
business_travel = st.selectbox("Business Travel", ["Travel Frequently", "Travel Rarely", "Non-Traveling"])
department = st.selectbox("Department", ["Research & Development", "Sales", "Human Resources"])

# ======================
# 🎯 Encoding kategori (GANTI SESUAI TRAINING)
# Contoh OverTime: Yes=1, No=0
# ======================

overtime_encoded = 1 if overtime == "Yes" else 0
business_travel_frequently = 1 if business_travel == "Travel Frequently" else 0
business_travel_rarely = 1 if business_travel == "Travel Rarely" else 0
department_rd = 1 if department == "Research & Development" else 0
department_sales = 1 if department == "Sales" else 0

# ======================
# 🎯 Buat dataframe input sesuai urutan training
# Tambahkan semua fitur lain sesuai modelmu
# ======================

# Contoh: jika modelmu pakai 45 fitur
# Pastikan urutannya sama dengan X_train.columns

input_data = pd.DataFrame({
    'Age': [age],
    # 'Attrition': [0],  # default, tidak dipakai prediksi
    'DailyRate': [daily_rate],
    'DistanceFromHome': [distance],
    'Education': [0],  # tambahkan input jika perlu
    'EmployeeNumber': [0],  # default, tidak dipakai prediksi
    'EnvironmentSatisfaction': [0],
    'HourlyRate': [0],
    'JobInvolvement': [0],
    'JobLevel': [0],
    'JobSatisfaction': [0],
    'MonthlyIncome': [income],
    'MonthlyRate': [0],
    'NumCompaniesWorked': [0],
    'PercentSalaryHike': [0],
    'PerformanceRating': [0],
    'RelationshipSatisfaction': [0],
    'StockOptionLevel': [0],
    'TotalWorkingYears': [0],
    'TrainingTimesLastYear': [0],
    'WorkLifeBalance': [0],
    'YearsAtCompany': [0],
    'YearsInCurrentRole': [0],
    'YearsSinceLastPromotion': [0],
    'YearsWithCurrManager': [0],
    # 'Attrition_numerik': [0],  # default
    'BusinessTravel_Travel_Frequently': [business_travel_frequently],
    'BusinessTravel_Travel_Rarely': [business_travel_rarely],
    'Department_Research & Development': [department_rd],
    'Department_Sales': [department_sales],
    'EducationField_Life Sciences': [0],
    'EducationField_Marketing': [0],
    'EducationField_Medical': [0],
    'EducationField_Other': [0],
    'EducationField_Technical Degree': [0],
    'Gender_Male': [0],
    'JobRole_Human Resources': [0],
    'JobRole_Laboratory Technician': [0],
    'JobRole_Manager': [0],
    'JobRole_Manufacturing Director': [0],
    'JobRole_Research Director': [0],
    'JobRole_Research Scientist': [0],
    'JobRole_Sales Executive': [0],
    'JobRole_Sales Representative': [0],
    'MaritalStatus_Married': [0],
    'MaritalStatus_Single': [0],
    'OverTime_Yes': [overtime_encoded],
})

# Jika scaler expect array, gunakan:
try:
    X_scaled = scaler.transform(input_data)
except Exception as e:
    st.error(f"Error saat scaling: {e}")
    X_scaled = None

# ======================
# 🎯 Prediksi
# ======================

if st.button("Prediksi Attrition"):
    if X_scaled is not None:
        try:
            y_pred = model.predict(X_scaled)
            if y_pred[0] == 1:
                st.warning("⚠️ Karyawan diprediksi akan **Resign (Attrition = Yes)**.")
            else:
                st.success("✅ Karyawan diprediksi akan **Bertahan (Attrition = No)**.")
        except Exception as e:
            st.error(f"Error saat prediksi: {e}")
    else:
        st.error("Input tidak valid untuk prediksi.")

# ======================
# 🎯 Catatan implementasi
# ======================
# - Lengkapi semua fitur input sesuai X_train.columns
# - Pastikan encoding sama dengan pipeline training
# - Cek versi scikit-learn sama saat training dan deploy

st.write("Versi scikit-learn yang digunakan:", sklearn.__version__)
