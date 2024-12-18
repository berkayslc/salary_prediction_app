import streamlit as st
import pandas as pd
import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler

# 1. Model ve Dönüşüm Araçlarını Yükleme
model = joblib.load('models/gradient_boosting_model.pkl')
label_encoders = joblib.load('encoders/label_encoders_gb.pkl')
scaler = joblib.load('scalers/scaler_gb.pkl')

# Streamlit Uygulaması Başlığı
st.title("Maaş Tahmin Uygulaması")
st.write("Gradient Boosting modeli ile maaş tahmini yapın.")

# Kullanıcıdan Girdi Alınacak Alanlar
st.header("Bilgilerinizi Girin:")
position = st.selectbox("Pozisyon", ['Backend Developer', 'Frontend Developer', 'Fullstack Developer', 'Data Scientist'])
level = st.selectbox("Seviye", ['Junior', 'Mid', 'Senior'])
experience = st.selectbox("Tecrübe", ['0-1 Yıl', '1-3 Yıl', '4-6 Yıl', '7+ Yıl'])
technology = st.text_input("Teknoloji", "Python, Django")
location = st.selectbox("Lokasyon", ['Türkiye', 'Yurt Dışı'])
way_of_working = st.selectbox("Çalışma Şekli", ['Remote', 'Hybrid', 'Yerinde / Ofiste'])
employees_number = st.selectbox("Çalışan Sayısı", ['0-50', '50-100', '100-300', '300-1000', '2000+'])
salary_type = st.selectbox("Maaş Türü", ['Türk Lirası', 'USD', 'EUR'])

# Kullanıcı Verilerini İşleme
def process_input(position, level, experience, technology, location, way_of_working, employees_number, salary_type):
    input_data = pd.DataFrame({
        'Position': [position],
        'Level': [level],
        'Experience': [experience],
        'Technology': [technology],
        'Location': [location],
        'Way_of_working': [way_of_working],
        'Employees_number': [employees_number],
        'Salary_type': [salary_type]
    })

    # Kategorik verileri LabelEncoder ile dönüştürme (unseen değerleri ele alarak)
    for column in input_data.columns:
        if column in label_encoders:
            le = label_encoders[column]
            input_data[column] = input_data[column].apply(
                lambda x: x if x in le.classes_ else 'unknown'
            )
            if 'unknown' not in le.classes_:
                le.classes_ = np.append(le.classes_, 'unknown')
            input_data[column] = le.transform(input_data[column])

    # Özellikleri ölçeklendirme
    input_data_scaled = scaler.transform(input_data)
    return input_data_scaled

# Tahmin Butonu
if st.button("Maaşı Tahmin Et"):
    # Giriş verilerini işleme
    input_data_scaled = process_input(position, level, experience, technology, location, way_of_working, employees_number, salary_type)

    # Model ile tahmin yapma
    predicted_salary = model.predict(input_data_scaled)
    
    # Tahmin Sonucunu Gösterme
    st.success(f"Tahmini Maaş - 2022: {predicted_salary[0]:,.2f} TL")
    st.success(f"Tahmini Maaş - 2024: {predicted_salary[0]*1.87:,.2f} TL")
