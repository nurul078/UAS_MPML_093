import streamlit as st
import pandas as pd
import joblib

# Judul aplikasi
st.title("🎯 Prediksi Pemesanan Makanan Online")
st.markdown("""
Aplikasi ini memprediksi apakah pelanggan akan memesan makanan online.
""")

# Load model
@st.cache_resource
def load_model():
    return joblib.load('model.pkl')

model = load_model()

# Input pengguna
def user_input():
    st.sidebar.header('📋 Data Pelanggan')
    age = st.sidebar.number_input('Usia', 10, 100, 30)
    gender = st.sidebar.selectbox('Jenis Kelamin', ['Female', 'Male'])
    marital = st.sidebar.selectbox('Status Pernikahan', ['Single', 'Married'])
    income = st.sidebar.selectbox('Pendapatan', ['No Income', 'Below Rs.10000', 'Above Rs.10000'])
    return pd.DataFrame({
        'Age': [age],
        'Gender': [gender],
        'Marital Status': [marital],
        'Monthly Income': [income]
    })

# Tampilkan input
input_data = user_input()
st.subheader('Data Input')
st.write(input_data)

# Prediksi
if st.button('🔮 Prediksi'):
    try:
        # Pastikan format input sama dengan data training
        processed = input_data.copy()
        processed['Gender'] = processed['Gender'].map({'Female': 0, 'Male': 1})
        processed['Marital Status'] = processed['Marital Status'].map({'Single': 0, 'Married': 1})
        
        prediction = model.predict(processed)
        st.success(f"Hasil: {'Akan Memesan' if prediction[0] else 'Tidak Akan Memesan'}")
    except Exception as e:
        st.error(f"Error: {str(e)}")