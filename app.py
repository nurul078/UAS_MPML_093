import streamlit as st
import pandas as pd
import pickle

# Load model dan preprocessing
@st.cache_resource
def load_model():
    model = pickle.load(open('best_regression_model.pkl', 'rb'))
    preprocessor = pickle.load(open('preprocessing_pipeline.pkl', 'rb'))
    return model, preprocessor

model, preprocessor = load_model()

# Judul Aplikasi
st.title("🎯 Prediksi Pemesanan Makanan Online")
st.write("Aplikasi ini memprediksi apakah customer akan memesan makanan online atau tidak")

# Input Form
with st.form("input_form"):
    st.header("Data Customer")
    
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.number_input("Umur", min_value=10, max_value=100, value=25)
        gender = st.selectbox("Jenis Kelamin", ["Male", "Female"])
        marital_status = st.selectbox("Status Pernikahan", ["Single", "Married", "Prefer not to say"])
        occupation = st.selectbox("Pekerjaan", ["Student", "Employee", "Self Employeed", "House wife"])
    
    with col2:
        monthly_income = st.selectbox("Pendapatan Bulanan", 
                                    ["No Income", "Below Rs.10000", "10001 to 25000", 
                                     "25001 to 50000", "More than 50000"])
        education = st.selectbox("Pendidikan", 
                               ["Graduate", "Post Graduate", "Ph.D", "School", "Uneducated"])
        family_size = st.number_input("Jumlah Anggota Keluarga", min_value=1, max_value=10, value=3)
    
    latitude = st.number_input("Latitude", value=12.9766)
    longitude = st.number_input("Longitude", value=77.5993)
    pin_code = st.number_input("Kode Pos", min_value=100000, max_value=999999, value=560001)
    
    submitted = st.form_submit_button("Prediksi Sekarang!")

# Prediksi saat tombol ditekan
if submitted:
    # Buat DataFrame dari input
    input_data = pd.DataFrame({
        'Age': [age],
        'Gender': [gender],
        'Marital Status': [marital_status],
        'Occupation': [occupation],
        'Monthly Income': [monthly_income],
        'Educational Qualifications': [education],
        'Family size': [family_size],
        'latitude': [latitude],
        'longitude': [longitude],
        'Pin code': [pin_code]
    })
    
    # Preprocess data
    processed_data = preprocessor.transform(input_data)
    
    # Prediksi
    prediction = model.predict(processed_data)
    proba = model.predict_proba(processed_data)[0]
    
    # Tampilkan hasil
    st.subheader("Hasil Prediksi")
    if prediction[0] == 1:
        st.success(f"✅ Customer akan memesan (Probabilitas: {proba[1]:.2%})")
    else:
        st.error(f"❌ Customer tidak akan memesan (Probabilitas: {proba[0]:.2%})")
    
    # Tampilkan feature importance (jika model punya)
    if hasattr(model, 'coef_'):
        st.subheader("Faktor yang Mempengaruhi")
        coef_df = pd.DataFrame({
            'Feature': preprocessor.get_feature_names_out(),
            'Importance': model.coef_[0]
        }).sort_values('Importance', ascending=False)
        st.bar_chart(coef_df.set_index('Feature'))