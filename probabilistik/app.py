import pandas as pd
import numpy as np
import streamlit as st
import joblib

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

# Veri setini yükle
df = pd.read_csv('train.csv')

# Tarih bilgisini işleyin
df['date'] = pd.to_datetime(df['date'])
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['day'] = df['date'].dt.day
df['hour'] = df['date'].dt.hour
df['minute'] = df['date'].dt.minute

# Özellik ve hedef değişkenleri ayırın
x = df.drop(['id', 'date', 'Temperature'], axis=1)
y = df[['Temperature']]  # Hedef değişken "Temperature"

# Tüm sütunların sayısal olduğundan emin olun
x = x.select_dtypes(include=[np.number])  # Yalnızca sayısal sütunları seç

# Eğitim ve test setlerine ayır
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=42)

# Sayısal veriler için ön işleyici
preprocessor = StandardScaler()

def time_pred(feature_AA, feature_AB, feature_BA, feature_BB, feature_CA, feature_CB, year, month, day, hour, minute):
    input_data = pd.DataFrame({
        'feature_AA': [feature_AA],
        'feature_AB': [feature_AB],
        'feature_BA': [feature_BA],
        'feature_BB': [feature_BB],
        'feature_CA': [feature_CA],
        'feature_CB': [feature_CB],
        'year': [year],
        'month': [month],
        'day': [day],
        'hour': [hour],
        'minute': [minute]
    })

    input_data_transformed = preprocessor.fit_transform(input_data)

    model = joblib.load('Sıcaklık.pkl')

    prediction = model.predict(input_data_transformed)
    return float(prediction[0])


st.title("Sıcaklık Tahmin Uygulaması")
st.write("Veri Girin")

feature_AA = st.number_input('feature_AA', min_value=-100.0, max_value=100.0, value=0.0, step=0.1)
feature_AB = st.number_input('feature_AB', min_value=-100.0, max_value=100.0, value=0.0, step=0.1)
feature_BA = st.number_input('feature_BA', min_value=-100.0, max_value=100.0, value=0.0, step=0.1)
feature_BB = st.number_input('feature_BB', min_value=-100.0, max_value=100.0, value=0.0, step=0.1)
feature_CA = st.number_input('feature_CA', min_value=-100.0, max_value=100.0, value=0.0, step=0.1)
feature_CB = st.number_input('feature_CB', min_value=-100.0, max_value=100.0, value=0.0, step=0.1)
year = st.number_input('Year', min_value=1900, max_value=2100, value=2024)
month = st.number_input('Month', min_value=1, max_value=12, value=9)
day = st.number_input('Day', min_value=1, max_value=31, value=29)
hour = st.number_input('Hour', min_value=0, max_value=23, value=0)
minute = st.number_input('Minute', min_value=0, max_value=59, value=0)

if st.button('Tahmin Et'):
    time = time_pred(feature_AA, feature_AB, feature_BA, feature_BB, feature_CA, feature_CB, year, month, day, hour, minute)
    st.write(f'Tahmin edilen sıcaklık: {time:.2f} °C')
