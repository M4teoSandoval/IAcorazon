import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Cargar modelos preentrenados
scaler = joblib.load("scaler.bin")
knn_model = joblib.load("knn_model.bin")

# Título y subtítulo
st.title("Modelo predicción problemas cardiacos con IA")
st.subheader("Realizado por Mateo Sandoval")

# Introducción
st.write(
    "Esta aplicación utiliza un modelo de inteligencia artificial para predecir si una persona "
    "tiene riesgo de sufrir problemas cardíacos, basándose en su edad y nivel de colesterol. "
    "Los resultados pueden ayudar a tomar decisiones preventivas en salud."
)

# Imagen
st.image(
    "https://salud.nih.gov/sites/salud/files/styles/max_1300x1300/public/2022-12/202109-dibujo-corazon-humano.jpg?itok=h82HrDUp", 
    use_column_width=True
)

# Selección de datos de entrada
edad = st.slider("Edad", 20, 80, 40)
colesterol = st.slider("Colesterol (mg/dL)", 100, 600, 200)

# Crear DataFrame con los datos ingresados
data = pd.DataFrame({"Edad": [edad], "Colesterol": [colesterol]})
st.write("### Datos ingresados")
st.dataframe(data)

# Normalizar los datos
data_scaled = scaler.transform(data)

# Realizar la predicción
prediccion = knn_model.predict(data_scaled)[0]

# Mostrar el resultado con colores y emojis
st.markdown("---")

if prediccion == 0:
    st.markdown(
        "<div style='background-color:blue; color:white; padding:20px; text-align:center; border-radius:10px;'>"
        "<h3>No tiene problemas cardiacos 😊</h3>"
        "</div>", unsafe_allow_html=True
    )
else:
    st.markdown(
        "<div style='background-color:red; color:white; padding:20px; text-align:center; border-radius:10px;'>"
        "<h3>Tiene riesgo de problemas cardiacos ⚠️</h3>"
        "</div>", unsafe_allow_html=True
    )

# Línea divisoria
st.markdown("---")

# Copyright
st.markdown("© Unab2025")
