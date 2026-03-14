import streamlit as st
import pandas as pd
import numpy as np
import joblib
#EMBELEZADO POR GEMINI

# Configuração da página (deve ser o primeiro comando Streamlit)
st.set_page_config(page_title="Preditor de Saúde", page_icon="🏥", layout="centered")

# Carregar o modelo
@st.cache_resource
def load_model():
    return joblib.load("models/modelo_diabetes.pkl")

modelo = load_model()

# Estilização Personalizada (CSS)
st.markdown("""
    <style>
    .main {
        background-color: #f5f7f9;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #007bff;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)

# Cabeçalho com ícone
st.title("🏥 Diagnóstico Assistido: Diabetes")
st.markdown("""
Esta ferramenta utiliza inteligência artificial para avaliar o risco de diabetes com base em dados clínicos. 
*Preencha os campos abaixo para obter a análise.*
""")

st.divider()

# Organizando as entradas em colunas para não ficar uma lista vertical infinita
col1, col2 = st.columns(2)

with col1:
    st.subheader("📊 Dados Biométricos")
    pregnancies = st.number_input("Gestações", min_value=0, max_value=20, value=0)
    glucose = st.number_input("Glicose (mg/dL)", min_value=0.0, max_value=300.0, value=100.0)
    blood_pressure = st.number_input("Pressão Arterial (mm Hg)", min_value=0.0, max_value=200.0, value=70.0)
    skin_thickness = st.number_input("Espessura da Pele (mm)", min_value=0.0, max_value=100.0, value=20.0)

with col2:
    st.subheader("🧬 Indicadores de Saúde")
    insulin = st.number_input("Insulina (mu U/ml)", min_value=0.0, max_value=900.0, value=80.0)
    bmi = st.number_input("IMC (Índice de Massa Corporal)", min_value=0.0, max_value=70.0, value=25.0, step=0.1)
    dpf = st.number_input("Função Histórico Familiar", min_value=0.0, max_value=3.0, value=0.5, step=0.01)
    age = st.number_input("Idade", min_value=0, max_value=120, value=30)

st.divider()

# Botão centralizado
if st.button("🚀 Realizar Análise"):
    # Organizar dados para o modelo
    dados_entrada = pd.DataFrame([[
        pregnancies, glucose, blood_pressure, skin_thickness, 
        insulin, bmi, dpf, age
    ]], columns=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'])
    
    # Predição
    pred = modelo.predict(dados_entrada)[0]
    prob = modelo.predict_proba(dados_entrada)[0] if hasattr(modelo, "predict_proba") else None
    
    # Exibição do Resultado
    st.subheader("📋 Resultado da Avaliação")
    
    if pred == 1:
        st.error("### ⚠️ Risco de Diabetes Detectado")
        st.markdown("A análise indica uma alta probabilidade de diabetes. Recomendamos a consulta com um profissional de saúde.")
    else:
        st.success("### ✅ Baixo Risco Detectado")
        st.markdown("A análise não identificou padrões de diabetes nos dados fornecidos.")

    # Exibir métricas de probabilidade em colunas
    if prob is not None:
        c1, c2 = st.columns(2)
        c1.metric("Probabilidade Negativa", f"{prob[0]*100:.1f}%")
        c2.metric("Probabilidade Positiva", f"{prob[1]*100:.1f}%")
        
        # Barra de progresso visual
        st.progress(float(prob[1]))

st.sidebar.info("Atenção: Este é um modelo preditivo para fins educacionais e não substitui exames laboratoriais.")