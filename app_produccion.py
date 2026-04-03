import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Predicción depósito bancario", page_icon="🏦", layout="centered")

st.title("Predicción de suscripción de depósito")
st.write(
    "Introduce los datos de un nuevo cliente para que el modelo prediga "
    "si suscribirá el producto bancario."
)

@st.cache_resource
def load_model():
    return joblib.load("modelo_final.joblib")

try:
    model = load_model()
except Exception as e:
    st.error(f"Error al cargar el modelo: {e}")
    st.stop()

st.markdown("### Introduce los valores de las variables")

with st.form("prediction_form"):
    st.subheader("Variables numéricas")
    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("age", min_value=18, max_value=100, value=40)
        balance = st.number_input("balance", value=0)
        day = st.number_input("day", min_value=1, max_value=31, value=15)
        duration = st.number_input("duration", min_value=0, value=180)

    with col2:
        campaign = st.number_input("campaign", min_value=1, value=1)
        pdays = st.number_input("pdays", value=-1)
        previous = st.number_input("previous", min_value=0, value=0)

    st.subheader("Variables categóricas")
    col3, col4 = st.columns(2)

    with col3:
        job = st.selectbox(
            "job",
            [
                "admin.",
                "blue-collar",
                "entrepreneur",
                "housemaid",
                "management",
                "retired",
                "self-employed",
                "services",
                "student",
                "technician",
                "unemployed",
                "unknown",
            ],
        )
        marital = st.selectbox("marital", ["divorced", "married", "single"])
        education = st.selectbox("education", ["primary", "secondary", "tertiary", "unknown"])
        default = st.selectbox("default", ["no", "yes"])
        housing = st.selectbox("housing", ["no", "yes"])

    with col4:
        loan = st.selectbox("loan", ["no", "yes"])
        contact = st.selectbox("contact", ["cellular", "telephone", "unknown"])
        month = st.selectbox(
            "month",
            ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"],
        )
        poutcome = st.selectbox("poutcome", ["failure", "other", "success", "unknown"])

    submitted = st.form_submit_button("Predecir", use_container_width=True)

if submitted:
    X_new = pd.DataFrame(
        {
            "age": [age],
            "job": [job],
            "marital": [marital],
            "education": [education],
            "default": [default],
            "balance": [balance],
            "housing": [housing],
            "loan": [loan],
            "contact": [contact],
            "day": [day],
            "month": [month],
            "duration": [duration],
            "campaign": [campaign],
            "pdays": [pdays],
            "previous": [previous],
            "poutcome": [poutcome],
        }
    )

    # Igual que en entrenamiento
    max_pdays_real = 871   # <-- cambia esto por el valor real de tu notebook
    replacement_pdays = 2 * max_pdays_real

    X_new["npdays"] = (X_new["pdays"] != -1).astype(int)
    X_new["pdays"] = X_new["pdays"].replace(-1, replacement_pdays)

    cat_cols = [
        "job",
        "marital",
        "education",
        "default",
        "housing",
        "loan",
        "contact",
        "month",
        "poutcome",
    ]

    for col in cat_cols:
        X_new[col] = X_new[col].astype("category")

    X_new["age"] = X_new["age"].astype(int)
    X_new["day"] = X_new["day"].astype(int)
    X_new["campaign"] = X_new["campaign"].astype(int)
    X_new["pdays"] = X_new["pdays"].astype(int)
    X_new["previous"] = X_new["previous"].astype(int)
    X_new["npdays"] = X_new["npdays"].astype(int)

    try:
        pred = model.predict(X_new)[0]
        pred_text = "yes" if pred == 1 else "no"

        st.success(f"Predicción: {pred_text}")

        st.markdown("#### Datos enviados al modelo")
        st.dataframe(X_new)

        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X_new)[0]
            st.markdown("#### Probabilidades")
            c1, c2 = st.columns(2)
            c1.metric("No", f"{proba[0]*100:.2f}%")
            c2.metric("Yes", f"{proba[1]*100:.2f}%")

    except Exception as e:
        st.error(f"Error durante la predicción: {e}")
