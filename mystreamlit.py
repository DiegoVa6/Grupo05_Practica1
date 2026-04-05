import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="Predicción Depósito Bancario",
    page_icon="🏦",
    layout="wide"
)

@st.cache_resource
def load_model():
    return joblib.load("modelo_final.joblib")

model = load_model()

st.title("🏦 Predicción de suscripción de depósito bancario")
st.markdown("Introduce los datos del cliente a la izquierda y pulsa **Predecir**.")
st.divider()

col_inputs, col_result = st.columns([1.2, 1])

# ── COLUMNA IZQUIERDA: inputs ──────────────────────────────────────────
with col_inputs:
    st.subheader("Datos del cliente")

    c1, c2 = st.columns(2)

    with c1:
        st.markdown("**Numéricas**")
        age      = st.number_input("age",      min_value=18, max_value=100, value=40)
        balance  = st.number_input("balance",  value=0)
        day      = st.number_input("day",      min_value=1, max_value=31, value=15)
        duration = st.number_input("duration", min_value=0, value=180)

    with c2:
        st.markdown("**Numéricas (cont.)**")
        campaign = st.number_input("campaign", min_value=1, value=1)
        pdays    = st.number_input("pdays",    value=-1)
        previous = st.number_input("previous", min_value=0, value=0)

    st.markdown("**Categóricas**")
    c3, c4 = st.columns(2)

    with c3:
        job       = st.selectbox("job",       ["admin.","blue-collar","entrepreneur",
                                               "housemaid","management","retired",
                                               "self-employed","services","student",
                                               "technician","unemployed","unknown"])
        marital   = st.selectbox("marital",   ["divorced","married","single"])
        education = st.selectbox("education", ["primary","secondary","tertiary","unknown"])
        default   = st.selectbox("default",   ["no","yes"])
        housing   = st.selectbox("housing",   ["no","yes"])

    with c4:
        loan     = st.selectbox("loan",     ["no","yes"])
        contact  = st.selectbox("contact",  ["cellular","telephone","unknown"])
        month    = st.selectbox("month",    ["jan","feb","mar","apr","may","jun",
                                             "jul","aug","sep","oct","nov","dec"])
        poutcome = st.selectbox("poutcome", ["failure","other","success","unknown"])

    predecir = st.button("🔮 Predecir", use_container_width=True, type="primary")

# ── COLUMNA DERECHA: resultado ─────────────────────────────────────────
with col_result:
    st.subheader("Resultado")

    if predecir:
        # Construir DataFrame
        X_new = pd.DataFrame([{
            "age": age, "job": job, "marital": marital, "education": education,
            "default": default, "balance": balance, "housing": housing,
            "loan": loan, "contact": contact, "day": day, "month": month,
            "duration": duration, "campaign": campaign, "pdays": pdays,
            "previous": previous, "poutcome": poutcome
        }])

        # Preprocesado igual que en entrenamiento
        X_new["npdays"] = (X_new["pdays"] != -1).astype(int)
        X_new["pdays"]  = X_new["pdays"].replace(-1, 1708)

        cat_cols = ["job","marital","education","default","housing",
                    "loan","contact","month","poutcome"]
        for col in cat_cols:
            X_new[col] = X_new[col].astype("category")

        pred  = model.predict(X_new)[0]
        proba = model.predict_proba(X_new)[0]

        # Resultado principal
        if pred == 1:
            st.success("✅ El cliente **SUSCRIBIRÁ** el depósito")
        else:
            st.error("❌ El cliente **NO** suscribirá el depósito")

        st.divider()

        # Probabilidades
        m1, m2 = st.columns(2)
        m1.metric("P(no)",  f"{proba[0]*100:.1f}%")
        m2.metric("P(yes)", f"{proba[1]*100:.1f}%")

        st.markdown("**Probabilidad de suscripción**")
        st.progress(float(proba[1]))

        st.divider()

        # Resumen de datos enviados
        st.markdown("**Resumen del cliente**")
        st.dataframe(
            X_new[["age","job","marital","education","balance",
                   "housing","loan","contact","month","duration",
                   "campaign","pdays","previous","poutcome","npdays"]]
            .T.rename(columns={0: "Valor"}),
            use_container_width=True
        )

    else:
        st.info("👈 Introduce los datos del cliente y pulsa **Predecir**.")