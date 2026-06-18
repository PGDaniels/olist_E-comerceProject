import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ── Page config ──────────────────────────────────────────────
st.set_page_config(page_title='Olist Satisfaction Predictor', page_icon='🛒', layout='wide')

# ── Load model and scaler ────────────────────────────────────
@st.cache_resource
def load_model():
    model  = joblib.load('olist_model.pkl')
    scaler = joblib.load('olist_scaler.pkl')
    return model, scaler

model, scaler = load_model()

# ── Title ────────────────────────────────────────────────────
st.title('🛒 Olist Customer Satisfaction Predictor')
st.markdown('Fill in the order details on the left and click **Predict** to see the result.')
st.divider()

# ── Sidebar inputs ───────────────────────────────────────────
st.sidebar.header('📋 Enter Order Details')

delivery_days        = st.sidebar.slider('Delivery Days', 1, 60, 10)
shipping_delay       = st.sidebar.slider('Shipping Delay (days)', -30, 30, 0)
price                = st.sidebar.number_input('Product Price (R$)', min_value=1.0, value=50.0)
freight_value        = st.sidebar.number_input('Freight Value (R$)', min_value=0.0, value=15.0)
total_payment        = st.sidebar.number_input('Total Payment (R$)', min_value=1.0, value=65.0)
payment_installments = st.sidebar.slider('Installments', 1, 24, 1)

predict_btn = st.sidebar.button('🔍 Predict Satisfaction', use_container_width=True)

# ── Main page default state ──────────────────────────────────
col1, col2, col3 = st.columns(3)

col1.metric('Delivery Days',   f'{delivery_days} days')
col2.metric('Shipping Delay',  f'{shipping_delay} days')
col3.metric('Total Payment',   f'R$ {total_payment:.2f}')

st.divider()

# ── Prediction ───────────────────────────────────────────────
if predict_btn:
    input_data = pd.DataFrame([[
        delivery_days, shipping_delay, price, freight_value,
        total_payment, payment_installments
    ]], columns=[
        'delivery_days', 'shipping_delay', 'price', 'freight_value',
        'total_payment', 'payment_installments'
    ])

    input_scaled = scaler.transform(input_data)
    prob         = model.predict_proba(input_scaled)[0][1]
    pred         = 'SATISFIED' if prob >= 0.5 else 'NOT SATISFIED'

    st.subheader('Prediction Result')

    if prob >= 0.5:
        st.success(f'✅  Prediction: {pred}')
    else:
        st.error(f'❌  Prediction: {pred}')

    col_a, col_b = st.columns(2)
    col_a.metric('Probability of Satisfaction', f'{prob*100:.1f}%')
    col_b.metric('Probability of Dissatisfaction', f'{(1-prob)*100:.1f}%')

    st.progress(prob)

    st.divider()
    st.markdown('### 📊 What This Means')
    if prob >= 0.5:
        st.info(f'Based on the order details entered, there is a **{prob*100:.1f}% probability** that this customer will be satisfied with their purchase experience.')
    else:
        st.warning(f'Based on the order details entered, there is only a **{prob*100:.1f}% probability** of satisfaction. The customer may be unhappy — likely due to long delivery time or shipping delays.')

else:
    st.info('👈  Adjust the sliders on the left and click **Predict Satisfaction** to get a result.')