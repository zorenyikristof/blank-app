
import streamlit as st
from prophet.serialize import model_from_json
import pandas as pd
import json
import matplotlib.pyplot as plt

st.title("📈 Prophet Forecast Dashboard")

# Modell laden
model_file = st.file_uploader("Upload Prophet-Model-file (.json)", type="json")

if model_file is not None:
    with model_file:
        model = model_from_json(json.load(model_file))
        st.success("✅ Model uploaded!")

    # Zeitfenster auswählen
    periods = st.slider("Hours to Forecast?", 1, 2160, 168)
    
    # Zukunft erzeugen & vorhersagen
    future = model.make_future_dataframe(periods=periods, freq='h')
    forecast = model.predict(future)

    # Ergebnisse anzeigen
    st.subheader("Forecast-Data (yhat):")
    st.dataframe(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(24))

    # Plot anzeigen
    st.subheader("📊 Prognosis-Plot")
    fig = model.plot(forecast)
    st.pyplot(fig)
