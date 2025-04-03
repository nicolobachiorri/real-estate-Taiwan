import streamlit as st
import pickle
import numpy as np

#  Caricamento modelli 
with open('model_lat_lon_rf.pkl', 'rb') as f:
    model_geo = pickle.load(f)

with open('model_extra_features_optimized.pkl', 'rb') as f:
    model_feat = pickle.load(f)

#  Interfaccia utente
st.title("Stima del Prezzo al Metro Quadro")
st.markdown("Predizione del prezzo degli immobili nella regione di Sindian (Nuova Taipei, Taiwan).")

# Selettore modalità 
mode = st.radio(
    "Scegli la modalità di inserimento:",
    ["Coordinate geografiche", "Caratteristiche dell'immobile"]
)

# Modalità 1: Latitudine e Longitudine
if mode == "Coordinate geografiche":
    st.subheader("Inserisci Latitudine e Longitudine")

    # Limiti validi (dataset)
    LAT_MIN, LAT_MAX = 24.93, 25.08
    LON_MIN, LON_MAX = 121.47, 121.56

    lat = st.number_input("Latitudine", min_value=LAT_MIN, max_value=LAT_MAX, format="%.6f")
    lon = st.number_input("Longitudine", min_value=LON_MIN, max_value=LON_MAX, format="%.6f")

    if st.button("Stima il Prezzo", key="geo"):
        input_geo = np.array([[lat, lon]])
        prediction = model_geo.predict(input_geo)[0]
        st.success(f" Prezzo stimato: {prediction:.2f} NT$/m²")

# Modalità 2: Caratteristiche dell'immobile 
elif mode == "Caratteristiche dell'immobile":
    st.subheader("Inserisci le caratteristiche dell'immobile")

    house_age = st.number_input("Età dell'immobile (anni)", min_value=0.0, max_value=100.0, step=0.5)
    mrt_distance = st.number_input("Distanza dalla stazione MRT (metri)", min_value=0.0, step=10.0)
    convenience_stores = st.number_input("Numero di minimarket nelle vicinanze", min_value=0, step=1)

    if st.button("Stima il Prezzo", key="features"):
        input_feat = np.array([[house_age, mrt_distance, convenience_stores]])
        prediction = model_feat.predict(input_feat)[0]
        st.success(f"Prezzo stimato: {prediction:.2f} NT$/m²")
