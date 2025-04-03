# Real Estate Price Estimator - Sindian (Taiwan)

Web app per stimare il **prezzo al metro quadro** di immobili nella regione di **Sindian, Nuova Taipei (Taiwan)**, basata sul dataset ufficiale *Real Estate Valuation Data Set*.

---

## Dataset

- Fonte: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/real+estate+valuation+data+set)
- Osservazioni: 414 immobili
- Caratteristiche principali:
  - `latitude`, `longitude`
  - `house_age`
  - `distance_to_MRT`
  - `number_of_convenience_stores`

---

## Modelli Utilizzati

Sono stati sviluppati due modelli regressivi distinti:

1. **Random Forest Regressor (coordinate geografiche)**  
   - Feature: `latitude`, `longitude`  
   - R² Cross-Validation: **0.676**

2. **Random Forest Regressor (caratteristiche immobile)**  
   - Feature: `house_age`, `mrt_distance`, `convenience_stores`  
   - R² Cross-Validation: **0.696**

Il secondo modello ha ottenuto performance leggermente superiori ed è stato selezionato come opzione alternativa nella web app.

---

## Documentazione Tecnica

Tutti i passaggi di sviluppo, tuning e le relative **giustificazioni statistiche e metodologiche** sono documentati nel file:

**`progetto_Nicolo_Bachiorri.ipynb`**

Questo notebook contiene l'intero flusso: esplorazione del dataset, addestramento dei modelli, tuning tramite GridSearchCV e analisi delle performance.

---

## Come eseguire l’app (da terminale)

pip install -r requirements.txt
streamlit run app.py 

---

## Tableau Public 

https://dub01.online.tableau.com/#/site/nbachiorri-b35dff0521/workbooks/2490910?:origin=card_share_link  

