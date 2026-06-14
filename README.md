# Predviđanje kvalitete zraka u Hrvatskoj

Završni rad - praktični dio  
Veleučilište Velika Gorica  
Studij: Održavanje računalnih sustava

## O projektu

Sustav za predviđanje koncentracije čestica PM2.5 u zraku 
na području Republike Hrvatske primjenom neuronskih mreža.

Implementirana su tri modela:
- LSTM (Long Short-Term Memory)
- GRU (Gated Recurrent Unit)  
- Hibridni LSTM-GRU model

## Tehnologije

- Python 3.13
- TensorFlow / Keras
- Pandas, NumPy, Scikit-learn
- Matplotlib, Seaborn

## Datoteke u projektu

- **main.py** - glavna skripta koja pokreće cijeli proces
- **config.py** - konfiguracija sustava
- **data_scraper.py** - prikupljanje podataka s API-ja
- **data_preprocessing.py** - obrada i priprema podataka
- **lstm_model.py** - LSTM model
- **gru_model.py** - GRU model
- **hybrid_model.py** - hibridni LSTM-GRU model
- **model_evaluator.py** - evaluacija modela
- **visualizer.py** - generiranje grafova
- **results_viewer.py** - pregled rezultata
- **requirements.txt** - lista Python paketa

## Mape

- **reports/figures/** - grafovi i slike (PNG, PDF)
- **reports/metrics/** - metrike modela (JSON, CSV)

## Pokretanje

Prvo treba instalirati potrebne pakete:

    pip install -r requirements.txt

Zatim pokrenuti glavnu skriptu:

    python main.py

## Modeli

| Model    | Broj parametara |
|----------|-----------------|
| LSTM     | 370.872         |
| GRU      | 281.336         |
| Hibridni | 145.848         |

## Napomena

CSV datoteke s podacima i trenirani modeli (.keras) nisu uključeni 
u repozitorij zbog ograničenja veličine. Generiraju se pokretanjem 
skripte main.py.

## Autor

Ivan Jambrešić  
Veleučilište Velika Gorica, 2026.
