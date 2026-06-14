"""
Glavni modul za predvidjanje kvalitete zraka u Republici Hrvatskoj.

Pipeline:
1. Prikupljanje podataka (web scraping)
2. Predobrada podataka (ciscenje, normalizacija, sekvence)
3. Treniranje 3 modela (LSTM, GRU, Hibridni)
4. Evaluacija i usporedba
5. Generiranje prognoza (24h i 7 dana)
6. Vizualizacija rezultata

Autor: Završni rad
Tema: LSTM/GRU neuronske mreže + TensorFlow
"""

import os
import sys
import logging
import numpy as np
import pandas as pd
import tensorflow as tf
from datetime import datetime

# Postavi encoding za Windows konzolu
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

# Random seed za reproducibilnost
np.random.seed(42)
tf.random.set_seed(42)

# ============================================================
# DIREKTORIJI
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')
PREDICTIONS_DIR = os.path.join(DATA_DIR, 'predictions')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
SAVED_MODELS_DIR = os.path.join(MODELS_DIR, 'saved')
CHECKPOINTS_DIR = os.path.join(MODELS_DIR, 'checkpoints')
REPORTS_DIR = os.path.join(BASE_DIR, 'reports')
FIGURES_DIR = os.path.join(REPORTS_DIR, 'figures')
METRICS_DIR = os.path.join(REPORTS_DIR, 'metrics')
LOGS_DIR = os.path.join(BASE_DIR, 'logs')

# Obrisi logs ako je datoteka (ne direktorij)
if os.path.isfile(LOGS_DIR):
    os.remove(LOGS_DIR)
    print("Obrisan stari logs file, kreiram logs direktorij...")

for d in [RAW_DATA_DIR, PROCESSED_DATA_DIR, PREDICTIONS_DIR,
          SAVED_MODELS_DIR, CHECKPOINTS_DIR, FIGURES_DIR,
          METRICS_DIR, LOGS_DIR]:
    os.makedirs(d, exist_ok=True)

# ============================================================
# LOGIRANJE
# ============================================================
log_file = os.path.join(LOGS_DIR, 'main.log')

for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file, encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# ============================================================
# IMPORTI MODULA
# ============================================================
try:
    from config import (
        SHORT_TERM_HOURS, MEDIUM_TERM_HOURS,
        LSTM_CONFIG, GRU_CONFIG, HYBRID_CONFIG
    )
except Exception as e:
    logger.warning(f"config.py: {e} - koristim defaultne vrijednosti")
    SHORT_TERM_HOURS = 24
    MEDIUM_TERM_HOURS = 168
    LSTM_CONFIG = {'lookback_hours': 72}

from data_scraper import AirQualityDataScraper
from data_preprocessing import DataPreprocessor
from lstm_model import LSTMModel
from gru_model import GRUModel
from hybrid_model import HybridModel
from model_evaluator import ModelEvaluator
from visualizer import Visualizer


# ============================================================
# POMOCNE FUNKCIJE
# ============================================================

def print_header():
    """Ispisuje zaglavlje programa."""
    print("")
    print("=" * 65)
    print("  PREDVIDANJE KVALITETE ZRAKA U REPUBLICI HRVATSKOJ")
    print("  LSTM / GRU neuronske mreze + TensorFlow")
    print("=" * 65)
    print(f"  Python verzija    : {sys.version.split()[0]}")
    print(f"  TensorFlow verzija: {tf.__version__}")
    print(f"  NumPy verzija     : {np.__version__}")
    print(f"  Pandas verzija    : {pd.__version__}")

    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"  GPU               : {len(gpus)} GPU(s) detektirano")
    else:
        print("  GPU               : Nije detektiran (CPU mod)")
    print("=" * 65)
    print("")


def get_tensorboard_logdir(model_name):
    """Vraca TensorBoard log direktorij za model."""
    tb_dir = os.path.join(LOGS_DIR, 'tensorboard', model_name)
    os.makedirs(tb_dir, exist_ok=True)
    return tb_dir


# ============================================================
# KORAK 1: PRIKUPLJANJE PODATAKA
# ============================================================

def step1_data_collection():
    """Prikuplja podatke web scrapingom ili generira sinteticke."""
    logger.info("=" * 60)
    logger.info("KORAK 1: PRIKUPLJANJE PODATAKA")
    logger.info("=" * 60)

    scraper = AirQualityDataScraper()
    data = scraper.collect_all_data()

    logger.info("Korak 1 zavrsen.")
    return data


# ============================================================
# KORAK 2: PREDOBRADA PODATAKA
# ============================================================

def step2_preprocessing(target_col='PM2.5', forecast_horizon=24):
    """Predobrada i priprema podataka za model."""
    logger.info("=" * 60)
    logger.info("KORAK 2: PREDOBRADA PODATAKA")
    logger.info("=" * 60)

    preprocessor = DataPreprocessor(scaling_method='minmax')
    result = preprocessor.full_preprocessing_pipeline(
        target_col=target_col,
        forecast_horizon=forecast_horizon
    )

    logger.info("Korak 2 zavrsen.")
    return result, preprocessor


# ============================================================
# KORAK 3: IZGRADNJA I TRENIRANJE MODELA
# ============================================================

def step3_build_and_train(prep_result):
    """Izgradnja i treniranje 3 modela: LSTM, GRU, Hibridni."""
    logger.info("=" * 60)
    logger.info("KORAK 3: IZGRADNJA I TRENIRANJE MODELA")
    logger.info("=" * 60)

    X_train = prep_result['X_train']
    X_val = prep_result['X_val']
    y_train = prep_result['y_train']
    y_val = prep_result['y_val']

    input_shape = (X_train.shape[1], X_train.shape[2])
    output_steps = y_train.shape[1]

    logger.info(f"Input shape : {input_shape}")
    logger.info(f"Output steps: {output_steps}")

    models = {}
    histories = {}

    # --- LSTM ---
    logger.info("")
    logger.info("--- LSTM Model ---")
    lstm = LSTMModel()
    lstm.build_model(input_shape, output_steps)

    for cb in lstm.get_callbacks('lstm_model'):
        if hasattr(cb, 'log_dir'):
            cb.log_dir = get_tensorboard_logdir('lstm_model')

    lstm_history = lstm.train(X_train, y_train, X_val, y_val)
    lstm.save_model()
    models['LSTM'] = lstm
    histories['LSTM'] = lstm_history

    # --- GRU ---
    logger.info("")
    logger.info("--- GRU Model ---")
    gru = GRUModel()
    gru.build_model(input_shape, output_steps)

    for cb in gru.get_callbacks('gru_model'):
        if hasattr(cb, 'log_dir'):
            cb.log_dir = get_tensorboard_logdir('gru_model')

    gru_history = gru.train(X_train, y_train, X_val, y_val)
    gru.save_model()
    models['GRU'] = gru
    histories['GRU'] = gru_history

    # --- Hibridni ---
    logger.info("")
    logger.info("--- Hibridni LSTM-GRU Model ---")
    hybrid = HybridModel()
    hybrid.build_model(input_shape, output_steps)

    for cb in hybrid.get_callbacks('hybrid_model'):
        if hasattr(cb, 'log_dir'):
            cb.log_dir = get_tensorboard_logdir('hybrid_model')

    hybrid_history = hybrid.train(X_train, y_train, X_val, y_val)
    hybrid.save_model()
    models['Hybrid'] = hybrid
    histories['Hybrid'] = hybrid_history

    logger.info("Korak 3 zavrsen.")
    return models, histories


# ============================================================
# KORAK 4: EVALUACIJA
# ============================================================

def step4_evaluation(models, prep_result):
    """Evaluira i usporeduje sve modele."""
    logger.info("=" * 60)
    logger.info("KORAK 4: EVALUACIJA MODELA")
    logger.info("=" * 60)

    X_test = prep_result['X_test']
    y_test = prep_result['y_test']
    target_col = prep_result['target_column']
    scalers = prep_result['scalers']

    evaluator = ModelEvaluator()
    predictions = {}

    target_scaler = scalers.get(f'{target_col}_original', None)

    for name, model in models.items():
        logger.info(f"Evaluiram {name}...")
        y_pred = model.predict(X_test)
        predictions[name] = y_pred
        evaluator.evaluate_model(name, y_test, y_pred, target_scaler)

    comparison = evaluator.compare_models()
    evaluator.save_results()

    logger.info("Korak 4 zavrsen.")
    return evaluator, predictions, comparison


# ============================================================
# KORAK 5: GENERIRANJE PROGNOZA
# ============================================================

def step5_forecasting(best_model, prep_result):
    """Generira kratkorocne (24h) i srednjorocne (7 dana) prognoze."""
    logger.info("=" * 60)
    logger.info("KORAK 5: GENERIRANJE PROGNOZA")
    logger.info("=" * 60)

    X_test = prep_result['X_test']
    target_col = prep_result['target_column']
    scalers = prep_result['scalers']

    target_scaler = scalers.get(f'{target_col}_original', None)

    # --- Kratkorocna prognoza (24h) ---
    logger.info("Kratkorocna prognoza (24h)...")
    last_seq = X_test[-1:]
    short_pred = best_model.predict(last_seq)

    if target_scaler is not None:
        short_values = target_scaler.inverse_transform(
            short_pred.reshape(-1, 1)
        ).flatten()
    else:
        short_values = short_pred.flatten()

    logger.info("Prognoza za sljedecih 24h:")
    for h, val in enumerate(short_values[:24]):
        logger.info(f"  t+{h+1:02d}h: {val:.2f} ug/m3")

    # --- Srednjorocna prognoza (7 x 24h) ---
    logger.info("Srednjorocna prognoza (7 dana) - autoregresivno...")
    medium_values = []
    current_seq = X_test[-1:].copy()
    feature_cols = prep_result['feature_columns']

    target_idx = feature_cols.index(target_col) \
        if target_col in feature_cols else 0

    for day in range(7):
        day_pred = best_model.predict(current_seq)

        if target_scaler is not None:
            day_vals = target_scaler.inverse_transform(
                day_pred.reshape(-1, 1)
            ).flatten()
        else:
            day_vals = day_pred.flatten()

        # Sigurnosna provjera duljine
        if len(day_vals) >= 24:
            medium_values.extend(day_vals[:24].tolist())
        else:
            # Ako je predikcija kraca, popuni s zadnjom vrijednoscu
            padded = list(day_vals) + [day_vals[-1]] * (24 - len(day_vals))
            medium_values.extend(padded[:24])

        # Pomakni prozor unaprijed (autoregresivno)
        try:
            pred_norm = day_pred.flatten()[:24]
            if len(pred_norm) < 24:
                pred_norm = np.pad(pred_norm,
                                   (0, 24 - len(pred_norm)),
                                   mode='edge')

            new_step = current_seq[0, -24:, :].copy()
            new_step[:, target_idx] = pred_norm

            current_seq = np.concatenate([
                current_seq[:, 24:, :],
                new_step.reshape(1, 24, -1)
            ], axis=1)
        except Exception as e:
            logger.warning(f"Pomicanje prozora dan {day+1}: {e}")
            break

    logger.info("Dnevni prosjeci za 7 dana:")
    for day in range(7):
        s = day * 24
        e = s + 24
        if e <= len(medium_values):
            avg = np.mean(medium_values[s:e])
            mx = np.max(medium_values[s:e])
            logger.info(f"  Dan {day+1}: prosjek={avg:.2f}, "
                        f"max={mx:.2f} ug/m3")

    # Spremi prognoze
    forecast_df = pd.DataFrame({
        'hour_ahead': range(1, len(medium_values) + 1),
        'day': [(h - 1) // 24 + 1 for h in range(1, len(medium_values) + 1)],
        'predicted_value': medium_values
    })
    forecast_file = os.path.join(PREDICTIONS_DIR, 'forecast_7days.csv')
    forecast_df.to_csv(forecast_file, index=False, encoding='utf-8')
    logger.info(f"Prognoza spremljena: {forecast_file}")

    short_df = pd.DataFrame({
        'hour_ahead': range(1, 25),
        'predicted_value': short_values[:24].tolist()
    })
    short_file = os.path.join(PREDICTIONS_DIR, 'forecast_24h.csv')
    short_df.to_csv(short_file, index=False, encoding='utf-8')
    logger.info(f"Kratkorocna prognoza spremljena: {short_file}")

    logger.info("Korak 5 zavrsen.")
    return short_values, medium_values


# ============================================================
# KORAK 6: VIZUALIZACIJA
# ============================================================

def step6_visualization(prep_result, histories, predictions,
                         comparison, short_term, medium_term,
                         target_col='PM2.5'):
    """Kreira sve vizualizacije."""
    logger.info("=" * 60)
    logger.info("KORAK 6: VIZUALIZACIJA")
    logger.info("=" * 60)

    try:
        viz = Visualizer()

        # Ucitaj obradjene podatke
        processed_file = os.path.join(PROCESSED_DATA_DIR, 'processed_data.csv')
        if os.path.exists(processed_file):
            df = pd.read_csv(processed_file, parse_dates=['datetime'],
                            encoding='utf-8')
        else:
            df = prep_result.get('processed_data', pd.DataFrame())

        if not df.empty:
            viz.plot_raw_data_overview(df)
            viz.plot_seasonal_patterns(df, target_col)
            viz.plot_correlation_matrix(df)
            logger.info("Grafovi podataka kreirani.")

        if histories:
            viz.plot_training_history(
                list(histories.values()),
                list(histories.keys())
            )
            logger.info("Grafovi treniranja kreirani.")

        for name, pred in predictions.items():
            viz.plot_predictions_vs_actual(
                prep_result['y_test'], pred, name, target_col
            )
        logger.info("Grafovi predikcija kreirani.")

        if comparison is not None:
            viz.plot_model_comparison(comparison)

        viz.plot_forecast(
            short_term[:24],
            target_col=target_col,
            title=f'Kratkorocna prognoza {target_col} (24h)'
        )

        viz.plot_forecast(
            medium_term[:168],
            target_col=target_col,
            title=f'Srednjorocna prognoza {target_col} (7 dana)'
        )

        logger.info("Sve vizualizacije kreirane.")

    except Exception as e:
        logger.error(f"Greska pri vizualizaciji: {e}", exc_info=True)
        logger.warning("Vizualizacija preskocena, nastavlja se dalje.")

    logger.info("Korak 6 zavrsen.")


# ============================================================
# GLAVNA FUNKCIJA
# ============================================================

def main():
    """Glavna funkcija - izvrsava cijeli pipeline."""
    start_time = datetime.now()

    print_header()

    # --- Korak 1: Podaci ---
    try:
        data = step1_data_collection()
    except Exception as e:
        logger.error(f"Korak 1 - greska: {e}", exc_info=True)
        sys.exit(1)

    # --- Korak 2: Predobrada ---
    target_col = 'PM2.5'
    try:
        prep_result, preprocessor = step2_preprocessing(
            target_col=target_col,
            forecast_horizon=24
        )
    except Exception as e:
        logger.error(f"Korak 2 - greska: {e}", exc_info=True)
        sys.exit(1)

    # --- Korak 3: Treniranje ---
    try:
        models, histories = step3_build_and_train(prep_result)
    except Exception as e:
        logger.error(f"Korak 3 - greska: {e}", exc_info=True)
        sys.exit(1)

    # --- Korak 4: Evaluacija ---
    try:
        evaluator, predictions, comparison = step4_evaluation(
            models, prep_result
        )
    except Exception as e:
        logger.error(f"Korak 4 - greska: {e}", exc_info=True)
        comparison = None
        predictions = {}
        evaluator = None

    # --- Odaberi najbolji model ---
    if comparison is not None and not comparison.empty:
        best_model_name = comparison.iloc[0]['Model']
    else:
        best_model_name = 'LSTM'

    best_model = models[best_model_name]
    logger.info(f"Najbolji model: {best_model_name}")

    # --- Korak 5: Prognoza ---
    try:
        short_term, medium_term = step5_forecasting(
            best_model, prep_result
        )
    except Exception as e:
        logger.error(f"Korak 5 - greska: {e}", exc_info=True)
        short_term = [0.0] * 24
        medium_term = [0.0] * 168

    # --- Korak 6: Vizualizacija ---
    step6_visualization(
        prep_result, histories, predictions, comparison,
        short_term, medium_term, target_col
    )

    # --- Zavrsetak ---
    elapsed = datetime.now() - start_time

    print("")
    print("=" * 65)
    print("  PROGRAM ZAVRSEN USPJESNO")
    print("=" * 65)
    print(f"  Ukupno vrijeme      : {elapsed}")
    print(f"  Najbolji model      : {best_model_name}")
    print(f"  Ciljna varijabla    : {target_col}")
    print(f"  Kratkorocna prognoza: data/predictions/forecast_24h.csv")
    print(f"  Srednjorocna prognoza: data/predictions/forecast_7days.csv")
    print(f"  Rezultati           : reports/metrics/")
    print(f"  Vizualizacije       : reports/figures/")
    print(f"  Modeli              : models/saved/")
    print("=" * 65)


if __name__ == '__main__':
    main()