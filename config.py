"""
Konfiguracija sustava za predviđanje kvalitete zraka.
Sadrži sve parametre, URL-ove izvora podataka i postavke modela.
"""

import os
from datetime import datetime

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

for dir_path in [RAW_DATA_DIR, PROCESSED_DATA_DIR, PREDICTIONS_DIR,
                 SAVED_MODELS_DIR, CHECKPOINTS_DIR, FIGURES_DIR,
                 METRICS_DIR, LOGS_DIR]:
    os.makedirs(dir_path, exist_ok=True)

# ============================================================
# IZVORI PODATAKA
# ============================================================
EEA_BASE_URL = "https://discomap.eea.europa.eu/map/fme/AirQualityExport.htm"
EEA_DOWNLOAD_URL = "https://fme.discomap.eea.europa.eu/fmedatastreaming/AirQualityDownload/AQData_Extract.fmw"
OPENAQ_API_URL = "https://api.openaq.org/v2/measurements"
OPENMETEO_API_URL = "https://archive-api.open-meteo.com/v1/archive"
DHMZ_BASE_URL = "https://meteo.hr"
DHMZ_KVALITETA_ZRAKA_URL = "https://zrak.hr"
MZOIP_URL = "https://mingor.gov.hr"
AQICN_API_URL = "https://api.waqi.info"

# ============================================================
# POSTAJE ZA MJERENJE KVALITETE ZRAKA U HRVATSKOJ
# ============================================================
MONITORING_STATIONS = {
    'Zagreb-1': {'id': 'HR0001A', 'name': 'Zagreb-1 (sjever)',
                 'lat': 45.8150, 'lon': 15.9819, 'city': 'Zagreb'},
    'Zagreb-2': {'id': 'HR0002A', 'name': 'Zagreb-2 (centar)',
                 'lat': 45.8130, 'lon': 15.9780, 'city': 'Zagreb'},
    'Zagreb-3': {'id': 'HR0003A', 'name': 'Zagreb-3 (istok)',
                 'lat': 45.8020, 'lon': 16.0100, 'city': 'Zagreb'},
    'Split':    {'id': 'HR0004A', 'name': 'Split-Bačvice',
                 'lat': 43.5081, 'lon': 16.4402, 'city': 'Split'},
    'Rijeka':   {'id': 'HR0005A', 'name': 'Rijeka-centar',
                 'lat': 45.3271, 'lon': 14.4422, 'city': 'Rijeka'},
    'Osijek':   {'id': 'HR0006A', 'name': 'Osijek-centar',
                 'lat': 45.5550, 'lon': 18.6955, 'city': 'Osijek'},
    'Sisak':    {'id': 'HR0007A', 'name': 'Sisak-industrijsko',
                 'lat': 45.4866, 'lon': 16.3728, 'city': 'Sisak'},
    'Kutina':   {'id': 'HR0008A', 'name': 'Kutina-industrijsko',
                 'lat': 45.4753, 'lon': 16.7841, 'city': 'Kutina'}
}

# ============================================================
# ZAGAĐIVAČI
# ============================================================
POLLUTANTS = {
    'PM10':  {'name': 'Čestice PM10', 'unit': 'µg/m³',
              'who_limit_24h': 45, 'who_limit_annual': 15,
              'eu_limit_24h': 50, 'eu_limit_annual': 40, 'eea_code': 5},
    'PM2.5': {'name': 'Čestice PM2.5', 'unit': 'µg/m³',
              'who_limit_24h': 15, 'who_limit_annual': 5,
              'eu_limit_annual': 25, 'eea_code': 6001},
    'NO2':   {'name': 'Dušikov dioksid', 'unit': 'µg/m³',
              'who_limit_24h': 25, 'who_limit_annual': 10,
              'eu_limit_1h': 200, 'eu_limit_annual': 40, 'eea_code': 8},
    'SO2':   {'name': 'Sumporov dioksid', 'unit': 'µg/m³',
              'who_limit_24h': 40, 'eu_limit_1h': 350,
              'eu_limit_24h': 125, 'eea_code': 1},
    'O3':    {'name': 'Ozon', 'unit': 'µg/m³',
              'who_limit_8h': 100, 'eu_limit_8h': 120, 'eea_code': 7},
    'CO':    {'name': 'Ugljikov monoksid', 'unit': 'mg/m³',
              'who_limit_24h': 4, 'eu_limit_8h': 10, 'eea_code': 10}
}

# ============================================================
# AQI KATEGORIJE
# ============================================================
AQI_CATEGORIES = {
    'Dobro':                 {'range': (0, 50),    'color': '#00e400'},
    'Umjereno':              {'range': (51, 100),  'color': '#ffff00'},
    'Nezdrav za osjetljive': {'range': (101, 150), 'color': '#ff7e00'},
    'Nezdrav':               {'range': (151, 200), 'color': '#ff0000'},
    'Vrlo nezdrav':          {'range': (201, 300), 'color': '#8f3f97'},
    'Opasan':                {'range': (301, 500), 'color': '#7e0023'}
}

# ============================================================
# METEOROLOŠKI PARAMETRI
# ============================================================
METEO_PARAMS = ['temperature', 'humidity', 'pressure', 'wind_speed',
                'wind_direction', 'precipitation', 'solar_radiation', 'cloud_cover']

# ============================================================
# PARAMETRI MODELA
# ============================================================
SHORT_TERM_HOURS = 24
MEDIUM_TERM_HOURS = 168

LSTM_CONFIG = {
    'lookback_hours': 72, 'units_layer1': 128, 'units_layer2': 64,
    'units_layer3': 32, 'dropout_rate': 0.2, 'recurrent_dropout': 0.0,
    'learning_rate': 0.001, 'batch_size': 32, 'epochs': 100,
    'patience': 15, 'optimizer': 'adam', 'loss': 'mse',
    'bidirectional': True, 'attention': True
}

GRU_CONFIG = {
    'lookback_hours': 72, 'units_layer1': 128, 'units_layer2': 64,
    'units_layer3': 32, 'dropout_rate': 0.2, 'recurrent_dropout': 0.0,
    'learning_rate': 0.001, 'batch_size': 32, 'epochs': 100,
    'patience': 15, 'optimizer': 'adam', 'loss': 'mse',
    'bidirectional': True
}

HYBRID_CONFIG = {
    'lookback_hours': 72, 'lstm_units': 64, 'gru_units': 64,
    'dense_units': 32, 'dropout_rate': 0.2, 'learning_rate': 0.0005,
    'batch_size': 32, 'epochs': 150, 'patience': 20
}

# ============================================================
# PARAMETRI TRENIRANJA
# ============================================================
TRAIN_TEST_SPLIT = 0.8
VALIDATION_SPLIT = 0.15
RANDOM_SEED = 42
SCALING_METHOD = 'minmax'
CROSS_VALIDATION_FOLDS = 5

# ============================================================
# WEB SCRAPING POSTAVKE
# ============================================================
SCRAPING_CONFIG = {
    'timeout': 30,
    'max_retries': 3,
    'delay_between_requests': 2,
    'user_agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
}

# ============================================================
# VREMENSKI RASPON PODATAKA (FIKSNI za reproducibilnost)
# ============================================================
DATA_START_DATE = '2022-01-01'
DATA_END_DATE = '2024-12-31'

print("✓ Konfiguracija učitana uspješno.")