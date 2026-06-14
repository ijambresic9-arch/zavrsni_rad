"""
Modul za pripremu i predobradu podataka za TensorFlow modele.

Algoritmi koji se koriste:
1. Interpolacija nedostajucih vrijednosti (linearna, kubna)
2. Min-Max normalizacija / Standardizacija
3. Klizni prozor (sliding window) za formiranje sekvenci
4. Detekcija i uklanjanje outliera (IQR metoda)
5. Feature scaling za neuronske mreze

ISPRAVCI U OVOJ VERZIJI:
- Popravljen DATA LEAKAGE: scaler se sada fit-a SAMO na trening podacima
- Zamijenjene deprecated .fillna(method=) metode s .ffill()/.bfill()
- Dodana validacija sekvenci
- Bolja agregacija po postajama
"""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import joblib
import logging
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# KONFIGURACIJA
# ============================================================
try:
    from config import (
        RAW_DATA_DIR, PROCESSED_DATA_DIR, LOGS_DIR,
        POLLUTANTS, MONITORING_STATIONS,
        TRAIN_TEST_SPLIT, VALIDATION_SPLIT,
        SCALING_METHOD, RANDOM_SEED, LSTM_CONFIG
    )
except Exception as e:
    print(f"config.py nije dostupan ({e}), koristim ugradjene vrijednosti.")

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
    PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')
    LOGS_DIR = os.path.join(BASE_DIR, 'logs')

    TRAIN_TEST_SPLIT = 0.8
    VALIDATION_SPLIT = 0.15
    RANDOM_SEED = 42
    SCALING_METHOD = 'minmax'

    POLLUTANTS = {
        'PM10':  {'name': 'Cestice PM10', 'unit': 'ug/m3', 'eu_limit_24h': 50},
        'PM2.5': {'name': 'Cestice PM2.5', 'unit': 'ug/m3'},
        'NO2':   {'name': 'Dusikov dioksid', 'unit': 'ug/m3', 'eu_limit_24h': 200},
        'SO2':   {'name': 'Sumporov dioksid', 'unit': 'ug/m3'},
        'O3':    {'name': 'Ozon', 'unit': 'ug/m3'},
        'CO':    {'name': 'Ugljikov monoksid', 'unit': 'mg/m3'},
    }

    MONITORING_STATIONS = {
        'Zagreb-1': {'id': 'HR0001A', 'name': 'Zagreb-1', 'city': 'Zagreb'},
    }

    LSTM_CONFIG = {'lookback_hours': 72}

for d in [RAW_DATA_DIR, PROCESSED_DATA_DIR, LOGS_DIR]:
    os.makedirs(d, exist_ok=True)

# ============================================================
# LOGIRANJE
# ============================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(LOGS_DIR, 'preprocessing.log'),
                            encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


# ============================================================
# GLAVNA KLASA ZA PREDOBRADU
# ============================================================

class DataPreprocessor:
    """
    Klasa za kompletnu predobradu podataka.

    Pipeline predobrade:
    1. Ucitavanje / generiranje podataka
    2. Spajanje izvora podataka
    3. Ciscenje podataka
    4. Detekcija outliera
    5. Interpolacija nedostajucih vrijednosti
    6. Feature engineering (vremenske i lag znacajke)
    7. Normalizacija (FIT SAMO NA TRENING!)
    8. Kreiranje vremenskih sekvenci
    9. Podjela na trening/validacija/test
    """

    def __init__(self, scaling_method=SCALING_METHOD):
        self.scaling_method = scaling_method
        self.scalers = {}
        self.feature_columns = []
        self.target_columns = []
        self.data = None
        logger.info(f"DataPreprocessor inicijaliziran (scaling: {scaling_method})")

    # ----------------------------------------------------------
    # KORAK 1: UCITAVANJE PODATAKA
    # ----------------------------------------------------------

    def load_and_merge_data(self):
        """Ucitava podatke iz CSV datoteka i spaja ih."""
        logger.info("Ucitavam podatke...")

        # --- Podaci o kvaliteti zraka ---
        aq_file = os.path.join(RAW_DATA_DIR, 'air_quality_data.csv')

        if not os.path.exists(aq_file):
            raise FileNotFoundError(
                f"Datoteka {aq_file} ne postoji!\n"
                f"Pokreni data_scraper.py prije ovog modula."
            )

        aq_data = pd.read_csv(aq_file, parse_dates=['datetime'])
        logger.info(f"  OK Kvaliteta zraka: {len(aq_data):,} zapisa")

        # --- Meteoroloski podaci ---
        meteo_file_omt = os.path.join(RAW_DATA_DIR, 'openmeteo_data.csv')
        meteo_file_syn = os.path.join(RAW_DATA_DIR, 'synthetic_meteo_data.csv')

        if os.path.exists(meteo_file_omt):
            meteo_data = pd.read_csv(meteo_file_omt, parse_dates=['datetime'])
            logger.info(f"  OK Open-Meteo: {len(meteo_data):,} zapisa")
        elif os.path.exists(meteo_file_syn):
            meteo_data = pd.read_csv(meteo_file_syn, parse_dates=['datetime'])
            logger.info(f"  OK Sinteticki meteo: {len(meteo_data):,} zapisa")
        else:
            raise FileNotFoundError(
                "Meteoroloski podaci ne postoje!\n"
                "Pokreni data_scraper.py prije ovog modula."
            )

        # --- Spajanje ---
        logger.info("Spajam podatke...")
        aq_data = aq_data.sort_values('datetime').reset_index(drop=True)
        meteo_data = meteo_data.sort_values('datetime').reset_index(drop=True)

        merged_data = pd.merge_asof(
            aq_data, meteo_data, on='datetime',
            direction='nearest', tolerance=pd.Timedelta('1h')
        )

        logger.info(f"  OK Spojeni podaci: {len(merged_data):,} zapisa, "
                    f"{len(merged_data.columns)} stupaca")

        self.data = merged_data
        return merged_data

    # ----------------------------------------------------------
    # KORAK 2: CISCENJE PODATAKA
    # ----------------------------------------------------------

    def clean_data(self, df=None):
        """
        Cisti podatke od nekonzistentnosti i gresaka.

        Algoritam:
        1. Ukloni potpuno prazne redove
        2. Zamijeni negativne koncentracije s NaN
        3. Ukloni fizikalno nemoguce vrijednosti
        4. Detektira outliere (IQR metoda)
        5. Ukloni duplikate
        """
        if df is None:
            df = self.data.copy()

        logger.info("Cistim podatke...")
        initial_len = len(df)

        df = df.dropna(how='all')

        pollutant_cols = [c for c in POLLUTANTS if c in df.columns]
        for col in pollutant_cols:
            neg_count = (df[col] < 0).sum()
            if neg_count > 0:
                df.loc[df[col] < 0, col] = np.nan
                logger.debug(f"  {col}: {neg_count} negativnih -> NaN")

        physical_limits = {
            'PM10': (0, 1000), 'PM2.5': (0, 500),
            'NO2': (0, 500), 'SO2': (0, 500),
            'O3': (0, 500), 'CO': (0, 50),
            'temperature': (-40, 50), 'humidity': (0, 100),
            'pressure': (900, 1100), 'wind_speed': (0, 50),
            'wind_direction': (0, 360), 'precipitation': (0, 200),
            'solar_radiation': (0, 1500), 'cloud_cover': (0, 100)
        }

        for col, (min_val, max_val) in physical_limits.items():
            if col in df.columns:
                mask = (df[col] < min_val) | (df[col] > max_val)
                if mask.sum() > 0:
                    df.loc[mask, col] = np.nan

        # IQR za zagadivace - konzervativniji factor=5
        for col in pollutant_cols:
            df = self._remove_outliers_iqr(df, col, factor=5.0)

        # Ukloni duplikate
        subset_cols = ['datetime', 'station_id'] \
            if 'station_id' in df.columns else ['datetime']
        df = df.drop_duplicates(subset=subset_cols, keep='first')

        removed = initial_len - len(df)
        logger.info(f"  OK Ciscenje zavrseno: uklonjeno {removed} zapisa, "
                    f"preostalo {len(df):,}")

        self.data = df
        return df

    def _remove_outliers_iqr(self, df, column, factor=1.5):
        """Uklanja outliere IQR metodom."""
        if column not in df.columns:
            return df

        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1

        if IQR == 0:
            return df

        lower = Q1 - factor * IQR
        upper = Q3 + factor * IQR

        mask = (df[column] < lower) | (df[column] > upper)
        n_out = mask.sum()

        if n_out > 0:
            df.loc[mask, column] = np.nan
            logger.debug(f"  IQR: {column} -> {n_out} outliera")

        return df

    # ----------------------------------------------------------
    # KORAK 3: INTERPOLACIJA (POPRAVLJENO!)
    # ----------------------------------------------------------

    def interpolate_missing(self, df=None):
        """
        Interpolira nedostajuce vrijednosti.

        ISPRAVAK: Zamijenjene deprecated metode .fillna(method=)
                  s novim .ffill() i .bfill() (Pandas 2.x kompatibilno)
        """
        if df is None:
            df = self.data.copy()

        logger.info("Interpoliram nedostajuce vrijednosti...")

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        total_before = df[numeric_cols].isnull().sum().sum()

        if 'datetime' in df.columns:
            df = df.sort_values('datetime').reset_index(drop=True)

        if 'station_id' in df.columns:
            groups = []
            for station, group in df.groupby('station_id'):
                group = group.sort_values('datetime')
                for col in numeric_cols:
                    # Linearna interpolacija za kratke razmake
                    group[col] = group[col].interpolate(method='linear', limit=6)
                    group[col] = group[col].interpolate(method='linear', limit=24)
                    # NOVO: ffill/bfill umjesto deprecated fillna(method=)
                    group[col] = group[col].ffill(limit=48)
                    group[col] = group[col].bfill(limit=48)
                groups.append(group)
            df = pd.concat(groups, ignore_index=True)
            df = df.sort_values(['station_id', 'datetime']).reset_index(drop=True)
        else:
            for col in numeric_cols:
                df[col] = df[col].interpolate(method='linear', limit=6)
                df[col] = df[col].interpolate(method='linear', limit=24)
                df[col] = df[col].ffill(limit=48)
                df[col] = df[col].bfill(limit=48)

        # Preostale NaN -> medijan
        for col in numeric_cols:
            if df[col].isnull().any():
                df[col] = df[col].fillna(df[col].median())

        total_after = df[numeric_cols].isnull().sum().sum()
        logger.info(f"  OK Interpolacija: {total_before} -> {total_after} nedostajucih")

        self.data = df
        return df

    # ----------------------------------------------------------
    # KORAK 4: FEATURE ENGINEERING
    # ----------------------------------------------------------

    def add_temporal_features(self, df=None):
        """
        Dodaje vremenske znacajke s ciklickim sin/cos transformacijama.

        Primjer: sat=23 i sat=0 su blizu u realnosti, ali daleko
        numericki. sin/cos transformacija to ispravlja.
        """
        if df is None:
            df = self.data.copy()

        logger.info("Dodajem vremenske znacajke...")

        if 'datetime' not in df.columns:
            logger.warning("  Nema datetime stupca!")
            return df

        dt = pd.to_datetime(df['datetime'])

        df['hour'] = dt.dt.hour
        df['day_of_week'] = dt.dt.dayofweek
        df['day_of_year'] = dt.dt.dayofyear
        df['month'] = dt.dt.month
        df['week_of_year'] = dt.dt.isocalendar().week.astype(int)

        # Ciklicke transformacije
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        df['doy_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365)
        df['doy_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365)

        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        df['is_rush_hour'] = (
            (df['hour'].between(7, 9)) | (df['hour'].between(16, 19))
        ).astype(int)
        df['is_night'] = (
            (df['hour'] >= 22) | (df['hour'] <= 5)
        ).astype(int)

        season_map = {12: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1,
                      6: 2, 7: 2, 8: 2, 9: 3, 10: 3, 11: 3}
        df['season'] = df['month'].map(season_map)

        logger.info("  OK Dodano 16 vremenskih znacajki")

        self.data = df
        return df

    def add_lag_features(self, df=None, target_cols=None,
                         lag_hours=None):
        """
        Dodaje lag (zaostale) i rolling (klizne) znacajke.
        """
        if df is None:
            df = self.data.copy()

        if lag_hours is None:
            lag_hours = [1, 2, 3, 6, 12, 24]

        if target_cols is None:
            # Ukljuci i meteoroloske varijable
            pollutant_lags = [c for c in POLLUTANTS if c in df.columns]
            meteo_lags = [c for c in ['temperature', 'humidity',
                                       'wind_speed', 'pressure']
                         if c in df.columns]
            target_cols = pollutant_lags + meteo_lags

        logger.info(f"Dodajem lag znacajke za {len(target_cols)} varijabli...")

        group_col = 'station_id' if 'station_id' in df.columns else None

        for col in target_cols:
            if col not in df.columns:
                continue

            for lag in lag_hours:
                col_name = f'{col}_lag_{lag}h'
                if group_col:
                    df[col_name] = df.groupby(group_col)[col].shift(lag)
                else:
                    df[col_name] = df[col].shift(lag)

            for window, suffix in [(24, '24h'), (48, '48h')]:
                if group_col:
                    df[f'{col}_mean_{suffix}'] = df.groupby(group_col)[col].transform(
                        lambda x: x.rolling(window=window, min_periods=1).mean()
                    )
                    df[f'{col}_std_{suffix}'] = df.groupby(group_col)[col].transform(
                        lambda x: x.rolling(window=window, min_periods=1).std().fillna(0)
                    )
                    df[f'{col}_max_{suffix}'] = df.groupby(group_col)[col].transform(
                        lambda x: x.rolling(window=window, min_periods=1).max()
                    )
                else:
                    df[f'{col}_mean_{suffix}'] = df[col].rolling(
                        window=window, min_periods=1).mean()
                    df[f'{col}_std_{suffix}'] = df[col].rolling(
                        window=window, min_periods=1).std().fillna(0)
                    df[f'{col}_max_{suffix}'] = df[col].rolling(
                        window=window, min_periods=1).max()

            # Razlike
            if group_col:
                df[f'{col}_diff_1h'] = df.groupby(group_col)[col].diff(1)
                df[f'{col}_diff_24h'] = df.groupby(group_col)[col].diff(24)
            else:
                df[f'{col}_diff_1h'] = df[col].diff(1)
                df[f'{col}_diff_24h'] = df[col].diff(24)

        # NOVO: bfill() umjesto fillna(method='bfill')
        df = df.bfill()
        df = df.fillna(0)

        new_cols = len(df.columns)
        logger.info(f"  OK Ukupno stupaca nakon lag znacajki: {new_cols}")

        self.data = df
        return df

    # ----------------------------------------------------------
    # KORAK 5: NORMALIZACIJA (ISPRAVLJEN DATA LEAKAGE!)
    # ----------------------------------------------------------

    def normalize_data(self, df=None, target_cols=None,
                       train_end_idx=None):
        """
        Normalizira numericke podatke.

        KRITICAN ISPRAVAK: Scaler se sada FIT-a SAMO na trening podacima!
        Stari kod je radio fit_transform na CIJELOM datasetu sto je
        uzrokovalo DATA LEAKAGE - model je posredno "vidio" test podatke.

        Min-Max normalizacija:
            x' = (x - min) / (max - min)  -> rezultat u [0, 1]
        """
        if df is None:
            df = self.data.copy()

        if target_cols is None:
            target_cols = [c for c in POLLUTANTS if c in df.columns]

        logger.info(f"Normaliziram podatke ({self.scaling_method})...")

        exclude = {'station_id', 'latitude', 'longitude', 'season',
                   'is_weekend', 'is_rush_hour', 'is_night',
                   'hour', 'day_of_week', 'day_of_year', 'month', 'week_of_year'}

        numeric_cols = [
            c for c in df.select_dtypes(include=[np.number]).columns
            if c not in exclude
        ]

        if not numeric_cols:
            logger.warning("  Nema numerickih stupaca!")
            return df

        # Odredi granicu trening seta
        if train_end_idx is None:
            train_end_idx = int(len(df) * TRAIN_TEST_SPLIT * (1 - VALIDATION_SPLIT))

        # KRITICNO: FIT SAMO na trening podacima!
        train_subset = df.iloc[:train_end_idx][numeric_cols]

        if self.scaling_method == 'minmax':
            self.scalers['features'] = MinMaxScaler(feature_range=(0, 1))
        else:
            self.scalers['features'] = StandardScaler()

        # Fit na train -> transform na sve
        self.scalers['features'].fit(train_subset)
        df[numeric_cols] = self.scalers['features'].transform(df[numeric_cols])

        logger.info(f"  OK Scaler fit na {train_end_idx:,} trening uzoraka")
        logger.info(f"  OK Transform primijenjen na svih {len(df):,} uzoraka")

        # Individualni scaleri za ciljne varijable
        for col in target_cols:
            if col in numeric_cols:
                if self.scaling_method == 'minmax':
                    self.scalers[col] = MinMaxScaler(feature_range=(0, 1))
                else:
                    self.scalers[col] = StandardScaler()

        # Spremi scalere
        scaler_file = os.path.join(PROCESSED_DATA_DIR, 'scalers.pkl')
        joblib.dump(self.scalers, scaler_file)
        logger.info(f"  OK Scaleri spremljeni: {scaler_file}")

        self.feature_columns = numeric_cols
        self.target_columns = target_cols
        self.data = df

        return df

    # ----------------------------------------------------------
    # KORAK 6: KREIRANJE SEKVENCI
    # ----------------------------------------------------------

    def create_sequences(self, df=None, lookback=None,
                         forecast_horizon=24, target_col='PM2.5',
                         station_id=None):
        """
        Kreira vremenske sekvence za LSTM/GRU.

        Klizni prozor (Sliding Window):
        Za svaki korak t:
          X[t] = podatci[t-lookback : t]       (svi stupci)
          y[t] = target[t : t+forecast_horizon] (samo ciljna var.)

        Dimenzije:
          X: (N, lookback, n_features)
          y: (N, forecast_horizon)
        """
        if df is None:
            df = self.data.copy()

        if lookback is None:
            lookback = LSTM_CONFIG.get('lookback_hours', 72)

        logger.info(f"Kreiram sekvence (lookback={lookback}h, "
                    f"forecast={forecast_horizon}h)...")

        # Agregacija po vremenu ako ima vise postaja
        if 'station_id' in df.columns:
            if station_id:
                df_seq = df[df['station_id'] == station_id].copy()
                logger.info(f"  Koristim postaju: {station_id}")
            else:
                logger.info("  Agregiram podatke svih postaja (prosjek)...")
                skip_cols = ['station_id', 'station_name', 'city',
                            'latitude', 'longitude']
                numeric_cols = [c for c in df.columns
                               if c not in skip_cols and
                               pd.api.types.is_numeric_dtype(df[c])]

                if 'datetime' in df.columns:
                    df_seq = df.groupby('datetime')[numeric_cols].mean().reset_index()
                    logger.info(f"  Agregirano: {len(df_seq):,} vremenskih tocaka")
                else:
                    df_seq = df.copy()
        else:
            df_seq = df.copy()

        if 'datetime' in df_seq.columns:
            df_seq = df_seq.sort_values('datetime').reset_index(drop=True)

        # Numericki stupci
        exclude = {'station_id', 'latitude', 'longitude'}
        feature_cols = [
            c for c in df_seq.select_dtypes(include=[np.number]).columns
            if c not in exclude
        ]

        if target_col not in feature_cols:
            raise ValueError(
                f"Ciljna varijabla '{target_col}' ne postoji!\n"
                f"Dostupne: {feature_cols[:10]}..."
            )

        target_idx = feature_cols.index(target_col)
        data_array = df_seq[feature_cols].values.astype(np.float32)

        X_list, y_list = [], []
        total = len(data_array)

        for i in range(lookback, total - forecast_horizon + 1):
            X_list.append(data_array[i - lookback: i])
            y_list.append(data_array[i: i + forecast_horizon, target_idx])

        if not X_list:
            raise ValueError(
                f"Nema dovoljno podataka za sekvence!\n"
                f"Potrebno: {lookback + forecast_horizon}, dostupno: {total}"
            )

        X = np.array(X_list, dtype=np.float32)
        y = np.array(y_list, dtype=np.float32)

        # Validacija
        assert not np.any(np.isnan(X)), "X sadrzi NaN!"
        assert not np.any(np.isnan(y)), "y sadrzi NaN!"

        logger.info(f"  OK X shape: {X.shape}")
        logger.info(f"  OK y shape: {y.shape}")
        logger.info(f"  OK Broj znacajki: {len(feature_cols)}")

        return X, y, feature_cols

    # ----------------------------------------------------------
    # KORAK 7: PODJELA PODATAKA
    # ----------------------------------------------------------

    def split_data(self, X, y, test_size=None, val_size=None):
        """
        Kronoloska podjela na trening / validacija / test.

        VAZNO: NE koristimo nasumicnu podjelu (shuffle)!
        Vremenski redoslijed mora biti ocuvan da se izbjegne
        data leakage (curenje buducih podataka u trening).
        """
        if test_size is None:
            test_size = 1 - TRAIN_TEST_SPLIT
        if val_size is None:
            val_size = VALIDATION_SPLIT

        n = len(X)
        test_start = int(n * (1 - test_size))
        val_start = int(test_start * (1 - val_size))

        X_train = X[:val_start]
        y_train = y[:val_start]
        X_val = X[val_start:test_start]
        y_val = y[val_start:test_start]
        X_test = X[test_start:]
        y_test = y[test_start:]

        logger.info("Podjela podataka:")
        logger.info(f"  Trening:    {len(X_train):>6,} uzoraka "
                    f"({len(X_train)/n*100:.1f}%)")
        logger.info(f"  Validacija: {len(X_val):>6,} uzoraka "
                    f"({len(X_val)/n*100:.1f}%)")
        logger.info(f"  Test:       {len(X_test):>6,} uzoraka "
                    f"({len(X_test)/n*100:.1f}%)")

        return X_train, X_val, X_test, y_train, y_val, y_test

    # ----------------------------------------------------------
    # KOMPLETNI PIPELINE
    # ----------------------------------------------------------

    def full_preprocessing_pipeline(self, target_col='PM2.5',
                                    station_id=None,
                                    forecast_horizon=24):
        """
        Izvrsava kompletni pipeline predobrade od A do Z.
        """
        logger.info("=" * 60)
        logger.info("KOMPLETNI PIPELINE PREDOBRADE PODATAKA")
        logger.info("=" * 60)

        # Koraci 1-5
        df = self.load_and_merge_data()
        df = self.clean_data(df)
        df = self.interpolate_missing(df)
        df = self.add_temporal_features(df)
        df = self.add_lag_features(df, target_cols=[target_col])

        # Spremi obradene podatke PRIJE normalizacije
        processed_file = os.path.join(PROCESSED_DATA_DIR, 'processed_data.csv')
        df.to_csv(processed_file, index=False, encoding='utf-8')
        logger.info(f"  OK Obradeni podaci spremljeni: {processed_file}")

        # KRITICNO: Fit orig scaler na trening dijelu PRIJE normalizacije
        if target_col in df.columns:
            n_total = len(df)
            train_end = int(n_total * TRAIN_TEST_SPLIT * (1 - VALIDATION_SPLIT))

            if self.scaling_method == 'minmax':
                orig_scaler = MinMaxScaler()
            else:
                orig_scaler = StandardScaler()

            # Fit SAMO na trening dijelu (nenormalizirano)
            orig_scaler.fit(df[target_col].values[:train_end].reshape(-1, 1))
            self.scalers[f'{target_col}_original'] = orig_scaler
            logger.info(f"  OK Original scaler za {target_col} kreiran "
                        f"(fit na {train_end:,} trening uzoraka)")

        # Korak 6: Normalizacija (s ispravnim train-only fit)
        n_total = len(df)
        train_end = int(n_total * TRAIN_TEST_SPLIT * (1 - VALIDATION_SPLIT))
        df = self.normalize_data(df, target_cols=[target_col],
                                  train_end_idx=train_end)

        # Korak 7: Sekvence
        X, y, feature_cols = self.create_sequences(
            df, forecast_horizon=forecast_horizon,
            target_col=target_col, station_id=station_id
        )

        # Korak 8: Podjela
        X_train, X_val, X_test, y_train, y_val, y_test = \
            self.split_data(X, y)

        # Spremi finalne scalere
        scaler_file = os.path.join(PROCESSED_DATA_DIR, 'scalers.pkl')
        joblib.dump(self.scalers, scaler_file)

        result = {
            'X_train': X_train, 'X_val': X_val, 'X_test': X_test,
            'y_train': y_train, 'y_val': y_val, 'y_test': y_test,
            'feature_columns': feature_cols,
            'target_column': target_col,
            'forecast_horizon': forecast_horizon,
            'scalers': self.scalers,
            'processed_data': df
        }

        logger.info("")
        logger.info("=" * 60)
        logger.info("OK PIPELINE PREDOBRADE ZAVRSEN USPJESNO!")
        logger.info(f"  Ciljna varijabla:  {target_col}")
        logger.info(f"  Prognozni horizont: {forecast_horizon}h")
        logger.info(f"  Broj znacajki:      {len(feature_cols)}")
        logger.info(f"  X_train shape:      {X_train.shape}")
        logger.info(f"  X_val shape:        {X_val.shape}")
        logger.info(f"  X_test shape:       {X_test.shape}")
        logger.info("=" * 60)

        return result


# ============================================================
# POKRETANJE KAO SAMOSTALNI SKRIPT
# ============================================================

if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("  DATA PREPROCESSING - Predvidjanje kvalitete zraka HR")
    print("=" * 60 + "\n")

    try:
        preprocessor = DataPreprocessor(scaling_method='minmax')

        result = preprocessor.full_preprocessing_pipeline(
            target_col='PM2.5',
            forecast_horizon=24,
            station_id=None
        )

        print("\nOK Predobrada zavrsena. Rezultati:\n")
        for key, val in result.items():
            if isinstance(val, np.ndarray):
                print(f"  {key:20s}: shape={val.shape}, dtype={val.dtype}")
            elif isinstance(val, list):
                print(f"  {key:20s}: {len(val)} elemenata")
            elif isinstance(val, dict):
                print(f"  {key:20s}: {len(val)} kljuceva")
            elif isinstance(val, pd.DataFrame):
                print(f"  {key:20s}: DataFrame {val.shape}")
            else:
                print(f"  {key:20s}: {val}")

    except Exception as e:
        logger.error(f"\nGRESKA: {e}", exc_info=True)
        sys.exit(1)