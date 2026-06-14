"""
Modul za prikupljanje podataka o kvaliteti zraka i meteorologiji.

Izvori podataka:
1. Open-Meteo API (BESPLATAN, radi - meteorologija)
2. OpenAQ API (BESPLATAN, radi - kvaliteta zraka)
3. EEA API (zastarjelo - fallback)
4. DHMZ scraping (fallback)
5. Sinteticki podaci (zadnji fallback)
"""

import os
import sys
import time
import logging
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import requests

warnings.filterwarnings("ignore")

# ============================================================
# KONFIGURACIJA
# ============================================================
try:
    from config import (
        RAW_DATA_DIR, LOGS_DIR, DATA_START_DATE, DATA_END_DATE,
        MONITORING_STATIONS, SCRAPING_CONFIG
    )
except Exception as e:
    print(f"config.py nije dostupan ({e}), koristim ugradjene vrijednosti.")

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, "data")
    RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
    LOGS_DIR = os.path.join(BASE_DIR, "logs")

    DATA_START_DATE = "2022-01-01"
    DATA_END_DATE = "2024-12-31"

    SCRAPING_CONFIG = {
        "timeout": 30, "max_retries": 3,
        "delay_between_requests": 2,
        "user_agent": "Mozilla/5.0"
    }

    MONITORING_STATIONS = {
        'Zagreb-1': {'id': 'HR0001A', 'name': 'Zagreb-1 (sjever)',
                     'lat': 45.8150, 'lon': 15.9819, 'city': 'Zagreb'},
        'Zagreb-2': {'id': 'HR0002A', 'name': 'Zagreb-2 (centar)',
                     'lat': 45.8130, 'lon': 15.9780, 'city': 'Zagreb'},
        'Split':    {'id': 'HR0004A', 'name': 'Split-Bacvice',
                     'lat': 43.5081, 'lon': 16.4402, 'city': 'Split'},
        'Rijeka':   {'id': 'HR0005A', 'name': 'Rijeka-centar',
                     'lat': 45.3271, 'lon': 14.4422, 'city': 'Rijeka'},
        'Osijek':   {'id': 'HR0006A', 'name': 'Osijek-centar',
                     'lat': 45.5550, 'lon': 18.6955, 'city': 'Osijek'},
    }

os.makedirs(RAW_DATA_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

# ============================================================
# LOGIRANJE
# ============================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(LOGS_DIR, "scraping.log"),
                            encoding="utf-8"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


# ============================================================
# KLASA ZA DOHVAT PODATAKA
# ============================================================

class AirQualityDataScraper:
    """
    Klasa za prikupljanje podataka o kvaliteti zraka.

    Implementira:
    - Web scraping s vise izvora (EEA, DHMZ, OpenAQ, Open-Meteo)
    - Retry logiku s eksponencijalnim backoff-om
    - Fallback na sinteticke podatke ako svi izvori padnu
    - Izracun AQI indeksa prema EPA standardu
    """

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": SCRAPING_CONFIG.get("user_agent", "Mozilla/5.0")
        })
        logger.info("AirQualityDataScraper inicijaliziran.")
        logger.info("Izvori podataka:")
        logger.info("  1. Open-Meteo API (meteorologija)")
        logger.info("  2. OpenAQ API (kvaliteta zraka)")
        logger.info("  3. EEA API (fallback)")
        logger.info("  4. Sinteticki podaci (zadnji fallback)")

    # --------------------------------------------------------
    # POMOCNE METODE
    # --------------------------------------------------------

    def _safe_request(self, url, params=None):
        """Siguran HTTP zahtjev s retry logikom."""
        retries = SCRAPING_CONFIG.get("max_retries", 3)
        timeout = SCRAPING_CONFIG.get("timeout", 30)
        delay = SCRAPING_CONFIG.get("delay_between_requests", 2)

        for attempt in range(retries):
            try:
                resp = self.session.get(url, params=params, timeout=timeout)
                resp.raise_for_status()
                time.sleep(delay)
                return resp
            except Exception as e:
                logger.warning(
                    f"Pokusaj {attempt+1}/{retries} neuspjesan: {e}"
                )
                time.sleep(delay * (attempt + 1))

        logger.error(f"Svi pokusaji neuspjesni za: {url}")
        return None

    def _aqi_formula(self, concentration, breakpoints):
        """
        Izracunava AQI prema linearnoj formuli (US EPA):
        AQI = ((I_hi - I_lo) / (BP_hi - BP_lo)) * (C - BP_lo) + I_lo
        """
        for bp_lo, bp_hi, i_lo, i_hi in breakpoints:
            if bp_lo <= concentration <= bp_hi:
                return ((i_hi - i_lo) / (bp_hi - bp_lo)) * \
                       (concentration - bp_lo) + i_lo
        return breakpoints[-1][3]

    def _calculate_aqi(self, df):
        """Izracunava ukupni AQI kao maksimum pojedinacnih AQI."""
        aqi_parts = pd.DataFrame(index=df.index)

        if "PM2.5" in df.columns:
            bp = [(0, 12, 0, 50), (12.1, 35.4, 51, 100),
                  (35.5, 55.4, 101, 150), (55.5, 150.4, 151, 200),
                  (150.5, 250.4, 201, 300), (250.5, 500, 301, 500)]
            aqi_parts["PM25_AQI"] = df["PM2.5"].apply(
                lambda x: self._aqi_formula(x, bp))

        if "PM10" in df.columns:
            bp = [(0, 54, 0, 50), (55, 154, 51, 100),
                  (155, 254, 101, 150), (255, 354, 151, 200),
                  (355, 424, 201, 300), (425, 604, 301, 500)]
            aqi_parts["PM10_AQI"] = df["PM10"].apply(
                lambda x: self._aqi_formula(x, bp))

        if "NO2" in df.columns:
            bp = [(0, 53, 0, 50), (54, 100, 51, 100),
                  (101, 360, 101, 150), (361, 649, 151, 200),
                  (650, 1249, 201, 300), (1250, 2049, 301, 500)]
            aqi_parts["NO2_AQI"] = df["NO2"].apply(
                lambda x: self._aqi_formula(x, bp))

        if aqi_parts.empty:
            return pd.Series([0] * len(df), index=df.index)

        return aqi_parts.max(axis=1).round().astype(int)

    def _get_aqi_category(self, aqi):
        """Vraca tekstualnu kategoriju za AQI vrijednost."""
        if aqi <= 50:
            return "Dobro"
        elif aqi <= 100:
            return "Umjereno"
        elif aqi <= 150:
            return "Nezdrav za osjetljive"
        elif aqi <= 200:
            return "Nezdrav"
        elif aqi <= 300:
            return "Vrlo nezdrav"
        return "Opasan"

    # --------------------------------------------------------
    # OPEN-METEO API (RADI 2024!)
    # --------------------------------------------------------

    def scrape_openmeteo_data(self):
        """
        Open-Meteo Historical Weather API - BESPLATAN, NEMA KLJUC!
        https://open-meteo.com/en/docs/historical-weather-api
        """
        logger.info("Dohvacam meteo podatke s Open-Meteo API-ja...")

        cities = {}
        for key, info in MONITORING_STATIONS.items():
            if info['city'] not in cities:
                cities[info['city']] = (info['lat'], info['lon'])

        all_data = []

        for city, (lat, lon) in cities.items():
            try:
                url = "https://archive-api.open-meteo.com/v1/archive"
                params = {
                    'latitude': lat,
                    'longitude': lon,
                    'start_date': DATA_START_DATE,
                    'end_date': DATA_END_DATE,
                    'hourly': ','.join([
                        'temperature_2m', 'relative_humidity_2m',
                        'surface_pressure', 'wind_speed_10m',
                        'wind_direction_10m', 'precipitation',
                        'cloud_cover', 'shortwave_radiation'
                    ]),
                    'timezone': 'Europe/Zagreb'
                }

                response = self._safe_request(url, params=params)

                if response and response.status_code == 200:
                    data = response.json()
                    hourly = data.get('hourly', {})

                    if 'time' in hourly:
                        df = pd.DataFrame({
                            'datetime': pd.to_datetime(hourly['time']),
                            'city': city,
                            'temperature': hourly.get('temperature_2m', []),
                            'humidity': hourly.get('relative_humidity_2m', []),
                            'pressure': hourly.get('surface_pressure', []),
                            'wind_speed': hourly.get('wind_speed_10m', []),
                            'wind_direction': hourly.get('wind_direction_10m', []),
                            'precipitation': hourly.get('precipitation', []),
                            'cloud_cover': hourly.get('cloud_cover', []),
                            'solar_radiation': hourly.get('shortwave_radiation', []),
                        })

                        all_data.append(df)
                        logger.info(f"  OK {city}: {len(df):,} zapisa")

            except Exception as e:
                logger.warning(f"  {city}: {e}")

        if all_data:
            combined = pd.concat(all_data, ignore_index=True)

            numeric_cols = [c for c in combined.columns
                          if c not in ['datetime', 'city']]

            avg_data = combined.groupby('datetime')[numeric_cols].mean().reset_index()

            out = os.path.join(RAW_DATA_DIR, "openmeteo_data.csv")
            avg_data.to_csv(out, index=False, encoding='utf-8')
            logger.info(f"  OK Spremljeno: {out} ({len(avg_data):,} zapisa)")

            return avg_data

        logger.warning("Open-Meteo nije dostupan.")
        return pd.DataFrame()

    # --------------------------------------------------------
    # OPENAQ API (RADI 2024!)
    # --------------------------------------------------------

    def scrape_openaq_data(self):
        """
        OpenAQ API - kvaliteta zraka. BESPLATAN!
        https://docs.openaq.org/
        """
        logger.info("Dohvacam podatke s OpenAQ API-ja...")

        cities = list(set(info['city'] for info in MONITORING_STATIONS.values()))
        all_data = []

        for city in cities:
            for parameter in ['pm25', 'pm10', 'no2', 'so2', 'o3', 'co']:
                try:
                    url = "https://api.openaq.org/v2/measurements"
                    params = {
                        'country': 'HR',
                        'city': city,
                        'parameter': parameter,
                        'date_from': DATA_START_DATE,
                        'date_to': DATA_END_DATE,
                        'limit': 10000,
                        'sort': 'desc'
                    }

                    response = self._safe_request(url, params=params)

                    if response and response.status_code == 200:
                        data = response.json()
                        results = data.get('results', [])

                        if results:
                            df = pd.json_normalize(results)
                            df['city'] = city
                            all_data.append(df)

                except Exception as e:
                    logger.debug(f"  {city}/{parameter}: {e}")

        if all_data:
            combined = pd.concat(all_data, ignore_index=True)
            logger.info(f"  OK OpenAQ: {len(combined):,} ukupnih mjerenja")
            return combined

        logger.warning("OpenAQ nije dostupan.")
        return pd.DataFrame()

    # --------------------------------------------------------
    # FALLBACK SCRAPERI
    # --------------------------------------------------------

    def scrape_eea_data(self):
        """Pokusava EEA API (stari, vjerojatno ne radi)."""
        logger.info("Pokusavam dohvatiti EEA podatke (fallback)...")
        try:
            url = ("https://fme.discomap.eea.europa.eu/fmedatastreaming/"
                   "AirQualityDownload/AQData_Extract.fmw")
            params = {
                "CountryCode": "HR", "Pollutant": 6001,
                "Year_from": 2022, "Year_to": 2022,
                "Source": "E1a", "Output": "TEXT", "TimeCoverage": "Year"
            }
            response = self._safe_request(url, params=params)
            if response is not None and len(response.text) > 100:
                from io import StringIO
                df = pd.read_csv(StringIO(response.text), sep=",", low_memory=False)
                logger.info(f"  OK EEA: {len(df):,} zapisa")
                return df
        except Exception as e:
            logger.warning(f"  EEA neuspjesan: {e}")

        return pd.DataFrame()

    def scrape_dhmz_meteo_data(self):
        """Pokusava DHMZ scraping (fallback)."""
        logger.info("Pokusavam DHMZ scraping (fallback)...")
        try:
            url = "https://meteo.hr/podaci.php?section=podaci_mreza"
            response = self._safe_request(url)
            if response is not None:
                try:
                    from bs4 import BeautifulSoup
                    soup = BeautifulSoup(response.content, "lxml")
                    tables = soup.find_all("table")
                    if tables:
                        dfs = pd.read_html(str(tables[0]))
                        if dfs:
                            logger.info(f"  OK DHMZ: {len(dfs[0]):,} zapisa")
                            return dfs[0]
                except ImportError:
                    logger.warning("BeautifulSoup nije instaliran")
        except Exception as e:
            logger.warning(f"  DHMZ neuspjesan: {e}")

        return pd.DataFrame()

    # --------------------------------------------------------
    # SINTETICKI PODACI (ZADNJI FALLBACK)
    # --------------------------------------------------------

    def generate_synthetic_air_quality_data(self):
        """
        Generira sinteticke podatke koji repliciraju stvarne obrasce:
        - Sezonske varijacije (zima vs ljeto)
        - Prometni vrsni sati (jutro/vecer)
        - Razlike medu gradovima (industrijski vs nekoristeni)
        - Vikend efekt (manji promet)
        """
        logger.info("Generiram sinteticke podatke o kvaliteti zraka...")
        np.random.seed(42)

        start = pd.Timestamp(DATA_START_DATE)
        end = pd.Timestamp(DATA_END_DATE)
        dates = pd.date_range(start=start, end=end, freq="h")
        n = len(dates)

        doy = dates.dayofyear.values
        hour = dates.hour.values
        dow = dates.dayofweek.values

        seasonal_winter = np.cos(2 * np.pi * (doy - 15) / 365)
        seasonal_summer = np.sin(2 * np.pi * (doy - 80) / 365)
        traffic = (np.exp(-0.5 * ((hour - 8) / 2) ** 2) +
                   np.exp(-0.5 * ((hour - 17) / 2) ** 2))
        weekend = np.where(dow >= 5, 0.7, 1.0)
        rw = np.random.randn(n).cumsum() / np.sqrt(n) * 5

        all_data = []

        for station_key, station_info in MONITORING_STATIONS.items():
            city = station_info["city"]
            uf = 1.2 if city == "Zagreb" else \
                 (1.3 if city in ["Sisak", "Kutina"] else 1.0)

            pm10 = np.clip(25 * uf + 15 * seasonal_winter +
                          5 * traffic * weekend + rw +
                          np.random.normal(0, 8, n), 2, 200)
            pm25 = np.clip(pm10 * (0.6 + 0.1 * np.random.randn(n)), 1, 150)
            no2 = np.clip(20 * uf + 10 * seasonal_winter +
                         15 * traffic * weekend +
                         np.random.normal(0, 8, n), 1, 200)
            so2 = np.clip(5 * uf + 8 * seasonal_winter +
                         np.random.normal(0, 4, n), 0.5, 100)
            o3 = np.clip(50 + 30 * seasonal_summer +
                        np.exp(-0.5 * ((hour - 14) / 4) ** 2) * 40 -
                        5 * traffic + np.random.normal(0, 15, n), 5, 250)
            co = np.clip(0.5 * uf + 0.3 * seasonal_winter +
                        0.5 * traffic * weekend +
                        np.random.normal(0, 0.2, n), 0.1, 10)

            df_station = pd.DataFrame({
                "datetime": dates,
                "station_id": station_info["id"],
                "station_name": station_info["name"],
                "city": station_info["city"],
                "latitude": station_info["lat"],
                "longitude": station_info["lon"],
                "PM10": np.round(pm10, 1),
                "PM2.5": np.round(pm25, 1),
                "NO2": np.round(no2, 1),
                "SO2": np.round(so2, 1),
                "O3": np.round(o3, 1),
                "CO": np.round(co, 2),
            })
            all_data.append(df_station)

        aq_df = pd.concat(all_data, ignore_index=True)
        aq_df["AQI"] = self._calculate_aqi(aq_df)
        aq_df["AQI_category"] = aq_df["AQI"].apply(self._get_aqi_category)

        out = os.path.join(RAW_DATA_DIR, "air_quality_data.csv")
        aq_df.to_csv(out, index=False, encoding="utf-8")
        logger.info(f"  OK Spremljeno: {out} ({len(aq_df):,} zapisa)")

        return aq_df

    def generate_synthetic_meteo_data(self):
        """Generira sinteticke meteoroloske podatke."""
        logger.info("Generiram sinteticke meteo podatke...")
        np.random.seed(43)

        start = pd.Timestamp(DATA_START_DATE)
        end = pd.Timestamp(DATA_END_DATE)
        dates = pd.date_range(start=start, end=end, freq="h")
        n = len(dates)

        doy = dates.dayofyear.values
        hour = dates.hour.values

        seasonal = np.sin(2 * np.pi * (doy - 80) / 365)
        daily_cycle = np.sin(2 * np.pi * (hour - 6) / 24)

        temperature = (12 + 12 * seasonal + 5 * daily_cycle +
                      np.random.normal(0, 2, n))
        humidity = np.clip(70 - 0.5 * (temperature - 12) +
                          np.random.normal(0, 10, n), 20, 100)
        pressure = np.clip(1013 + 10 * np.random.randn(n).cumsum() / np.sqrt(n),
                          980, 1050)
        wind_speed = np.clip(np.random.weibull(2, n) * 3, 0, 25)
        wind_direction = np.random.uniform(0, 360, n)

        rain_prob = 0.15 + 0.1 * np.cos(2 * np.pi * (doy - 100) / 365)
        rain_mask = np.random.random(n) < rain_prob
        precipitation = np.zeros(n)
        precipitation[rain_mask] = np.random.exponential(3, rain_mask.sum())

        solar = np.clip((800 + 200 * seasonal) * np.maximum(daily_cycle, 0) +
                       np.random.normal(0, 50, n), 0, 1200)
        solar[precipitation > 5] *= 0.3
        cloud_cover = np.clip(40 + 20 * np.random.randn(n) - 10 * daily_cycle,
                             0, 100)

        meteo_df = pd.DataFrame({
            "datetime": dates,
            "temperature": np.round(temperature, 1),
            "humidity": np.round(humidity, 1),
            "pressure": np.round(pressure, 1),
            "wind_speed": np.round(wind_speed, 1),
            "wind_direction": np.round(wind_direction, 0),
            "precipitation": np.round(precipitation, 1),
            "solar_radiation": np.round(solar, 0),
            "cloud_cover": np.round(cloud_cover, 0),
        })

        out = os.path.join(RAW_DATA_DIR, "synthetic_meteo_data.csv")
        meteo_df.to_csv(out, index=False, encoding="utf-8")
        logger.info(f"  OK Spremljeno: {out} ({len(meteo_df):,} zapisa)")

        return meteo_df

    # --------------------------------------------------------
    # GLAVNA METODA - HIJERARHIJSKI POKUSAJ
    # --------------------------------------------------------

    def collect_all_data(self):
        """
        Hijerarhijski pokusava prikupiti podatke:
        1. Postojece CSV datoteke (cache)
        2. Open-Meteo API (meteo) + OpenAQ (zrak)
        3. EEA + DHMZ (stari API-ji)
        4. Sinteticki podaci (zadnji fallback)
        """
        logger.info("=" * 60)
        logger.info("POKRECEM PRIKUPLJANJE PODATAKA")
        logger.info("=" * 60)

        # --- 1. KVALITETA ZRAKA ---
        aq_file = os.path.join(RAW_DATA_DIR, "air_quality_data.csv")

        if os.path.exists(aq_file):
            logger.info(f"Ucitavam cached: {aq_file}")
            air_quality_data = pd.read_csv(aq_file, parse_dates=["datetime"])
            logger.info(f"  OK {len(air_quality_data):,} zapisa")
        else:
            air_quality_data = self.scrape_openaq_data()

            if air_quality_data.empty:
                air_quality_data = self.scrape_eea_data()

            if air_quality_data.empty:
                logger.warning("Stvarni izvori nisu dostupni!")
                logger.warning("Generiram sinteticke podatke...")
                air_quality_data = self.generate_synthetic_air_quality_data()

        # --- 2. METEOROLOGIJA ---
        meteo_file_omt = os.path.join(RAW_DATA_DIR, "openmeteo_data.csv")
        meteo_file_syn = os.path.join(RAW_DATA_DIR, "synthetic_meteo_data.csv")

        if os.path.exists(meteo_file_omt):
            logger.info(f"Ucitavam cached Open-Meteo: {meteo_file_omt}")
            meteo_data = pd.read_csv(meteo_file_omt, parse_dates=["datetime"])
            logger.info(f"  OK {len(meteo_data):,} zapisa")
        elif os.path.exists(meteo_file_syn):
            logger.info(f"Ucitavam cached sinteticke: {meteo_file_syn}")
            meteo_data = pd.read_csv(meteo_file_syn, parse_dates=["datetime"])
            logger.info(f"  OK {len(meteo_data):,} zapisa")
        else:
            meteo_data = self.scrape_openmeteo_data()

            if meteo_data.empty:
                meteo_data = self.scrape_dhmz_meteo_data()

            if meteo_data.empty:
                logger.warning("Stvarni meteo izvori nisu dostupni!")
                logger.warning("Generiram sinteticke podatke...")
                meteo_data = self.generate_synthetic_meteo_data()

        logger.info("")
        logger.info("=" * 60)
        logger.info("PRIKUPLJANJE ZAVRSENO")
        logger.info(f"  Kvaliteta zraka: {len(air_quality_data):,} zapisa")
        logger.info(f"  Meteorologija:   {len(meteo_data):,} zapisa")
        logger.info("=" * 60)

        return {
            "air_quality": air_quality_data,
            "meteorology": meteo_data,
        }


# ============================================================
# POKRETANJE
# ============================================================

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  DATA SCRAPER - Predvidjanje kvalitete zraka HR")
    print("=" * 60 + "\n")

    scraper = AirQualityDataScraper()
    data = scraper.collect_all_data()

    print("\nOK Zavrseno!\n")
    for key, value in data.items():
        if isinstance(value, pd.DataFrame):
            print(f"  {key:15s}: {value.shape[0]:>8,} zapisa, "
                  f"{value.shape[1]} stupaca")