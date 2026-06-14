"""
Modul za pregled rezultata nakon treniranja modela.

Funkcionalnosti:
- Provjera svih ocekivanih izlaznih datoteka
- Prikaz usporedbe modela
- Prikaz kratkorocnih (24h) i srednjorocnih (7 dana) prognoza
- Generiranje dodatnih grafova prognoza
- Provjera spremljenih modela

ISPRAVCI U OVOJ VERZIJI:
- Dodan encoding='utf-8' pri citanju JSON datoteka
- Popravljena provjera 'R2' vs 'R²' kolone
- Oznacen ilustrativni raspon (nije pravi CI)
"""

import os
import json
import logging
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Za Windows bez GUI problema
import matplotlib.pyplot as plt

# ============================================================
# DIREKTORIJI
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
METRICS_DIR = os.path.join(BASE_DIR, 'reports', 'metrics')
FIGURES_DIR = os.path.join(BASE_DIR, 'reports', 'figures')
PREDICTIONS_DIR = os.path.join(BASE_DIR, 'data', 'predictions')
MODELS_DIR = os.path.join(BASE_DIR, 'models', 'saved')

os.makedirs(FIGURES_DIR, exist_ok=True)

logger = logging.getLogger(__name__)


# ============================================================
# FUNKCIJE ZA PREGLED
# ============================================================

def show_model_comparison():
    """Prikazuje usporedbu modela iz JSON/CSV datoteke."""
    json_file = os.path.join(METRICS_DIR, 'evaluation_results.json')
    csv_file = os.path.join(METRICS_DIR, 'model_comparison.csv')

    print("\n" + "=" * 60)
    print("  USPOREDBA MODELA")
    print("=" * 60)

    if os.path.exists(csv_file):
        df = pd.read_csv(csv_file, encoding='utf-8')
        print(df.to_string(index=False))
        print(f"\nNajbolji model: {df.iloc[0]['Model']}")

    elif os.path.exists(json_file):
        with open(json_file, 'r', encoding='utf-8') as f:
            results = json.load(f)
        for model, metrics in results.items():
            print(f"\n  {model}:")
            for k, v in metrics.items():
                if not isinstance(v, dict):
                    print(f"    {k:15s}: {v:.6f}")
    else:
        print("  Nema rezultata evaluacije.")
        print(f"  Ocekivana lokacija: {json_file}")


def show_predictions():
    """Prikazuje kratkorocne i srednjorocne predikcije."""
    print("\n" + "=" * 60)
    print("  PREDIKCIJE")
    print("=" * 60)

    # Kratkorocna prognoza
    short_file = os.path.join(PREDICTIONS_DIR, 'forecast_24h.csv')
    if os.path.exists(short_file):
        df = pd.read_csv(short_file, encoding='utf-8')
        print("\n  Kratkorocna prognoza PM2.5 (sljedecih 24h):")
        print(f"  {'Sat':>5} | {'PM2.5 (ug/m3)':>15}")
        print("  " + "-" * 25)
        for _, row in df.iterrows():
            print(f"  t+{int(row['hour_ahead']):02d}h  | "
                  f"{row['predicted_value']:>12.2f}")
    else:
        print("  Nema kratkorocne prognoze.")

    # Srednjorocna prognoza
    medium_file = os.path.join(PREDICTIONS_DIR, 'forecast_7days.csv')
    if os.path.exists(medium_file):
        df = pd.read_csv(medium_file, encoding='utf-8')
        print("\n  Srednjorocna prognoza PM2.5 (7 dana):")
        print(f"  {'Dan':>5} | {'Prosjek':>10} | {'Max':>10} | {'Min':>10}")
        print("  " + "-" * 45)
        for day in range(1, 8):
            day_data = df[df['day'] == day]['predicted_value']
            if not day_data.empty:
                print(f"  Dan {day}  | "
                      f"{day_data.mean():>10.2f} | "
                      f"{day_data.max():>10.2f} | "
                      f"{day_data.min():>10.2f}")
    else:
        print("  Nema srednjorocne prognoze.")


def show_saved_models():
    """Prikazuje koji modeli su spremljeni."""
    print("\n" + "=" * 60)
    print("  SPREMLJENI MODELI")
    print("=" * 60)

    model_files = [
        'lstm_model.keras',
        'gru_model.keras',
        'hybrid_model.keras'
    ]

    for mf in model_files:
        path = os.path.join(MODELS_DIR, mf)
        if os.path.exists(path):
            size_mb = os.path.getsize(path) / (1024 * 1024)
            print(f"  OK  {mf:30s} ({size_mb:.1f} MB)")
        else:
            print(f"  --  {mf:30s} (nije pronadjen)")


def plot_forecast_24h():
    """Crta graf kratkorocne prognoze."""
    short_file = os.path.join(PREDICTIONS_DIR, 'forecast_24h.csv')
    if not os.path.exists(short_file):
        return

    df = pd.read_csv(short_file, encoding='utf-8')

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(df['hour_ahead'], df['predicted_value'],
            'b-o', linewidth=2, markersize=5, label='Prognoza PM2.5')

    ax.axhline(y=25, color='orange', linestyle='--',
               linewidth=2, label='WHO limit (25 ug/m3)')
    ax.axhline(y=50, color='red', linestyle='--',
               linewidth=2, label='EU limit (50 ug/m3)')

    # NAPOMENA: Ovo NIJE pravi 95% interval pouzdanosti,
    # vec ilustrativni raspon +-15% za vizualnu referencu
    ax.fill_between(df['hour_ahead'],
                    df['predicted_value'] * 0.85,
                    df['predicted_value'] * 1.15,
                    alpha=0.2, color='blue',
                    label='Ilustrativni raspon (+-15%)')

    ax.set_title('Kratkorocna prognoza kvalitete zraka (24h)',
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('Sati unaprijed')
    ax.set_ylabel('PM2.5 (ug/m3)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    out = os.path.join(FIGURES_DIR, 'forecast_24h.png')
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Graf 24h prognoza: {out}")


def plot_forecast_7days():
    """Crta graf srednjorocne prognoze."""
    medium_file = os.path.join(PREDICTIONS_DIR, 'forecast_7days.csv')
    if not os.path.exists(medium_file):
        return

    df = pd.read_csv(medium_file, encoding='utf-8')

    daily = df.groupby('day')['predicted_value'].agg(
        ['mean', 'max', 'min']
    ).reset_index()

    fig, ax = plt.subplots(figsize=(12, 5))
    days = daily['day'].values

    ax.plot(days, daily['mean'], 'b-o',
            linewidth=2, markersize=8, label='Dnevni prosjek')
    ax.fill_between(days, daily['min'], daily['max'],
                    alpha=0.2, color='blue', label='Min-Max raspon')

    ax.axhline(y=25, color='orange', linestyle='--',
               linewidth=2, label='WHO limit (25 ug/m3)')
    ax.axhline(y=50, color='red', linestyle='--',
               linewidth=2, label='EU limit (50 ug/m3)')

    ax.set_title('Srednjorocna prognoza kvalitete zraka (7 dana)',
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('Dan')
    ax.set_ylabel('PM2.5 (ug/m3)')
    ax.set_xticks(days)
    ax.set_xticklabels([f'Dan {d}' for d in days])
    ax.legend()
    ax.grid(True, alpha=0.3)

    out = os.path.join(FIGURES_DIR, 'forecast_7days.png')
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Graf 7-dnevna prognoza: {out}")


def plot_model_comparison_bar():
    """Crta stupcasti dijagram usporedbe modela."""
    csv_file = os.path.join(METRICS_DIR, 'model_comparison.csv')

    if not os.path.exists(csv_file):
        print("  Nema podataka za graf usporedbe modela.")
        return

    df = pd.read_csv(csv_file, encoding='utf-8')

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('Usporedba modela', fontsize=16, fontweight='bold')

    # POPRAVAK: Provjeri obje varijante imena R kvadrat
    r2_col = None
    if 'R2' in df.columns:
        r2_col = 'R2'
    elif 'R²' in df.columns:
        r2_col = 'R²'

    metrics_to_plot = []
    if 'RMSE' in df.columns:
        metrics_to_plot.append('RMSE')
    if 'MAE' in df.columns:
        metrics_to_plot.append('MAE')
    if r2_col:
        metrics_to_plot.append(r2_col)
    elif 'MAPE (%)' in df.columns:
        metrics_to_plot.append('MAPE (%)')

    colors = ['#2196F3', '#4CAF50', '#FF9800']

    for ax, metric, color in zip(axes, metrics_to_plot[:3], colors):
        bars = ax.bar(df['Model'], df[metric],
                      color=color, alpha=0.8, edgecolor='black')
        ax.set_title(metric, fontsize=13)
        ax.set_ylabel(metric)
        for bar, val in zip(bars, df[metric]):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() * 1.02,
                    f'{val:.4f}', ha='center', va='bottom', fontsize=10)
        ax.tick_params(axis='x', rotation=15)

    plt.tight_layout()
    out = os.path.join(FIGURES_DIR, 'model_comparison.png')
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Graf usporedbe modela: {out}")


def plot_all_forecasts():
    """Crta sve grafove prognoza."""
    print("\n" + "=" * 60)
    print("  CRTANJE GRAFOVA")
    print("=" * 60)

    plot_forecast_24h()
    plot_forecast_7days()
    plot_model_comparison_bar()


def check_all_files():
    """Provjera svih ocekivanih izlaznih datoteka."""
    print("\n" + "=" * 60)
    print("  PROVJERA IZLAZNIH DATOTEKA")
    print("=" * 60)

    expected_files = {
        'Podaci - kvaliteta zraka': 'data/raw/air_quality_data.csv',
        'Podaci - meteorologija':   'data/raw/synthetic_meteo_data.csv',
        'Open-Meteo (opcionalno)':  'data/raw/openmeteo_data.csv',
        'Obradjeni podaci':         'data/processed/processed_data.csv',
        'Scaleri':                  'data/processed/scalers.pkl',
        'LSTM model':               'models/saved/lstm_model.keras',
        'GRU model':                'models/saved/gru_model.keras',
        'Hybrid model':             'models/saved/hybrid_model.keras',
        'Prognoza 24h':             'data/predictions/forecast_24h.csv',
        'Prognoza 7 dana':          'data/predictions/forecast_7days.csv',
        'Rezultati evaluacije':     'reports/metrics/evaluation_results.json',
        'Usporedba modela':         'reports/metrics/model_comparison.csv',
    }

    ok_count = 0
    missing = []

    for name, rel_path in expected_files.items():
        full_path = os.path.join(BASE_DIR, rel_path)
        if os.path.exists(full_path):
            size = os.path.getsize(full_path)
            print(f"  OK  {name:30s} ({size:,} bytes)")
            ok_count += 1
        else:
            print(f"  --  {name:30s} NEDOSTAJE")
            missing.append(name)

    print(f"\n  Pronadjeno: {ok_count}/{len(expected_files)} datoteka")

    if missing:
        print("\n  Nedostajuce datoteke:")
        for m in missing:
            print(f"    - {m}")
    else:
        print("\n  Sve datoteke su prisutne!")

    return ok_count, missing


def main():
    """Glavna funkcija - prikazuje sve rezultate."""
    print("\n" + "=" * 60)
    print("  PREGLED REZULTATA - Predvidjanje kvalitete zraka HR")
    print("=" * 60)

    # 1. Provjera datoteka
    ok_count, missing = check_all_files()

    # 2. Usporedba modela
    show_model_comparison()

    # 3. Predikcije
    show_predictions()

    # 4. Spremljeni modeli
    show_saved_models()

    # 5. Crtanje grafova
    plot_all_forecasts()

    print("\n" + "=" * 60)
    print("  PREGLED ZAVRSEN")
    print(f"  Grafovi su spremljeni u: reports/figures/")
    print("=" * 60)


if __name__ == '__main__':
    main()