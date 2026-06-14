"""
Modul za sve vizualizacije u projektu.

Generira:
- Pregled sirovih podataka
- Sezonske obrasce (mjesec, sat, dan u tjednu)
- Korelacijsku matricu
- Povijest treniranja (loss, MAE)
- Predikcije vs stvarne vrijednosti
- Usporedbu modela
- Prognoze (24h, 7 dana)

ISPRAVCI U OVOJ VERZIJI:
- Try/except za import config-a
- Provjera 'mae' kljuca u history (fallback)
- Robusnija korelacijska matrica
- Bolje rukovanje praznim podacima
"""

import os
import logging
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Za Windows bez GUI problema
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================
# KONFIGURACIJA (s fallback)
# ============================================================
try:
    from config import FIGURES_DIR, AQI_CATEGORIES, POLLUTANTS
except ImportError:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    FIGURES_DIR = os.path.join(BASE_DIR, 'reports', 'figures')
    os.makedirs(FIGURES_DIR, exist_ok=True)

    AQI_CATEGORIES = {
        'Dobro':    {'range': (0, 50),    'color': '#00e400'},
        'Umjereno': {'range': (51, 100),  'color': '#ffff00'},
        'Nezdrav':  {'range': (101, 200), 'color': '#ff0000'},
    }

    POLLUTANTS = {
        'PM10':  {'name': 'Cestice PM10',  'unit': 'ug/m3', 'eu_limit_24h': 50},
        'PM2.5': {'name': 'Cestice PM2.5', 'unit': 'ug/m3'},
        'NO2':   {'name': 'Dusikov dioksid', 'unit': 'ug/m3', 'eu_limit_24h': 200},
        'SO2':   {'name': 'Sumporov dioksid', 'unit': 'ug/m3'},
        'O3':    {'name': 'Ozon', 'unit': 'ug/m3'},
        'CO':    {'name': 'Ugljikov monoksid', 'unit': 'mg/m3'},
    }

logger = logging.getLogger(__name__)

# Postavi stil
try:
    plt.style.use('seaborn-v0_8-whitegrid')
except Exception:
    plt.style.use('default')

sns.set_palette('husl')


# ============================================================
# KLASA ZA VIZUALIZACIJU
# ============================================================

class Visualizer:
    """Klasa za sve vizualizacije u projektu."""

    def __init__(self):
        self.fig_count = 0

    def _save_figure(self, fig, name):
        """Sprema figuru u PNG i PDF format."""
        png_path = os.path.join(FIGURES_DIR, f'{name}.png')
        pdf_path = os.path.join(FIGURES_DIR, f'{name}.pdf')
        fig.savefig(png_path, dpi=150, bbox_inches='tight')
        try:
            fig.savefig(pdf_path, bbox_inches='tight')
        except Exception as e:
            logger.warning(f"PDF spremanje neuspjesno za {name}: {e}")
        plt.close(fig)
        logger.info(f"  OK Grafikon spremljen: {png_path}")

    def plot_raw_data_overview(self, df, station_name=None):
        """Prikazuje pregled sirovih podataka za sve zagadivace."""
        pollutant_cols = [col for col in POLLUTANTS.keys() if col in df.columns]
        n_plots = len(pollutant_cols)

        if n_plots == 0:
            logger.warning("Nema stupaca zagadivaca za vizualizaciju!")
            return

        fig, axes = plt.subplots(n_plots, 1, figsize=(16, 3 * n_plots),
                                 sharex=True)
        if n_plots == 1:
            axes = [axes]

        title = 'Pregled podataka o kvaliteti zraka'
        if station_name:
            title += f' - {station_name}'
        fig.suptitle(title, fontsize=16, fontweight='bold')

        for ax, col in zip(axes, pollutant_cols):
            if 'datetime' in df.columns:
                ax.plot(df['datetime'], df[col], linewidth=0.5, alpha=0.7)
            else:
                ax.plot(df[col].values, linewidth=0.5, alpha=0.7)

            ax.set_ylabel(f'{col} ({POLLUTANTS[col]["unit"]})', fontsize=10)
            ax.set_title(POLLUTANTS[col]['name'], fontsize=11, loc='left')

            # EU limit
            if 'eu_limit_24h' in POLLUTANTS[col]:
                ax.axhline(y=POLLUTANTS[col]['eu_limit_24h'], color='red',
                          linestyle='--', linewidth=1, label='EU limit (24h)')
                ax.legend(loc='upper right', fontsize=8)

        axes[-1].set_xlabel('Datum', fontsize=12)
        plt.tight_layout()
        self._save_figure(fig, 'raw_data_overview')

    def plot_training_history(self, histories, model_names):
        """Prikazuje povijest treniranja (loss + MAE)."""
        n_models = len(histories)
        if n_models == 0:
            logger.warning("Nema histories za prikaz!")
            return

        fig, axes = plt.subplots(n_models, 2, figsize=(16, 5 * n_models))

        if n_models == 1:
            axes = axes.reshape(1, -1)

        fig.suptitle('Povijest treniranja modela', fontsize=16, fontweight='bold')

        for i, (history, name) in enumerate(zip(histories, model_names)):
            h = history.history

            # Loss
            if 'loss' in h:
                axes[i, 0].plot(h['loss'], label='Trening', linewidth=2)
            if 'val_loss' in h:
                axes[i, 0].plot(h['val_loss'], label='Validacija', linewidth=2)
            axes[i, 0].set_title(f'{name} - Gubitak (MSE)', fontsize=12)
            axes[i, 0].set_xlabel('Epoha')
            axes[i, 0].set_ylabel('MSE')
            axes[i, 0].legend()
            axes[i, 0].set_yscale('log')

            # MAE - provjeri koji kljuc postoji
            mae_key = None
            val_mae_key = None
            for k in ['mae', 'mean_absolute_error']:
                if k in h:
                    mae_key = k
                if f'val_{k}' in h:
                    val_mae_key = f'val_{k}'

            if mae_key:
                axes[i, 1].plot(h[mae_key], label='Trening', linewidth=2)
            if val_mae_key:
                axes[i, 1].plot(h[val_mae_key], label='Validacija', linewidth=2)

            if mae_key or val_mae_key:
                axes[i, 1].set_title(f'{name} - MAE', fontsize=12)
                axes[i, 1].set_xlabel('Epoha')
                axes[i, 1].set_ylabel('MAE')
                axes[i, 1].legend()
            else:
                axes[i, 1].text(0.5, 0.5, 'MAE nije dostupan',
                               transform=axes[i, 1].transAxes,
                               ha='center', va='center')

        plt.tight_layout()
        self._save_figure(fig, 'training_history')

    def plot_predictions_vs_actual(self, y_true, y_pred, model_name,
                                    target_col='PM2.5', n_samples=500):
        """Prikazuje usporedbu predikcija i stvarnih vrijednosti."""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(
            f'{model_name} - Predikcije vs. Stvarne vrijednosti ({target_col})',
            fontsize=16, fontweight='bold'
        )

        # Uzmi prvu predikciju (t+1)
        if y_true.ndim > 1:
            y_t = y_true[:n_samples, 0]
            y_p = y_pred[:n_samples, 0]
        else:
            y_t = y_true[:n_samples]
            y_p = y_pred[:n_samples]

        # 1. Vremenska usporedba
        axes[0, 0].plot(y_t, label='Stvarno', alpha=0.8, linewidth=1)
        axes[0, 0].plot(y_p, label='Predvideno', alpha=0.8, linewidth=1)
        axes[0, 0].set_title('Vremenska usporedba (t+1)')
        axes[0, 0].set_xlabel('Uzorak')
        axes[0, 0].set_ylabel(target_col)
        axes[0, 0].legend()

        # 2. Scatter plot
        axes[0, 1].scatter(y_t, y_p, alpha=0.3, s=10, color='steelblue')
        min_val = float(min(y_t.min(), y_p.min()))
        max_val = float(max(y_t.max(), y_p.max()))
        axes[0, 1].plot([min_val, max_val], [min_val, max_val], 'r--',
                       linewidth=2, label='Savrsena predikcija')
        axes[0, 1].set_title('Scatter: Stvarno vs. Predvideno')
        axes[0, 1].set_xlabel('Stvarne vrijednosti')
        axes[0, 1].set_ylabel('Predvidene vrijednosti')
        axes[0, 1].legend()

        # 3. Distribucija gresaka
        errors = y_t - y_p
        axes[1, 0].hist(errors, bins=50, edgecolor='black', alpha=0.7,
                       color='steelblue')
        axes[1, 0].axvline(x=0, color='red', linestyle='--', linewidth=2)
        axes[1, 0].set_title(
            f'Distribucija gresaka (mean={errors.mean():.4f}, '
            f'std={errors.std():.4f})'
        )
        axes[1, 0].set_xlabel('Greska (stvarno - predvideno)')
        axes[1, 0].set_ylabel('Frekvencija')

        # 4. Greska po vremenskom horizontu
        if y_true.ndim > 1 and y_true.shape[1] > 1:
            n_horizons = min(y_true.shape[1], 24)
            horizons = list(range(1, n_horizons + 1))
            mae_by_horizon = []
            for h in range(n_horizons):
                mae = np.mean(np.abs(y_true[:, h] - y_pred[:, h]))
                mae_by_horizon.append(mae)

            axes[1, 1].bar(horizons, mae_by_horizon,
                          color='steelblue', alpha=0.8)
            axes[1, 1].set_title('MAE po vremenskom horizontu')
            axes[1, 1].set_xlabel('Sati unaprijed')
            axes[1, 1].set_ylabel('MAE')

        plt.tight_layout()
        safe_name = model_name.lower().replace(' ', '_').replace('-', '_')
        self._save_figure(fig, f'predictions_{safe_name}')

    def plot_model_comparison(self, comparison_df):
        """Vizualizira usporedbu modela."""
        if comparison_df is None or comparison_df.empty:
            logger.warning("Nema podataka za usporedbu modela!")
            return

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle('Usporedba modela', fontsize=16, fontweight='bold')

        # Prilagodi imena metrika
        r2_col = 'R2' if 'R2' in comparison_df.columns else \
                 ('R²' if 'R²' in comparison_df.columns else None)

        metrics_to_plot = []
        if 'RMSE' in comparison_df.columns:
            metrics_to_plot.append('RMSE')
        if 'MAE' in comparison_df.columns:
            metrics_to_plot.append('MAE')
        if r2_col:
            metrics_to_plot.append(r2_col)
        elif 'MAPE (%)' in comparison_df.columns:
            metrics_to_plot.append('MAPE (%)')

        colors = ['#2196F3', '#4CAF50', '#FF9800']

        for ax, metric, color in zip(axes, metrics_to_plot[:3], colors):
            bars = ax.bar(comparison_df['Model'], comparison_df[metric],
                         color=color, alpha=0.8, edgecolor='black')
            ax.set_title(metric, fontsize=14)
            ax.set_ylabel(metric)

            # Vrijednosti na stupcima
            for bar, val in zip(bars, comparison_df[metric]):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                       f'{val:.4f}', ha='center', va='bottom', fontsize=10)

            ax.tick_params(axis='x', rotation=45)

        plt.tight_layout()
        self._save_figure(fig, 'model_comparison')

    def plot_forecast(self, forecast_values, dates=None, target_col='PM2.5',
                      confidence=None, title=None):
        """Prikazuje prognoziranu kvalitetu zraka."""
        fig, ax = plt.subplots(figsize=(14, 6))

        if dates is None:
            dates = list(range(len(forecast_values)))

        ax.plot(dates, forecast_values, 'b-o', linewidth=2, markersize=4,
               label='Prognoza')

        if confidence is not None:
            ax.fill_between(dates, confidence[0], confidence[1],
                          alpha=0.2, color='blue',
                          label='95% interval pouzdanosti')

        # EU limiti
        if target_col in POLLUTANTS and 'eu_limit_24h' in POLLUTANTS[target_col]:
            limit = POLLUTANTS[target_col]['eu_limit_24h']
            unit = POLLUTANTS[target_col]['unit']
            ax.axhline(y=limit, color='red', linestyle='--', linewidth=2,
                      label=f'EU limit (24h): {limit} {unit}')

        if title is None:
            name = POLLUTANTS.get(target_col, {}).get('name', target_col)
            title = f'Prognoza {name}'

        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('Sati unaprijed')
        unit = POLLUTANTS.get(target_col, {}).get('unit', '')
        ax.set_ylabel(f'{target_col} ({unit})')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        safe_name = target_col.lower().replace('.', '')
        # Generiraj jedinstveno ime na temelju naslova
        if 'Kratkorocna' in (title or '') or '24h' in (title or ''):
            suffix = '_24h'
        elif 'Srednjorocna' in (title or '') or '7' in (title or ''):
            suffix = '_7days'
        else:
            suffix = ''
        self._save_figure(fig, f'forecast_{safe_name}{suffix}')

    def plot_seasonal_patterns(self, df, target_col='PM2.5'):
        """Vizualizira sezonske obrasce u podacima."""
        if 'datetime' not in df.columns:
            logger.warning("Nema datetime stupca za sezonsku analizu!")
            return

        if target_col not in df.columns:
            logger.warning(f"Stupac {target_col} ne postoji!")
            return

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'Sezonski obrasci - {target_col}',
                    fontsize=16, fontweight='bold')

        dt = pd.to_datetime(df['datetime'])

        # 1. Po mjesecima
        monthly = df.groupby(dt.dt.month)[target_col].agg(['mean', 'std'])
        months = ['Sij', 'Velj', 'Ozu', 'Tra', 'Svi', 'Lip',
                  'Srp', 'Kol', 'Ruj', 'Lis', 'Stu', 'Pro']
        axes[0, 0].bar(months[:len(monthly)], monthly['mean'],
                      yerr=monthly['std'], color='steelblue',
                      alpha=0.8, capsize=3)
        axes[0, 0].set_title('Mjesecni prosjek')
        axes[0, 0].set_ylabel(target_col)
        axes[0, 0].tick_params(axis='x', rotation=45)

        # 2. Po satu u danu
        hourly = df.groupby(dt.dt.hour)[target_col].agg(['mean', 'std'])
        axes[0, 1].plot(hourly.index, hourly['mean'], 'b-o', linewidth=2)
        axes[0, 1].fill_between(hourly.index,
                               hourly['mean'] - hourly['std'],
                               hourly['mean'] + hourly['std'],
                               alpha=0.2)
        axes[0, 1].set_title('Dnevni obrazac')
        axes[0, 1].set_xlabel('Sat')
        axes[0, 1].set_ylabel(target_col)

        # 3. Po danu u tjednu
        days = ['Pon', 'Uto', 'Sri', 'Cet', 'Pet', 'Sub', 'Ned']
        daily = df.groupby(dt.dt.dayofweek)[target_col].agg(['mean', 'std'])
        axes[1, 0].bar(days[:len(daily)], daily['mean'],
                      yerr=daily['std'], color='coral',
                      alpha=0.8, capsize=3)
        axes[1, 0].set_title('Tjedni obrazac')
        axes[1, 0].set_ylabel(target_col)

        # 4. Heatmap: sat x mjesec
        try:
            df_temp = df.copy()
            df_temp['hour'] = dt.dt.hour
            df_temp['month'] = dt.dt.month
            pivot = df_temp.pivot_table(values=target_col, index='hour',
                                       columns='month', aggfunc='mean')
            sns.heatmap(pivot, ax=axes[1, 1], cmap='YlOrRd', annot=False)
            axes[1, 1].set_title('Toplinska karta: Sat x Mjesec')
            axes[1, 1].set_ylabel('Sat')
            axes[1, 1].set_xlabel('Mjesec')
        except Exception as e:
            logger.warning(f"Heatmap neuspjesan: {e}")

        plt.tight_layout()
        safe_name = target_col.lower().replace('.', '')
        self._save_figure(fig, f'seasonal_patterns_{safe_name}')

    def plot_correlation_matrix(self, df):
        """Prikazuje korelacijsku matricu između varijabli."""
        # Odaberi samo osnovne stupce (bez lag i ciklickih)
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        # Filtriraj na osnovne stupce
        basic_cols = []
        for col in numeric_cols:
            # Preskoci lag, rolling, diff i ciklicke stupce
            if any(x in col for x in ['_lag_', '_mean_', '_std_', '_max_',
                                        '_diff_', '_sin', '_cos']):
                continue
            # Preskoci kategoricke
            if col in ['hour', 'day_of_week', 'day_of_year', 'month',
                      'week_of_year', 'season', 'is_weekend',
                      'is_rush_hour', 'is_night']:
                continue
            basic_cols.append(col)

        # Ogranici na maksimum 15 stupaca
        basic_cols = basic_cols[:15]

        if len(basic_cols) < 2:
            logger.warning("Nedovoljno stupaca za korelacijsku matricu!")
            return

        corr = df[basic_cols].corr()

        fig, ax = plt.subplots(figsize=(14, 12))
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, mask=mask, annot=True, fmt='.2f',
                   cmap='RdBu_r', center=0, ax=ax,
                   square=True, linewidths=0.5,
                   cbar_kws={'shrink': 0.8})
        ax.set_title('Korelacijska matrica', fontsize=16, fontweight='bold')

        plt.tight_layout()
        self._save_figure(fig, 'correlation_matrix')

    def create_all_visualizations(self, df, histories, model_names,
                                  predictions, y_test, comparison_df,
                                  target_col='PM2.5'):
        """Kreira sve vizualizacije."""
        logger.info("Kreiram vizualizacije...")

        self.plot_raw_data_overview(df)
        self.plot_seasonal_patterns(df, target_col)
        self.plot_correlation_matrix(df)

        if histories and model_names:
            self.plot_training_history(histories, model_names)

        for name, pred in predictions.items():
            self.plot_predictions_vs_actual(y_test, pred, name, target_col)

        if comparison_df is not None:
            self.plot_model_comparison(comparison_df)

        logger.info("OK Sve vizualizacije kreirane")