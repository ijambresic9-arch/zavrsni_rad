"""
Modul za evaluaciju i usporedbu modela.

Metrike:
- MSE (Mean Squared Error)
- RMSE (Root Mean Squared Error)
- MAE (Mean Absolute Error)
- R^2 (Coefficient of Determination)
- MAPE (Mean Absolute Percentage Error)
- Metrike po vremenskom horizontu

ISPRAVCI U OVOJ VERZIJI:
- Try/except za import config-a (radi i bez config.py)
- Popravljen by_horizon (sada se ispravno racuna)
- Dodan encoding='utf-8' pri JSON pisanju
- save_results() sada sprema i by_horizon podatke
"""

import os
import sys
import json
import logging
import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score
)

# ============================================================
# KONFIGURACIJA (s fallback)
# ============================================================
try:
    from config import METRICS_DIR
except ImportError:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    METRICS_DIR = os.path.join(BASE_DIR, 'reports', 'metrics')
    os.makedirs(METRICS_DIR, exist_ok=True)

logger = logging.getLogger(__name__)


# ============================================================
# KLASA ZA EVALUACIJU
# ============================================================

class ModelEvaluator:
    """
    Klasa za evaluaciju i usporedbu modela.

    Sprema rezultate u JSON i CSV formatu radi
    kasnijeg pregleda i analize.
    """

    def __init__(self):
        self.results = {}
        self.comparison = None

    def _make_serializable(self, obj):
        """
        Rekurzivno pretvara numpy tipove u Python tipove.
        Potrebno za JSON serializaciju.
        """
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._make_serializable(v) for v in obj]
        elif isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    def evaluate_model(self, model_name, y_true, y_pred, scaler=None):
        """
        Evaluira jedan model na test skupu.

        Args:
            model_name: ime modela (npr. 'LSTM', 'GRU', 'Hybrid')
            y_true: prave vrijednosti (2D: samples x horizon)
            y_pred: predikcije (2D: samples x horizon)
            scaler: opcionalni scaler za denormalizaciju

        Returns:
            dict s metrikama
        """
        logger.info(f"Evaluiram model: {model_name}")

        # Sacuvaj 2D oblike za by-horizon metrike
        y_true_2d = y_true if y_true.ndim == 2 else y_true.reshape(-1, 1)
        y_pred_2d = y_pred if y_pred.ndim == 2 else y_pred.reshape(-1, 1)

        # Flatten za skalarne metrike
        y_t_flat = y_true_2d.flatten()
        y_p_flat = y_pred_2d.flatten()

        # Denormalizacija ako je dostupan scaler
        if scaler is not None:
            try:
                y_true_orig = scaler.inverse_transform(
                    y_t_flat.reshape(-1, 1)
                ).flatten()
                y_pred_orig = scaler.inverse_transform(
                    y_p_flat.reshape(-1, 1)
                ).flatten()
            except Exception as e:
                logger.warning(f"Denormalizacija neuspjesna: {e}")
                y_true_orig = y_t_flat
                y_pred_orig = y_p_flat
        else:
            y_true_orig = y_t_flat
            y_pred_orig = y_p_flat

        metrics = {}

        # Normalizirane metrike
        metrics['MSE'] = float(mean_squared_error(y_t_flat, y_p_flat))
        metrics['RMSE'] = float(np.sqrt(metrics['MSE']))
        metrics['MAE'] = float(mean_absolute_error(y_t_flat, y_p_flat))
        metrics['R2'] = float(r2_score(y_t_flat, y_p_flat))

        # MAPE (izbjegni dijeljenje s nulom)
        mask = np.abs(y_t_flat) > 1e-8
        if mask.sum() > 0:
            metrics['MAPE'] = float(
                np.mean(np.abs(
                    (y_t_flat[mask] - y_p_flat[mask]) / y_t_flat[mask]
                )) * 100
            )
        else:
            metrics['MAPE'] = float('inf')

        # Metrike u originalnim jedinicama
        metrics['MSE_original'] = float(
            mean_squared_error(y_true_orig, y_pred_orig)
        )
        metrics['RMSE_original'] = float(np.sqrt(metrics['MSE_original']))
        metrics['MAE_original'] = float(
            mean_absolute_error(y_true_orig, y_pred_orig)
        )

        # ISPRAVAK: By-horizon metrike (sada se racunaju ispravno)
        n_steps = y_true_2d.shape[1]
        if n_steps > 1:
            horizon_metrics = {}
            for h in range(n_steps):
                h_mse = float(mean_squared_error(
                    y_true_2d[:, h], y_pred_2d[:, h]
                ))
                horizon_metrics[f'hour_{h+1}'] = {
                    'MSE': h_mse,
                    'RMSE': float(np.sqrt(h_mse)),
                    'MAE': float(mean_absolute_error(
                        y_true_2d[:, h], y_pred_2d[:, h]
                    ))
                }
            metrics['by_horizon'] = horizon_metrics

            # Prvih 6 horizonta u log
            logger.info("  MAE po horizontu:")
            for h in range(min(6, n_steps)):
                logger.info(
                    f"    t+{h+1:02d}h: "
                    f"MAE={horizon_metrics[f'hour_{h+1}']['MAE']:.6f}"
                )

        self.results[model_name] = metrics

        # Ispis
        logger.info(f"\n  Rezultati za {model_name}:")
        logger.info(f"    MSE  (norm):  {metrics['MSE']:.6f}")
        logger.info(f"    RMSE (norm):  {metrics['RMSE']:.6f}")
        logger.info(f"    MAE  (norm):  {metrics['MAE']:.6f}")
        logger.info(f"    R^2:          {metrics['R2']:.4f}")
        logger.info(f"    MAPE:         {metrics['MAPE']:.2f}%")
        logger.info(f"    RMSE (orig):  {metrics['RMSE_original']:.2f} ug/m3")

        return metrics

    def compare_models(self):
        """
        Usporeduje sve evaluirane modele.

        Returns:
            DataFrame sortiran po RMSE (manji = bolji)
        """
        if not self.results:
            logger.warning("Nema rezultata za usporedbu!")
            return None

        comparison_data = []
        for model_name, metrics in self.results.items():
            row = {
                'Model': model_name,
                'MSE': metrics['MSE'],
                'RMSE': metrics['RMSE'],
                'MAE': metrics['MAE'],
                'R2': metrics['R2'],
                'MAPE (%)': metrics['MAPE'],
                'RMSE (orig.)': metrics.get('RMSE_original', 0)
            }
            comparison_data.append(row)

        self.comparison = pd.DataFrame(comparison_data)
        self.comparison = self.comparison.sort_values('RMSE')

        logger.info("\n" + "=" * 80)
        logger.info("USPOREDBA MODELA")
        logger.info("=" * 80)
        logger.info(f"\n{self.comparison.to_string(index=False)}")

        best_model = self.comparison.iloc[0]['Model']
        logger.info(f"\nNajbolji model: {best_model}")

        return self.comparison

    def save_results(self):
        """
        Sprema rezultate u JSON i CSV format.

        ISPRAVAK: Sada koristi _make_serializable() koji
        rekurzivno obraduje sve nivoe (ukljucujuci by_horizon).
        """
        # JSON (sa svim podacima, ukljucujuci by_horizon)
        json_file = os.path.join(METRICS_DIR, 'evaluation_results.json')
        serializable = self._make_serializable(self.results)

        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(serializable, f, indent=2, ensure_ascii=False)

        logger.info(f"  OK Rezultati spremljeni: {json_file}")

        # CSV
        if self.comparison is not None:
            csv_file = os.path.join(METRICS_DIR, 'model_comparison.csv')
            self.comparison.to_csv(csv_file, index=False, encoding='utf-8')
            logger.info(f"  OK Usporedba spremljena: {csv_file}")

        return serializable


# ============================================================
# TESTIRANJE
# ============================================================

if __name__ == '__main__':
    print("Test ModelEvaluator-a...")

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )

    # Dummy podaci
    np.random.seed(42)
    n_samples = 100
    horizon = 24

    y_true = np.random.randn(n_samples, horizon)
    y_pred_lstm = y_true + np.random.randn(n_samples, horizon) * 0.3
    y_pred_gru = y_true + np.random.randn(n_samples, horizon) * 0.4
    y_pred_hybrid = y_true + np.random.randn(n_samples, horizon) * 0.25

    evaluator = ModelEvaluator()
    evaluator.evaluate_model('LSTM', y_true, y_pred_lstm)
    evaluator.evaluate_model('GRU', y_true, y_pred_gru)
    evaluator.evaluate_model('Hybrid', y_true, y_pred_hybrid)

    comparison = evaluator.compare_models()
    evaluator.save_results()

    print("\nModelEvaluator test prosao uspjesno!")