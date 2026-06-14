"""
Hibridni LSTM-GRU model za predvidjanje kvalitete zraka.

Arhitektura:
- DVIJE PARALELNE GRANE:
  1. LSTM grana (sa Attention)
  2. GRU grana (sa Attention)
- Spajanje (Concatenate) izlaza obje grane
- Dense slojevi za finalnu predikciju

Prednosti hibrida:
- LSTM je bolji za dugotrajne ovisnosti
- GRU je brzi i ucinkovitiji
- Spajanjem dobivamo kombinirane prednosti
- Attention u obje grane fokusira se na razlicite aspekte

ISPRAVAK U OVOJ VERZIJI:
- Dodan Attention sloj u OBJE grane (prije nije postojao)
- Bolja regularizacija
"""

import os
import sys
import logging
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, LSTM, GRU, Dense, Dropout,
    Bidirectional, BatchNormalization, Concatenate
)
from tensorflow.keras.callbacks import (
    EarlyStopping, ModelCheckpoint,
    ReduceLROnPlateau, TensorBoard
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l1_l2

# ============================================================
# KONFIGURACIJA
# ============================================================
try:
    from config import HYBRID_CONFIG, SAVED_MODELS_DIR, CHECKPOINTS_DIR, LOGS_DIR
except Exception:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    LOGS_DIR = os.path.join(BASE_DIR, 'logs')
    SAVED_MODELS_DIR = os.path.join(BASE_DIR, 'models', 'saved')
    CHECKPOINTS_DIR = os.path.join(BASE_DIR, 'models', 'checkpoints')
    HYBRID_CONFIG = {
        'lookback_hours': 72,
        'lstm_units': 64,
        'gru_units': 64,
        'dense_units': 32,
        'dropout_rate': 0.2,
        'learning_rate': 0.0005,
        'batch_size': 32,
        'epochs': 150,
        'patience': 20
    }

for _d in [LOGS_DIR, SAVED_MODELS_DIR, CHECKPOINTS_DIR]:
    os.makedirs(_d, exist_ok=True)

TB_LOGS_DIR = os.path.join(LOGS_DIR, 'tensorboard')
os.makedirs(TB_LOGS_DIR, exist_ok=True)

# ============================================================
# LOGIRANJE
# ============================================================
for _h in logging.root.handlers[:]:
    logging.root.removeHandler(_h)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(
            os.path.join(LOGS_DIR, 'hybrid_model.log'),
            encoding='utf-8'
        ),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Uvoz AttentionLayer
try:
    from lstm_model import AttentionLayer
except Exception as e:
    logger.warning(f"AttentionLayer nije uvezen: {e}")
    from tensorflow.keras.layers import Layer

    class AttentionLayer(Layer):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)

        def build(self, input_shape):
            dim = input_shape[-1]
            self.W = self.add_weight('W', (dim, dim), initializer='glorot_uniform')
            self.b = self.add_weight('b', (dim,), initializer='zeros')
            self.u = self.add_weight('u', (dim,), initializer='glorot_uniform')
            super().build(input_shape)

        def call(self, inputs):
            score = tf.nn.tanh(tf.tensordot(inputs, self.W, axes=1) + self.b)
            alpha = tf.nn.softmax(tf.tensordot(score, self.u, axes=1), axis=1)
            return tf.reduce_sum(inputs * tf.expand_dims(alpha, -1), axis=1)

        def get_config(self):
            return super().get_config()


# ============================================================
# HIBRIDNI MODEL
# ============================================================

class HybridModel:
    """
    Hibridni LSTM-GRU model s dvije paralelne grane.

    Arhitektura:
    Input -> [LSTM grana s Attention]
          -> [GRU grana s Attention]
          -> Concatenate -> Dense -> Output

    Treniranje je sporo ali daje najbolje rezultate
    jer kombinira jake strane oba pristupa.
    """

    def __init__(self, config=None):
        self.config = config or HYBRID_CONFIG
        self.model = None
        self.history = None
        logger.info("HybridModel inicijaliziran.")

    def build_model(self, input_shape, output_steps):
        """
        Gradi hibridnu LSTM-GRU arhitekturu s Attention slojevima.

        Args:
            input_shape  : (lookback, n_features)
            output_steps : broj koraka predikcije
        """
        logger.info(
            f"Gradim hibridni model: input={input_shape}, output={output_steps}"
        )

        lstm_u = self.config['lstm_units']
        gru_u = self.config['gru_units']
        drop = self.config['dropout_rate']

        inputs = Input(shape=input_shape, name='input_sequence')

        # ===================================================
        # LSTM GRANA (s Attention)
        # ===================================================
        lstm_x = Bidirectional(
            LSTM(
                units=lstm_u,
                return_sequences=True,
                dropout=drop,
                kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4),
                name='lstm_branch_1'
            ),
            name='bilstm_branch'
        )(inputs)

        lstm_x = BatchNormalization(name='bn_lstm_1')(lstm_x)
        lstm_x = Dropout(drop, name='dropout_lstm_1')(lstm_x)

        # NOVO: return_sequences=True za Attention
        lstm_x = LSTM(
            units=lstm_u // 2,
            return_sequences=True,
            dropout=drop,
            name='lstm_branch_2'
        )(lstm_x)

        lstm_x = BatchNormalization(name='bn_lstm_2')(lstm_x)

        # NOVO: Attention sloj na LSTM grani
        lstm_x = AttentionLayer(name='attention_lstm')(lstm_x)
        lstm_x = Dropout(drop, name='dropout_lstm_2')(lstm_x)

        # ===================================================
        # GRU GRANA (s Attention)
        # ===================================================
        gru_x = Bidirectional(
            GRU(
                units=gru_u,
                return_sequences=True,
                dropout=drop,
                kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4),
                reset_after=True,
                name='gru_branch_1'
            ),
            name='bigru_branch'
        )(inputs)

        gru_x = BatchNormalization(name='bn_gru_1')(gru_x)
        gru_x = Dropout(drop, name='dropout_gru_1')(gru_x)

        # NOVO: return_sequences=True za Attention
        gru_x = GRU(
            units=gru_u // 2,
            return_sequences=True,
            dropout=drop,
            reset_after=True,
            name='gru_branch_2'
        )(gru_x)

        gru_x = BatchNormalization(name='bn_gru_2')(gru_x)

        # NOVO: Attention sloj na GRU grani
        gru_x = AttentionLayer(name='attention_gru')(gru_x)
        gru_x = Dropout(drop, name='dropout_gru_2')(gru_x)

        # ===================================================
        # SPAJANJE GRANA
        # ===================================================
        merged = Concatenate(name='merge')([lstm_x, gru_x])
        merged = BatchNormalization(name='bn_merged')(merged)

        # Dense slojevi
        x = Dense(
            self.config['dense_units'] * 2,
            activation='relu',
            name='dense_1'
        )(merged)
        x = Dropout(drop, name='dropout_merged')(x)
        x = Dense(
            self.config['dense_units'],
            activation='relu',
            name='dense_2'
        )(x)

        # Izlaz
        outputs = Dense(output_steps, activation='linear', name='output')(x)

        self.model = Model(
            inputs=inputs, outputs=outputs, name='Hybrid_LSTM_GRU'
        )

        self.model.compile(
            optimizer=Adam(learning_rate=self.config['learning_rate']),
            loss='mse',
            metrics=['mae', 'mse']
        )

        summary_lines = []
        self.model.summary(print_fn=lambda line: summary_lines.append(line))
        for line in summary_lines:
            logger.info(line)

        n_params = self.model.count_params()
        logger.info(f"Hibridni model izgraden: {n_params:,} parametara")

        return self.model

    def get_callbacks(self, model_name='hybrid_model'):
        """Kreira callback funkcije za treniranje."""
        tb_dir = os.path.join(TB_LOGS_DIR, model_name)
        os.makedirs(tb_dir, exist_ok=True)

        ckpt_path = os.path.join(
            CHECKPOINTS_DIR, f'{model_name}_best.keras'
        )

        return [
            EarlyStopping(
                monitor='val_loss',
                patience=self.config['patience'],
                restore_best_weights=True,
                verbose=1, mode='min'
            ),
            ModelCheckpoint(
                filepath=ckpt_path,
                monitor='val_loss',
                save_best_only=True, verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss', factor=0.5,
                patience=7, min_lr=1e-7, verbose=1
            ),
            TensorBoard(
                log_dir=tb_dir, histogram_freq=0,
                write_graph=False, update_freq='epoch'
            )
        ]

    def train(self, X_train, y_train, X_val, y_val):
        """Trenira hibridni model."""
        if self.model is None:
            raise ValueError("Model nije izgraden! Pozovi build_model() prvo.")

        logger.info("Zapocinje treniranje hibridnog modela...")
        logger.info(f"  Trening uzorci    : {X_train.shape[0]:,}")
        logger.info(f"  Validacijski uzorci: {X_val.shape[0]:,}")
        logger.info(f"  Batch size        : {self.config['batch_size']}")
        logger.info(f"  Max epoha         : {self.config['epochs']}")

        callbacks = self.get_callbacks('hybrid_model')

        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=self.config['epochs'],
            batch_size=self.config['batch_size'],
            callbacks=callbacks,
            verbose=1,
            shuffle=False
        )

        best_val = min(self.history.history['val_loss'])
        n_epochs = len(self.history.history['loss'])
        logger.info(f"Treniranje zavrseno: {n_epochs} epoha, "
                    f"best val_loss={best_val:.6f}")

        return self.history

    def predict(self, X):
        """Generira predikcije."""
        if self.model is None:
            raise ValueError("Model nije izgraden ili ucitan!")
        return self.model.predict(X, verbose=0)

    def save_model(self, filepath=None):
        """Sprema model na disk."""
        if filepath is None:
            filepath = os.path.join(SAVED_MODELS_DIR, 'hybrid_model.keras')
        self.model.save(filepath)
        logger.info(f"Hibridni model spremljen: {filepath}")

    def load_model(self, filepath=None):
        """Ucitava model s diska."""
        if filepath is None:
            filepath = os.path.join(SAVED_MODELS_DIR, 'hybrid_model.keras')
        self.model = tf.keras.models.load_model(
            filepath,
            custom_objects={'AttentionLayer': AttentionLayer}
        )
        logger.info(f"Hibridni model ucitan: {filepath}")
        return self.model


# ============================================================
# TESTIRANJE
# ============================================================

if __name__ == '__main__':
    print("Test hibridnog modela...")

    X_dummy = np.random.randn(100, 72, 45).astype(np.float32)
    y_dummy = np.random.randn(100, 24).astype(np.float32)

    X_tr, X_vl = X_dummy[:70], X_dummy[70:]
    y_tr, y_vl = y_dummy[:70], y_dummy[70:]

    hybrid = HybridModel()
    hybrid.build_model(input_shape=(72, 45), output_steps=24)

    test_cfg = hybrid.config.copy()
    test_cfg['epochs'] = 2
    test_cfg['patience'] = 2
    hybrid.config = test_cfg

    hybrid.train(X_tr, y_tr, X_vl, y_vl)

    preds = hybrid.predict(X_vl)
    print(f"Predikcije shape: {preds.shape}")
    print("Hibridni model test prosao uspjesno!")