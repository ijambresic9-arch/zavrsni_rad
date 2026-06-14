"""
GRU (Gated Recurrent Unit) model za predvidjanje kvalitete zraka.

GRU je pojednostavljena varijanta LSTM-a:
- Manje parametara (brze treniranje)
- Samo 2 gate-a: update i reset (umjesto 3 u LSTM-u)
- Slicna performansa za vecinu problema

Reference:
- Cho et al. (2014) "Learning Phrase Representations using
  RNN Encoder-Decoder for Statistical Machine Translation"
"""

import os
import sys
import logging
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, GRU, Dense, Dropout, Bidirectional,
    BatchNormalization
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
    from config import GRU_CONFIG, SAVED_MODELS_DIR, CHECKPOINTS_DIR, LOGS_DIR
except Exception:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    LOGS_DIR = os.path.join(BASE_DIR, 'logs')
    SAVED_MODELS_DIR = os.path.join(BASE_DIR, 'models', 'saved')
    CHECKPOINTS_DIR = os.path.join(BASE_DIR, 'models', 'checkpoints')
    GRU_CONFIG = {
        'lookback_hours': 72,
        'units_layer1': 128,
        'units_layer2': 64,
        'units_layer3': 32,
        'dropout_rate': 0.2,
        'recurrent_dropout': 0.0,
        'learning_rate': 0.001,
        'batch_size': 32,
        'epochs': 100,
        'patience': 15,
        'bidirectional': True
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
            os.path.join(LOGS_DIR, 'gru_model.log'),
            encoding='utf-8'
        ),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Uvoz AttentionLayer iz lstm_model
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
# GRU MODEL
# ============================================================

class GRUModel:
    """
    GRU model za predvidjanje vremenskih serija kvalitete zraka.

    Prednosti GRU-a u odnosu na LSTM:
    - Manje parametara (~25% manje)
    - Brze treniranje
    - Slicna performansa za vecinu zadataka
    """

    def __init__(self, config=None):
        self.config = config or GRU_CONFIG
        self.model = None
        self.history = None
        logger.info("GRUModel inicijaliziran.")

    def build_model(self, input_shape, output_steps):
        """Gradi GRU model."""
        logger.info(
            f"Gradim GRU model: input={input_shape}, output={output_steps}"
        )

        # GPU provjera
        gpus = tf.config.list_physical_devices('GPU')
        rec_dropout = 0.0 if gpus else self.config.get('recurrent_dropout', 0.0)

        inputs = Input(shape=input_shape, name='input_sequence')

        # --- Sloj 1: Bidirekcijski GRU ---
        if self.config.get('bidirectional', True):
            x = Bidirectional(
                GRU(
                    units=self.config['units_layer1'],
                    return_sequences=True,
                    dropout=self.config['dropout_rate'],
                    recurrent_dropout=rec_dropout,
                    kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4),
                    reset_after=True,
                    name='gru_1'
                ),
                name='bigru_1'
            )(inputs)
        else:
            x = GRU(
                units=self.config['units_layer1'],
                return_sequences=True,
                dropout=self.config['dropout_rate'],
                recurrent_dropout=rec_dropout,
                reset_after=True,
                name='gru_1'
            )(inputs)

        x = BatchNormalization(name='bn_1')(x)
        x = Dropout(self.config['dropout_rate'], name='dropout_1')(x)

        # --- Sloj 2: Bidirekcijski GRU ---
        if self.config.get('bidirectional', True):
            x = Bidirectional(
                GRU(
                    units=self.config['units_layer2'],
                    return_sequences=True,
                    dropout=self.config['dropout_rate'],
                    recurrent_dropout=rec_dropout,
                    reset_after=True,
                    name='gru_2'
                ),
                name='bigru_2'
            )(x)
        else:
            x = GRU(
                units=self.config['units_layer2'],
                return_sequences=True,
                dropout=self.config['dropout_rate'],
                reset_after=True,
                name='gru_2'
            )(x)

        x = BatchNormalization(name='bn_2')(x)
        x = Dropout(self.config['dropout_rate'], name='dropout_2')(x)

        # --- Sloj 3: GRU ---
        x = GRU(
            units=self.config['units_layer3'],
            return_sequences=True,
            dropout=self.config['dropout_rate'],
            reset_after=True,
            name='gru_3'
        )(x)
        x = BatchNormalization(name='bn_3')(x)

        # --- Attention ---
        x = AttentionLayer(name='attention')(x)

        # --- Dense slojevi ---
        x = Dense(64, activation='relu', name='dense_1')(x)
        x = Dropout(self.config['dropout_rate'], name='dropout_3')(x)
        x = Dense(32, activation='relu', name='dense_2')(x)

        # --- Izlaz ---
        outputs = Dense(output_steps, activation='linear', name='output')(x)

        self.model = Model(
            inputs=inputs, outputs=outputs, name='GRU_AirQuality'
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
        logger.info(f"GRU model izgraden: {n_params:,} parametara")

        return self.model

    def get_callbacks(self, model_name='gru_model'):
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
        """Trenira GRU model."""
        if self.model is None:
            raise ValueError("Model nije izgraden! Pozovi build_model() prvo.")

        logger.info("Zapocinje treniranje GRU modela...")
        logger.info(f"  Trening uzorci    : {X_train.shape[0]:,}")
        logger.info(f"  Validacijski uzorci: {X_val.shape[0]:,}")
        logger.info(f"  Batch size        : {self.config['batch_size']}")
        logger.info(f"  Max epoha         : {self.config['epochs']}")

        callbacks = self.get_callbacks('gru_model')

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
            filepath = os.path.join(SAVED_MODELS_DIR, 'gru_model.keras')
        self.model.save(filepath)
        logger.info(f"GRU model spremljen: {filepath}")

    def load_model(self, filepath=None):
        """Ucitava model s diska."""
        if filepath is None:
            filepath = os.path.join(SAVED_MODELS_DIR, 'gru_model.keras')
        self.model = tf.keras.models.load_model(
            filepath,
            custom_objects={'AttentionLayer': AttentionLayer}
        )
        logger.info(f"GRU model ucitan: {filepath}")
        return self.model


# ============================================================
# TESTIRANJE
# ============================================================

if __name__ == '__main__':
    print("Test GRU modela...")

    X_dummy = np.random.randn(100, 72, 45).astype(np.float32)
    y_dummy = np.random.randn(100, 24).astype(np.float32)

    X_tr, X_vl = X_dummy[:70], X_dummy[70:]
    y_tr, y_vl = y_dummy[:70], y_dummy[70:]

    gru = GRUModel()
    gru.build_model(input_shape=(72, 45), output_steps=24)

    test_cfg = gru.config.copy()
    test_cfg['epochs'] = 2
    test_cfg['patience'] = 2
    gru.config = test_cfg

    gru.train(X_tr, y_tr, X_vl, y_vl)

    preds = gru.predict(X_vl)
    print(f"Predikcije shape: {preds.shape}")
    print("GRU test prosao uspjesno!")