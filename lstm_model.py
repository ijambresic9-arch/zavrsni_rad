"""
LSTM (Long Short-Term Memory) model za predvidjanje kvalitete zraka.

Arhitektura:
- 3 sloja Bidirectional LSTM s opadajucim brojem jedinica
- BatchNormalization i Dropout za regularizaciju
- Attention mehanizam za fokusiranje na vazne vremenske korake
- Dense slojevi za finalnu predikciju
- L1/L2 regularizacija

Reference:
- Hochreiter & Schmidhuber (1997) "Long Short-Term Memory"
- Bahdanau et al. (2015) "Neural Machine Translation by
  Jointly Learning to Align and Translate" (Attention)
"""

import os
import sys
import logging
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, LSTM, Dense, Dropout, Bidirectional,
    BatchNormalization, Layer
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
    from config import LSTM_CONFIG, SAVED_MODELS_DIR, CHECKPOINTS_DIR, LOGS_DIR
except Exception:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    LOGS_DIR = os.path.join(BASE_DIR, 'logs')
    SAVED_MODELS_DIR = os.path.join(BASE_DIR, 'models', 'saved')
    CHECKPOINTS_DIR = os.path.join(BASE_DIR, 'models', 'checkpoints')
    LSTM_CONFIG = {
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
        'bidirectional': True,
        'attention': True
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
            os.path.join(LOGS_DIR, 'lstm_model.log'),
            encoding='utf-8'
        ),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


# ============================================================
# ATTENTION SLOJ
# ============================================================

class AttentionLayer(Layer):
    """
    Bahdanau Attention mehanizam.

    Omogucuje modelu da se "fokusira" na razlicite dijelove
    ulazne sekvence prilikom predvidjanja.

    Matematicki model:
        score = tanh(W * h + b)
        alpha = softmax(score * u)
        context = sum(alpha * h)
    """

    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        dim = input_shape[-1]
        self.W = self.add_weight(
            name='attn_W', shape=(dim, dim),
            initializer='glorot_uniform', trainable=True
        )
        self.b = self.add_weight(
            name='attn_b', shape=(dim,),
            initializer='zeros', trainable=True
        )
        self.u = self.add_weight(
            name='attn_u', shape=(dim,),
            initializer='glorot_uniform', trainable=True
        )
        super(AttentionLayer, self).build(input_shape)

    def call(self, inputs):
        # Score za svaki vremenski korak
        score = tf.nn.tanh(tf.tensordot(inputs, self.W, axes=1) + self.b)
        # Attention tezine (softmax)
        alpha = tf.nn.softmax(
            tf.tensordot(score, self.u, axes=1), axis=1
        )
        # Kontekst vektor (weighted sum)
        context = tf.reduce_sum(
            inputs * tf.expand_dims(alpha, -1), axis=1
        )
        return context

    def get_config(self):
        return super(AttentionLayer, self).get_config()


# ============================================================
# LSTM MODEL
# ============================================================

class LSTMModel:
    """
    LSTM model za predvidjanje vremenskih serija kvalitete zraka.

    Koristi:
    - Bidirectional LSTM (hvata ovisnosti u oba smjera)
    - Attention mehanizam (fokus na vazne korake)
    - BatchNormalization (stabilnost treniranja)
    - Dropout + L1/L2 regularizacija (sprjecavanje overfittinga)
    - Adam optimizator s ReduceLROnPlateau
    """

    def __init__(self, config=None):
        self.config = config or LSTM_CONFIG
        self.model = None
        self.history = None
        logger.info("LSTMModel inicijaliziran.")

    def build_model(self, input_shape, output_steps):
        """
        Gradi LSTM model.

        Args:
            input_shape: (lookback, n_features)
            output_steps: broj koraka predikcije
        """
        logger.info(
            f"Gradim LSTM model: input={input_shape}, output={output_steps}"
        )

        # Provjeri GPU - recurrent_dropout blokira CuDNN
        gpus = tf.config.list_physical_devices('GPU')
        rec_dropout = 0.0 if gpus else self.config.get('recurrent_dropout', 0.0)

        if gpus and self.config.get('recurrent_dropout', 0.0) > 0:
            logger.warning("recurrent_dropout > 0 blokira CuDNN GPU. "
                          "Postavljam na 0.0 za GPU mod.")

        inputs = Input(shape=input_shape, name='input_sequence')

        # --- Sloj 1: Bidirekcijski LSTM ---
        if self.config.get('bidirectional', True):
            x = Bidirectional(
                LSTM(
                    units=self.config['units_layer1'],
                    return_sequences=True,
                    dropout=self.config['dropout_rate'],
                    recurrent_dropout=rec_dropout,
                    kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4),
                    name='lstm_1'
                ),
                name='bilstm_1'
            )(inputs)
        else:
            x = LSTM(
                units=self.config['units_layer1'],
                return_sequences=True,
                dropout=self.config['dropout_rate'],
                recurrent_dropout=rec_dropout,
                kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4),
                name='lstm_1'
            )(inputs)

        x = BatchNormalization(name='bn_1')(x)
        x = Dropout(self.config['dropout_rate'], name='dropout_1')(x)

        # --- Sloj 2: Bidirekcijski LSTM ---
        if self.config.get('bidirectional', True):
            x = Bidirectional(
                LSTM(
                    units=self.config['units_layer2'],
                    return_sequences=True,
                    dropout=self.config['dropout_rate'],
                    recurrent_dropout=rec_dropout,
                    name='lstm_2'
                ),
                name='bilstm_2'
            )(x)
        else:
            x = LSTM(
                units=self.config['units_layer2'],
                return_sequences=True,
                dropout=self.config['dropout_rate'],
                name='lstm_2'
            )(x)

        x = BatchNormalization(name='bn_2')(x)
        x = Dropout(self.config['dropout_rate'], name='dropout_2')(x)

        # --- Sloj 3: LSTM ---
        x = LSTM(
            units=self.config['units_layer3'],
            return_sequences=True,
            dropout=self.config['dropout_rate'],
            name='lstm_3'
        )(x)
        x = BatchNormalization(name='bn_3')(x)

        # --- Attention ---
        if self.config.get('attention', True):
            x = AttentionLayer(name='attention')(x)
        else:
            x = tf.keras.layers.Lambda(
                lambda t: t[:, -1, :], name='last_step'
            )(x)

        # --- Dense slojevi ---
        x = Dense(64, activation='relu', name='dense_1')(x)
        x = Dropout(self.config['dropout_rate'], name='dropout_3')(x)
        x = Dense(32, activation='relu', name='dense_2')(x)

        # --- Izlaz ---
        outputs = Dense(output_steps, activation='linear', name='output')(x)

        self.model = Model(
            inputs=inputs, outputs=outputs, name='LSTM_AirQuality'
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
        logger.info(f"LSTM model izgraden: {n_params:,} parametara")

        return self.model

    def get_callbacks(self, model_name='lstm_model'):
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
        """Trenira LSTM model."""
        if self.model is None:
            raise ValueError("Model nije izgraden! Pozovi build_model() prvo.")

        logger.info("Zapocinje treniranje LSTM modela...")
        logger.info(f"  Trening uzorci    : {X_train.shape[0]:,}")
        logger.info(f"  Validacijski uzorci: {X_val.shape[0]:,}")
        logger.info(f"  Batch size        : {self.config['batch_size']}")
        logger.info(f"  Max epoha         : {self.config['epochs']}")

        callbacks = self.get_callbacks('lstm_model')

        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=self.config['epochs'],
            batch_size=self.config['batch_size'],
            callbacks=callbacks,
            verbose=1,
            shuffle=False      # Vremenske serije - ne mijesaj!
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
            filepath = os.path.join(SAVED_MODELS_DIR, 'lstm_model.keras')
        self.model.save(filepath)
        logger.info(f"LSTM model spremljen: {filepath}")

    def load_model(self, filepath=None):
        """Ucitava model s diska."""
        if filepath is None:
            filepath = os.path.join(SAVED_MODELS_DIR, 'lstm_model.keras')
        self.model = tf.keras.models.load_model(
            filepath,
            custom_objects={'AttentionLayer': AttentionLayer}
        )
        logger.info(f"LSTM model ucitan: {filepath}")
        return self.model


# ============================================================
# TESTIRANJE
# ============================================================

if __name__ == '__main__':
    print("Test LSTM modela...")

    # Dummy podaci
    X_dummy = np.random.randn(100, 72, 45).astype(np.float32)
    y_dummy = np.random.randn(100, 24).astype(np.float32)

    X_tr, X_vl = X_dummy[:70], X_dummy[70:]
    y_tr, y_vl = y_dummy[:70], y_dummy[70:]

    lstm = LSTMModel()
    lstm.build_model(input_shape=(72, 45), output_steps=24)

    # Kratko testno treniranje (2 epohe)
    test_cfg = lstm.config.copy()
    test_cfg['epochs'] = 2
    test_cfg['patience'] = 2
    lstm.config = test_cfg

    lstm.train(X_tr, y_tr, X_vl, y_vl)

    preds = lstm.predict(X_vl)
    print(f"Predikcije shape: {preds.shape}")
    print("LSTM test prosao uspjesno!")