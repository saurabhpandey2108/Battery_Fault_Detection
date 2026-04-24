"""
CWT-CNN model, ported from Wavelet_Analysis/cnn_model/model.py.

Changes vs. upstream:
    * Default image_shape is (224, 224, 3) — three scalogram channels
      [V_meas, V_pred, residual] instead of the original (V, I) stack.
    * Output head is configurable (binary sigmoid, multiclass softmax,
      or the original SOC regression head) via the `head` kwarg.
"""

from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf


def build_cnn_model(image_shape=(224, 224, 3),
                    use_temperature_scalar=False,
                    noise_std=0.0,
                    head="binary",
                    n_classes=2,
                    learning_rate=1e-4):
    """CWT-CNN model with a swappable output head.

    Architecture (from the original paper):
        [GaussianNoise] → 3 × (Conv2D → BN → ReLU → MaxPool)
        → Flatten → Dropout(0.5) → [concat(temp_scalar)] → Dense(64) → Dense(out)

    Parameters
    ----------
    image_shape : tuple (H, W, C)
    use_temperature_scalar : bool
    noise_std : float
        Set 0.0 to disable the GaussianNoise augmentation layer.
    head : {"binary", "multiclass", "regression"}
        - "binary": Dense(1, sigmoid), BCE loss, metrics=[accuracy, AUC]
        - "multiclass": Dense(n_classes, softmax), sparse CCE loss, metrics=[accuracy]
        - "regression": Dense(1, linear), MSE loss (original SOC head)
    n_classes : int
        Number of classes when head == "multiclass".
    """
    img_input = keras.Input(shape=image_shape, name='image_input')

    x = img_input
    if noise_std > 0:
        x = layers.GaussianNoise(noise_std, name='input_noise')(x)

    # Block 1
    x = layers.Conv2D(32, 3, padding='same', kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D(pool_size=2)(x)

    # Block 2
    x = layers.Conv2D(64, 3, padding='same', kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D(pool_size=2)(x)

    # Block 3
    x = layers.Conv2D(128, 3, padding='same', kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D(pool_size=2)(x)

    x = layers.Flatten()(x)
    x = layers.Dropout(0.5)(x)

    if use_temperature_scalar:
        temp_input = keras.Input(shape=(1,), name='temp_input')
        x = layers.Concatenate()([x, temp_input])

    x = layers.Dense(64, activation='relu', kernel_initializer='he_normal')(x)

    if head == "binary":
        out = layers.Dense(1, activation='sigmoid', name='fault')(x)
        loss = 'binary_crossentropy'
        metrics = ['accuracy', tf.keras.metrics.AUC(name='auc')]
    elif head == "multiclass":
        out = layers.Dense(n_classes, activation='softmax', name='severity')(x)
        loss = 'sparse_categorical_crossentropy'
        metrics = ['accuracy']
    elif head == "regression":
        out = layers.Dense(1, activation='linear', name='soc')(x)
        loss = 'mse'
        metrics = ['mae', tf.keras.metrics.RootMeanSquaredError(name='rmse')]
    else:
        raise ValueError(f"Unknown head: {head!r}")

    if use_temperature_scalar:
        model = keras.Model([img_input, temp_input], out)
    else:
        model = keras.Model(img_input, out)

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=1.0)
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    return model
