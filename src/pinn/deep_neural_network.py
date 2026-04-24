"""
Deep Neural Network implementation for Battery Parameter Estimation
(ported verbatim from Battery_Passport; only the BatteryModel import path
has been updated to match the new package layout).
"""

import numpy as np
import time
from scipy.interpolate import interp1d


class DeepNeuralNetwork:
    """Deep Neural Network for battery parameter estimation.

    Predicts 8 electrochemical parameters (C1, C2, R0, R1, R2, gamma1, M0, M)
    from 14 engineered features. Output layer is linear — no bounding of the
    8 parameters is applied here or anywhere downstream.
    """

    def __init__(self, sizes, activation='relu', dropout_rate=0.2):
        assert len(sizes) >= 3, "Network must have at least 3 layers (input, hidden, output)."
        self.sizes = sizes
        self.dropout_rate = dropout_rate

        if activation == 'relu':
            self.activation = self.relu
        elif activation == 'sigmoid':
            self.activation = self.sigmoid
        else:
            raise ValueError("Activation function is currently not supported, please use 'relu' or 'sigmoid' instead.")

        self.params = self.initialize()
        self.cache = {}
        self.adam_opt = self.initialize_adam_optimizer()
        self.t = 1

    def relu(self, x, derivative=False):
        if derivative:
            return np.where(x > 0, 1, 0)
        return np.maximum(0, x)

    def sigmoid(self, x, derivative=False):
        if derivative:
            sig = self.sigmoid(x)
            return sig * (1 - sig)
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

    def initialize(self):
        params = {}
        for i in range(1, len(self.sizes)):
            scale = np.sqrt(2.0 / self.sizes[i-1])
            params[f"W{i}"] = np.random.randn(self.sizes[i], self.sizes[i-1]) * scale
            params[f"b{i}"] = np.zeros((self.sizes[i], 1))
        return params

    def initialize_adam_optimizer(self):
        return {
            "m": {key: np.zeros_like(value) for key, value in self.params.items()},
            "v": {key: np.zeros_like(value) for key, value in self.params.items()}
        }

    def feed_forward(self, x, training=True):
        self.cache["A0"] = x.T

        for i in range(1, len(self.sizes) - 1):
            self.cache[f"Z{i}"] = np.matmul(self.params[f"W{i}"], self.cache[f"A{i-1}"]) + self.params[f"b{i}"]
            self.cache[f"A{i}"] = self.activation(self.cache[f"Z{i}"])

            if training and i < len(self.sizes) - 2:
                mask = np.random.binomial(1, 1-self.dropout_rate, size=self.cache[f"A{i}"].shape) / (1-self.dropout_rate)
                self.cache[f"A{i}"] *= mask

        final_layer = len(self.sizes) - 2
        self.cache[f"Z{final_layer + 1}"] = np.matmul(self.params[f"W{final_layer + 1}"],
                                                     self.cache[f"A{final_layer}"]) + self.params[f"b{final_layer + 1}"]
        self.cache[f"A{final_layer + 1}"] = self.cache[f"Z{final_layer + 1}"].T
        return self.cache[f"A{final_layer + 1}"]

    def back_propagate(self, voltage_error):
        current_batch_size = voltage_error.shape[0]
        grads = {}

        dZ = np.zeros((voltage_error.shape[0], self.sizes[-1]))
        for i in range(self.sizes[-1]):
            dZ[:, i] = voltage_error.flatten()

        dZ = dZ.T
        last_layer = len(self.sizes) - 2

        grads[f"W{last_layer + 1}"] = (1./current_batch_size) * np.matmul(dZ, self.cache[f"A{last_layer}"].T)
        grads[f"b{last_layer + 1}"] = (1./current_batch_size) * np.sum(dZ, axis=1, keepdims=True)

        for i in range(last_layer, 0, -1):
            dA = np.matmul(self.params[f"W{i + 1}"].T, dZ)
            dZ = dA * self.activation(self.cache[f"Z{i}"], derivative=True)
            grads[f"W{i}"] = (1./current_batch_size) * np.matmul(dZ, self.cache[f"A{i - 1}"].T)
            grads[f"b{i}"] = (1./current_batch_size) * np.sum(dZ, axis=1, keepdims=True)

        self.grads = grads
        return self.grads

    def optimize(self, l_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        for key in self.params:
            self.adam_opt['m'][key] = beta1 * self.adam_opt['m'][key] + (1 - beta1) * self.grads[key]
            self.adam_opt['v'][key] = beta2 * self.adam_opt['v'][key] + (1 - beta2) * (self.grads[key] ** 2)

            m_hat = self.adam_opt['m'][key] / (1 - beta1 ** self.t)
            v_hat = self.adam_opt['v'][key] / (1 - beta2 ** self.t)

            self.params[key] -= l_rate * m_hat / (np.sqrt(v_hat) + epsilon)
        self.t += 1

    def rmse_loss(self, y, output):
        return np.sqrt(np.mean((y - output) ** 2))

    def train(self, dataset, epochs=50, batch_size=32, l_rate=0.0001,
              validation_split=0.2, early_stopping_patience=10):
        val_size = int(len(dataset) * validation_split)
        np.random.shuffle(dataset)
        val_data = dataset[:val_size]
        train_data = dataset[val_size:]

        num_batches = len(train_data) // batch_size
        if num_batches == 0:
            raise ValueError("Batch size is too large for dataset.")

        train_losses = []
        val_losses = []
        best_val_loss = float('inf')
        patience_counter = 0

        print(f"Training on {len(train_data)} samples, validating on {len(val_data)} samples")
        print(f"Batch size: {batch_size}, Batches per epoch: {num_batches}")

        for epoch in range(epochs):
            start_time = time.time()
            epoch_train_loss = 0

            np.random.shuffle(train_data)

            for batch_idx in range(num_batches):
                batch = train_data[batch_idx * batch_size:(batch_idx + 1) * batch_size]

                x = batch[:, 1:]
                actual_voltages = batch[:, 2].reshape(-1, 1)

                predicted_params = self.feed_forward(x, training=True)
                predicted_voltages = self._calculate_voltages(predicted_params, x)
                voltage_error = actual_voltages - predicted_voltages.reshape(-1, 1)

                self.back_propagate(voltage_error)
                self.optimize(l_rate=l_rate)

                batch_loss = self.rmse_loss(actual_voltages, predicted_voltages.reshape(-1, 1))
                epoch_train_loss += batch_loss

            avg_train_loss = epoch_train_loss / num_batches
            train_losses.append(avg_train_loss)

            val_loss = self._validate(val_data)
            val_losses.append(val_loss)

            epoch_time = time.time() - start_time
            print(f"Epoch {epoch + 1}/{epochs} - {epoch_time:.2f}s - "
                  f"Train RMSE: {avg_train_loss:.6f} - Val RMSE: {val_loss:.6f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                self.best_params = {key: value.copy() for key, value in self.params.items()}
            else:
                patience_counter += 1

            if patience_counter >= early_stopping_patience:
                print(f"Early stopping at epoch {epoch + 1}")
                self.params = self.best_params
                break

        return {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'best_val_loss': best_val_loss
        }

    def _validate(self, val_data):
        total_loss = 0
        num_samples = len(val_data)

        for i in range(0, num_samples, 32):
            batch = val_data[i:i+32]
            x = batch[:, 1:]
            actual_voltages = batch[:, 2].reshape(-1, 1)

            predicted_params = self.feed_forward(x, training=False)
            predicted_voltages = self._calculate_voltages(predicted_params, x)

            batch_loss = self.rmse_loss(actual_voltages, predicted_voltages.reshape(-1, 1))
            total_loss += batch_loss * len(batch)

        return total_loss / num_samples

    def _calculate_voltages(self, predicted_params, x):
        """Calculate voltages using the battery model (preserved from upstream).

        Each sample is treated as an isolated call to calculate_voltage with
        default initial state. State threading across timesteps happens
        explicitly in infer_pinn.predict_voltage_series for inference.
        """
        from .battery_physics import BatteryModel

        battery_model = BatteryModel()
        predicted_voltages = []

        for idx in range(len(x)):
            try:
                C1, C2, R0, R1, R2, gamma1, M0, M = predicted_params[idx]
                i = x[idx][1]
                v, _, _, _, _, _ = battery_model.calculate_voltage(C1, C2, R0, R1, R2, gamma1, M0, M, i)
                predicted_voltages.append(v)
            except Exception:
                predicted_voltages.append(3.0)

        return np.array(predicted_voltages)

    def predict(self, x):
        return self.feed_forward(x, training=False)

    def save_model(self, filepath):
        np.savez(filepath, **self.params)
        print(f"Model saved to {filepath}")

    def load_model(self, filepath):
        loaded = np.load(filepath)
        self.params = {key: loaded[key] for key in loaded.keys()}
        print(f"Model loaded from {filepath}")
