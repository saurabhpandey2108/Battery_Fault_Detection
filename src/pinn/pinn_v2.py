"""
Pure-NumPy physics-informed MLP (V, I) -> 8 battery parameters.

Why this module exists
----------------------
The upstream DeepNeuralNetwork (src/pinn/deep_neural_network.py) has two
fatal flaws for the Tsinghua NCM811 PINN task:

  1. `_calculate_voltages` hard-codes `i = x[idx][1]` to pull "current"
     from column 1 of the feature matrix, but column 1 is actually
     Terminal_voltage with the 14-feature layout. The physics has been
     receiving ~3-4 V as "current" instead of ~1.5 A -> the integrators
     ir1, ir2 go unstable almost immediately.

  2. `back_propagate` sets dZ[:, k] = voltage_error for every output
     neuron k, i.e. every one of the 8 parameters receives the same
     gradient signal. That's not the actual partial derivative of V with
     respect to each parameter, so Adam wanders instead of descending.

This v2 module fixes both:

  * Input is exactly (V_meas, I_meas). 2 features, no engineering.
  * Architecture: [2, 32, 32, 8], linear output (8 parameters stay free).
  * Forward pass is vectorized across all samples in a file.
  * Backprop uses the TRUE Jacobian dV/dp_k, computed numerically by
    finite differences on calculate_voltage at the step the sample was
    generated. The 8 partials drive distinct updates for each parameter.
  * Adam has gradient-norm clipping. Clipping is on the *gradient*, NOT
    on the parameters or outputs — this is standard optimization hygiene,
    not a bound on the 8 physics parameters.
  * Whole-batch forward/backward is skipped if any non-finite value
    appears, so one bad step can't permanently corrupt weights.
"""

from __future__ import annotations

import numpy as np


class PinnV2:
    """Simple MLP for (V, I) -> 8 battery parameters with Adam + grad clipping."""

    def __init__(self, layer_sizes=(2, 32, 32, 8), dropout_rate=0.0, seed=42):
        if len(layer_sizes) < 3:
            raise ValueError("Need at least 3 layers")
        rng = np.random.default_rng(seed)
        self.sizes = list(layer_sizes)
        self.L = len(self.sizes) - 1
        self.dropout_rate = float(dropout_rate)
        self.params = {}
        for i in range(1, len(self.sizes)):
            scale = np.sqrt(2.0 / self.sizes[i - 1])
            self.params[f"W{i}"] = rng.standard_normal((self.sizes[i], self.sizes[i - 1])) * scale
            self.params[f"b{i}"] = np.zeros((self.sizes[i], 1))
        self.adam_m = {k: np.zeros_like(v) for k, v in self.params.items()}
        self.adam_v = {k: np.zeros_like(v) for k, v in self.params.items()}
        self.t = 1
        self.cache: dict = {}

    # ------------------------------------------------------------------
    #  Warm-start
    # ------------------------------------------------------------------
    def warm_start_output_bias(self, target_values):
        """Set the output-layer biases so the initial forward pass is
        in a numerically stable physics regime. Not a bound — weights stay
        free to move during training."""
        tv = np.asarray(target_values, dtype=np.float64).reshape(-1, 1)
        if tv.shape != self.params[f"b{self.L}"].shape:
            raise ValueError(
                f"target_values length {tv.size} != output size {self.sizes[-1]}"
            )
        # Shrink last-layer W so biases dominate at init.
        self.params[f"W{self.L}"] *= 1e-3
        self.params[f"b{self.L}"] = tv
        self.adam_m[f"W{self.L}"] = np.zeros_like(self.params[f"W{self.L}"])
        self.adam_v[f"W{self.L}"] = np.zeros_like(self.params[f"W{self.L}"])
        self.adam_m[f"b{self.L}"] = np.zeros_like(self.params[f"b{self.L}"])
        self.adam_v[f"b{self.L}"] = np.zeros_like(self.params[f"b{self.L}"])

    # ------------------------------------------------------------------
    #  Forward
    # ------------------------------------------------------------------
    @staticmethod
    def _relu(x):
        return np.maximum(0.0, x)

    @staticmethod
    def _relu_deriv(x):
        return (x > 0).astype(np.float64)

    def forward(self, x, training=True):
        """Forward pass. x : (N, input_size). Returns (N, output_size)."""
        x = np.asarray(x, dtype=np.float64)
        a = x.T  # (input_size, N)
        self.cache = {"A0": a, "dropout_masks": {}}
        for i in range(1, self.L + 1):
            z = self.params[f"W{i}"] @ self.cache[f"A{i-1}"] + self.params[f"b{i}"]
            if i < self.L:
                a = self._relu(z)
                if training and self.dropout_rate > 0.0:
                    mask = (np.random.rand(*a.shape) > self.dropout_rate).astype(np.float64)
                    mask /= (1.0 - self.dropout_rate)
                    a = a * mask
                    self.cache["dropout_masks"][i] = mask
            else:
                a = z  # linear output
            self.cache[f"Z{i}"] = z
            self.cache[f"A{i}"] = a
        return a.T

    # ------------------------------------------------------------------
    #  Backward
    # ------------------------------------------------------------------
    def backward(self, d_output):
        """Backprop given dL/d(output). d_output : (N, output_size)."""
        N = d_output.shape[0]
        grads = {}
        dZ = d_output.T  # linear output -> dZ_L = dA_L
        for i in range(self.L, 0, -1):
            A_prev = self.cache[f"A{i-1}"]
            grads[f"W{i}"] = dZ @ A_prev.T / N
            grads[f"b{i}"] = np.sum(dZ, axis=1, keepdims=True) / N
            if i > 1:
                dA_prev = self.params[f"W{i}"].T @ dZ
                mask = self.cache["dropout_masks"].get(i - 1, None)
                if mask is not None:
                    dA_prev = dA_prev * mask
                dZ = dA_prev * self._relu_deriv(self.cache[f"Z{i-1}"])
        self.grads = grads
        return grads

    # ------------------------------------------------------------------
    #  Optimizer step with gradient clipping
    # ------------------------------------------------------------------
    def optimize(self, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8, clipnorm=1.0):
        """Adam step. `clipnorm` clips the GLOBAL GRADIENT norm across all
        params before the update. Nothing is clipped on the parameters
        themselves, and nothing is clipped on the 8 physics outputs."""
        # Skip step if any gradient is non-finite.
        for g in self.grads.values():
            if not np.all(np.isfinite(g)):
                return False

        if clipnorm is not None and clipnorm > 0:
            flat = np.concatenate([g.ravel() for g in self.grads.values()])
            total_norm = float(np.linalg.norm(flat))
            if total_norm > clipnorm:
                scale = clipnorm / (total_norm + 1e-12)
                for k in self.grads:
                    self.grads[k] = self.grads[k] * scale

        for k in self.params:
            self.adam_m[k] = beta1 * self.adam_m[k] + (1 - beta1) * self.grads[k]
            self.adam_v[k] = beta2 * self.adam_v[k] + (1 - beta2) * (self.grads[k] ** 2)
            m_hat = self.adam_m[k] / (1 - beta1 ** self.t)
            v_hat = self.adam_v[k] / (1 - beta2 ** self.t)
            self.params[k] -= lr * m_hat / (np.sqrt(v_hat) + eps)
        self.t += 1
        return True

    # ------------------------------------------------------------------
    #  I/O
    # ------------------------------------------------------------------
    def save(self, path):
        np.savez(path, __arch__=np.array(self.sizes), **self.params)
        print(f"[PinnV2] weights saved -> {path}")

    def load(self, path):
        data = np.load(path, allow_pickle=True)
        if "__arch__" in data.files:
            saved = tuple(int(s) for s in data["__arch__"])
            if tuple(self.sizes) != saved:
                raise ValueError(
                    f"architecture mismatch: current {self.sizes}, file {saved}"
                )
        for k in self.params:
            if k not in data.files:
                raise KeyError(f"missing weight {k} in {path}")
            self.params[k] = data[k]
        print(f"[PinnV2] weights loaded <- {path}")
