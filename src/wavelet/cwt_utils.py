"""
Continuous Wavelet Transform (CWT) — Morlet, pure NumPy.
(Ported verbatim from Wavelet_Analysis/cwt_calc/cwt_utils.py.)

CWT(a, b) = (1/√a) ∫ x(t) · ψ*((t - b) / a) dt
f_pseudo  = ω₀ / (2π · a · dt)
Morlet (ω₀ = 6):   ψ(t) = π^(-1/4) · e^(jω₀t) · e^(-t²/2)

Kernel is built per-scale (length 2·ceil(4a)+1); the input is reflection-
padded so the FFT-based linear convolution does not see a cliff at the
window boundaries.
"""

import numpy as np


def coi_mask(n, scales):
    """Cone-of-Influence mask (True = inside COI = unreliable coefficient)."""
    t = np.arange(n)
    sqrt2 = np.sqrt(2.0)
    mask = np.zeros((len(scales), n), dtype=bool)
    for i, a in enumerate(scales):
        tau = a * sqrt2
        mask[i] = (t < tau) | (t > (n - 1 - tau))
    return mask


def morlet_wavelet(t, w0=6):
    return (np.pi ** -0.25) * np.exp(1j * w0 * t) * np.exp(-t ** 2 / 2)


def _wavelet_kernel(scale, dt, w0=6):
    half = int(np.ceil(4 * scale))
    kernel_len = 2 * half + 1
    tk = (np.arange(kernel_len) - half) * dt
    wavelet = morlet_wavelet(tk / scale, w0) / np.sqrt(scale)
    return wavelet, half


def compute_cwt(signal, fs, scales, w0=6):
    signal = np.asarray(signal, dtype=np.float64)
    n = len(signal)
    dt = 1.0 / fs

    cwt_matrix = np.zeros((len(scales), n), dtype=np.complex128)
    freqs = np.zeros(len(scales))

    for i, scale in enumerate(scales):
        freqs[i] = w0 / (2 * np.pi * scale * dt)
        wavelet, half = _wavelet_kernel(scale, dt, w0=w0)

        if half > 0:
            signal_padded = np.pad(signal, half, mode='reflect')
        else:
            signal_padded = signal

        kernel = np.conj(wavelet[::-1])
        N_full = len(signal_padded) + len(kernel) - 1
        conv = np.fft.ifft(np.fft.fft(signal_padded, N_full) *
                           np.fft.fft(kernel, N_full))

        start = 2 * half
        cwt_matrix[i, :] = conv[start:start + n] * dt

    return cwt_matrix, freqs
