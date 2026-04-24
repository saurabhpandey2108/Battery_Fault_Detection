"""
Battery Physics Model (ported from Battery_Passport).

Edits vs. upstream:
    * Removed all np.clip() bounds on the 8 electrochemical parameters
      (C1, C2, R0, R1, R2, gamma1, M0, M). SOC clip and eps safety are kept.
    * sampling_time is now a constructor argument (default 1.0 s for Tsinghua data,
      was hardcoded 0.0001 s in upstream).
"""

import numpy as np
from scipy.interpolate import interp1d


class BatteryModel:
    """Electrochemical battery model for voltage calculation."""

    def __init__(self, sampling_time: float = 1.0, capacity_as: float = 10 * 3600):
        self.Ts = float(sampling_time)
        self.Q = float(capacity_as)
        self.setup_ocv_curve()

    def setup_ocv_curve(self):
        self.ocv_y = np.array([2.5, 2.5999, 2.757, 3.0026, 3.1401, 3.2088, 3.2383,
                               3.2726, 3.2972, 3.3119, 3.3119, 3.3365, 3.3709, 3.4887, 3.5])
        self.ocv_x = np.array([0, 0.18474, 0.71411, 3.5374, 7.243, 12.36, 20.124,
                               32.3, 44.828, 60.0004, 70.591, 84.708, 97.413, 99.707, 100])
        self.ocv_interp = interp1d(self.ocv_x, self.ocv_y, kind='linear', fill_value="extrapolate")

    def calculate_ocv(self, soc):
        soc = np.clip(soc, 0, 100)
        return float(self.ocv_interp(soc))

    def calculate_voltage(self, C1, C2, R0, R1, R2, gamma1, M0, M, i,
                          ir1=0, ir2=0, z=100, h=0, s=0):
        """
        V = OCV(z) + Vh - R1*ir1 - R2*ir2 - R0*i

        The 8 electrochemical parameters are passed through without clipping —
        the PINN learns whatever regime best fits V_meas. SOC is still clipped
        to [0, 100] so OCV interpolation stays in-domain.
        """
        Ts = self.Ts
        Q = self.Q
        n = 1
        eps = 1e-10

        try:
            # State of charge update
            z = z - Ts * n * i / (Q + eps)
            z = np.clip(z, 0, 100)

            # RC circuit currents (first-order dynamics)
            tau1 = (R1 + eps) * (C1 + eps)
            tau2 = (R2 + eps) * (C2 + eps)

            ir1 = np.exp(-Ts / tau1) * ir1 + (1 - np.exp(-Ts / tau1)) * i
            ir2 = np.exp(-Ts / tau2) * ir2 + (1 - np.exp(-Ts / tau2)) * i

            # Hysteresis
            u = np.exp(-abs(n * i * gamma1 * Ts / (Q + eps)))
            h = u * h - (1 - u) * (1.0 if i > 0 else 0.0)
            s = 1 if i > 0 else s
            vh = M0 * s + M * h

            # Terminal voltage
            v = self.calculate_ocv(z) + vh - R1 * ir1 - R2 * ir2 - R0 * i

            return round(v, 4), ir1, ir2, z, h, s

        except Exception:
            return 3.0, ir1, ir2, z, h, s

    def simulate_discharge(self, params, current_profile, initial_soc=100):
        C1, C2, R0, R1, R2, gamma1, M0, M = params
        ir1, ir2 = 0, 0
        z = initial_soc
        h, s = 0, 0

        voltages, socs = [], []
        for i in current_profile:
            v, ir1, ir2, z, h, s = self.calculate_voltage(
                C1, C2, R0, R1, R2, gamma1, M0, M, i, ir1, ir2, z, h, s
            )
            voltages.append(v)
            socs.append(z)

        return {
            'voltages': np.array(voltages),
            'socs': np.array(socs),
            'currents': current_profile
        }
