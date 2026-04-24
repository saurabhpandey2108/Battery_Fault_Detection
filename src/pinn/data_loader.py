"""
Data Processing Module (ported verbatim from Battery_Passport).

The feature-engineering contract (14 engineered features from
Time / Current_sense / Terminal_voltage) is preserved unchanged so that
the DeepNeuralNetwork weights trained here remain compatible with the
original Battery_Passport input spec.
"""

import pandas as pd
import numpy as np
from typing import Tuple


class BatteryDataProcessor:
    """Handles loading, cleaning, and feature engineering of battery data."""

    def __init__(self, window_size: int = 5):
        self.window_size = window_size
        self.feature_columns = None
        self.target_column = 'Terminal_voltage'

    def load_raw_data(self, filepaths: list) -> pd.DataFrame:
        dataframes = []
        for filepath in filepaths:
            try:
                df = pd.read_excel(filepath)
                print(f"Loaded {len(df)} samples from {filepath}")
                dataframes.append(df)
            except Exception as e:
                print(f"Error loading {filepath}: {e}")

        if dataframes:
            combined_df = pd.concat(dataframes, ignore_index=True)
            print(f"Combined dataset shape: {combined_df.shape}")
            return combined_df
        else:
            raise ValueError("No data files could be loaded")

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Produce the 14 engineered features from Time, Current_sense, Terminal_voltage."""
        processed_df = df.copy()

        processed_df['Avg_Current'] = df['Current_sense'].rolling(
            window=self.window_size, min_periods=1).mean()
        processed_df['Avg_Voltage'] = df['Terminal_voltage'].rolling(
            window=self.window_size, min_periods=1).mean()

        processed_df['Max_Current'] = df['Current_sense'].rolling(
            window=self.window_size, min_periods=1).max()
        processed_df['Min_Current'] = df['Current_sense'].rolling(
            window=self.window_size, min_periods=1).min()
        processed_df['Max_Voltage'] = df['Terminal_voltage'].rolling(
            window=self.window_size, min_periods=1).max()
        processed_df['Min_Voltage'] = df['Terminal_voltage'].rolling(
            window=self.window_size, min_periods=1).min()

        processed_df['Peak_Current'] = processed_df['Max_Current'] - processed_df['Min_Current']
        processed_df['Peak_Voltage'] = processed_df['Max_Voltage'] - processed_df['Min_Voltage']

        processed_df['RMS_Current'] = np.sqrt(
            (df['Current_sense']**2).rolling(window=self.window_size, min_periods=1).mean())
        processed_df['RMS_Voltage'] = np.sqrt(
            (df['Terminal_voltage']**2).rolling(window=self.window_size, min_periods=1).mean())

        processed_df['dV/dt'] = df['Terminal_voltage'].diff()
        processed_df['dI/dt'] = df['Current_sense'].diff()

        processed_df['dV/dt'] = processed_df['dV/dt'].fillna(0)
        processed_df['dI/dt'] = processed_df['dI/dt'].fillna(0)

        processed_df = processed_df.dropna()

        return processed_df

    def normalize_features(self, df: pd.DataFrame, fit_transform: bool = True) -> pd.DataFrame:
        feature_cols = [col for col in df.columns if col not in ['Time', 'Terminal_voltage']]

        if fit_transform:
            self.feature_means = df[feature_cols].mean()
            self.feature_stds = df[feature_cols].std()
            self.voltage_mean = df['Terminal_voltage'].mean()
            self.voltage_std = df['Terminal_voltage'].std()

        normalized_df = df.copy()
        normalized_df[feature_cols] = (df[feature_cols] - self.feature_means) / self.feature_stds
        # Voltage kept unnormalized for physics-informed loss.
        return normalized_df

    def prepare_dataset(self, df: pd.DataFrame, test_size: float = 0.2) -> Tuple[np.ndarray, np.ndarray]:
        data_array = df.to_numpy()
        np.random.shuffle(data_array)

        split_idx = int(len(data_array) * (1 - test_size))
        train_data = data_array[:split_idx]
        test_data = data_array[split_idx:]

        print(f"Training samples: {len(train_data)}")
        print(f"Testing samples: {len(test_data)}")
        return train_data, test_data
