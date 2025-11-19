"""
Long Wave Analysis Module
==========================

Analyzes Kondratiev long waves and other long-run cyclical dynamics in capitalism.

Theoretical frameworks:
- Kondratiev waves (~50-60 year cycles)
- Schumpeter's innovation cycles
- Mandel's late capitalist long waves
- Perez's techno-economic paradigms

Methods:
- Spectral analysis (identify dominant frequencies)
- Band-pass filtering (isolate long wave component)
- Peak/trough dating
- Phase analysis (expansion vs contraction)
- Turning point detection
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from scipy import signal, fft
from scipy.signal import butter, filtfilt, find_peaks
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')


class LongWaveAnalyzer:
    """
    Analyze long waves in economic time series.

    Implements multiple methods for detecting and characterizing
    long-run cyclical dynamics in capitalist development.
    """

    def __init__(self, data: pd.DataFrame):
        """
        Initialize long wave analyzer.

        Parameters
        ----------
        data : pd.DataFrame
            Time series data with year column
        """
        self.data = data
        self.waves = {}

    def spectral_analysis(self, variable: str) -> Dict:
        """
        Perform spectral analysis to identify dominant frequencies.

        Uses Fourier transform to decompose series into frequency components.

        Parameters
        ----------
        variable : str
            Variable to analyze

        Returns
        -------
        Dict
            Frequency spectrum and dominant periods
        """
        df = self.data[self.data[variable].notna()].copy()
        df = df.sort_values('year')

        y = df[variable].values

        # Detrend (remove linear trend)
        t = np.arange(len(y))
        coeffs = np.polyfit(t, y, 1)
        trend = coeffs[0] * t + coeffs[1]
        y_detrended = y - trend

        # Fourier transform
        n = len(y_detrended)
        freq = fft.fftfreq(n, d=1.0)  # Annual data
        spectrum = fft.fft(y_detrended)
        power = np.abs(spectrum) ** 2

        # Keep positive frequencies only
        pos_mask = freq > 0
        freq_pos = freq[pos_mask]
        power_pos = power[pos_mask]

        # Convert frequency to period (years)
        periods = 1 / freq_pos

        # Find dominant periods
        # Focus on long waves: 30-70 year range
        long_wave_mask = (periods >= 30) & (periods <= 70)
        long_wave_periods = periods[long_wave_mask]
        long_wave_power = power_pos[long_wave_mask]

        if len(long_wave_power) > 0:
            # Find peaks in power spectrum
            peak_indices = find_peaks(long_wave_power, height=np.mean(long_wave_power))[0]

            dominant_periods = []
            for idx in peak_indices:
                dominant_periods.append({
                    'period': long_wave_periods[idx],
                    'power': long_wave_power[idx]
                })

            # Sort by power
            dominant_periods = sorted(dominant_periods, key=lambda x: x['power'], reverse=True)
        else:
            dominant_periods = []

        return {
            'variable': variable,
            'frequencies': freq_pos,
            'periods': periods,
            'power_spectrum': power_pos,
            'dominant_periods': dominant_periods,
            'long_wave_range': [30, 70]
        }

    def bandpass_filter(self,
                       variable: str,
                       period_low: int = 30,
                       period_high: int = 70) -> pd.Series:
        """
        Extract long wave component using band-pass filter.

        Isolates cyclical component in specified period range.

        Parameters
        ----------
        variable : str
            Variable to filter
        period_low : int
            Lower period bound (years)
        period_high : int
            Upper period bound (years)

        Returns
        -------
        pd.Series
            Filtered long wave component
        """
        df = self.data[self.data[variable].notna()].copy()
        df = df.sort_values('year')

        y = df[variable].values

        # Detrend first
        t = np.arange(len(y))
        coeffs = np.polyfit(t, y, 1)
        trend = coeffs[0] * t + coeffs[1]
        y_detrended = y - trend

        # Design band-pass filter
        # Convert periods to frequencies (cycles per year)
        freq_low = 1 / period_high
        freq_high = 1 / period_low

        # Normalize by Nyquist frequency (0.5 for annual data)
        nyquist = 0.5
        low = freq_low / nyquist
        high = freq_high / nyquist

        # Butterworth band-pass filter
        b, a = butter(N=3, Wn=[low, high], btype='band')

        # Apply filter
        y_filtered = filtfilt(b, a, y_detrended)

        return pd.Series(y_filtered, index=df.index, name=f'{variable}_longwave')

    def identify_kondratiev_waves(self, variable: str) -> Dict:
        """
        Identify Kondratiev waves in the time series.

        Detects complete long waves with expansion and contraction phases.

        Parameters
        ----------
        variable : str
            Variable to analyze

        Returns
        -------
        Dict
            Identified waves with phases and turning points
        """
        # Extract long wave component
        long_wave = self.bandpass_filter(variable, period_low=40, period_high=65)

        # Standardize for peak detection
        lw_std = (long_wave - long_wave.mean()) / long_wave.std()

        # Find peaks (expansion peaks)
        peaks, _ = find_peaks(lw_std.values, distance=20, prominence=0.3)

        # Find troughs (contraction troughs)
        troughs, _ = find_peaks(-lw_std.values, distance=20, prominence=0.3)

        # Match peaks and troughs to identify complete waves
        df = self.data.iloc[long_wave.index].copy()
        df['long_wave'] = lw_std.values

        waves = []
        wave_id = 1

        # Combine and sort turning points
        turning_points = []
        for p in peaks:
            turning_points.append({'index': p, 'type': 'peak', 'value': lw_std.values[p]})
        for t in troughs:
            turning_points.append({'index': t, 'type': 'trough', 'value': lw_std.values[t]})

        turning_points = sorted(turning_points, key=lambda x: x['index'])

        # Identify waves
        i = 0
        while i < len(turning_points) - 1:
            tp1 = turning_points[i]
            tp2 = turning_points[i + 1]

            # Look for trough-peak-trough pattern
            if i < len(turning_points) - 2:
                tp3 = turning_points[i + 2]

                if tp1['type'] == 'trough' and tp2['type'] == 'peak' and tp3['type'] == 'trough':
                    wave = {
                        'wave_id': wave_id,
                        'trough1_year': int(df.iloc[tp1['index']]['year']),
                        'peak_year': int(df.iloc[tp2['index']]['year']),
                        'trough2_year': int(df.iloc[tp3['index']]['year']),
                        'expansion_years': tp2['index'] - tp1['index'],
                        'contraction_years': tp3['index'] - tp2['index'],
                        'total_years': tp3['index'] - tp1['index'],
                        'amplitude': tp2['value'] - min(tp1['value'], tp3['value'])
                    }
                    waves.append(wave)
                    wave_id += 1
                    i += 2
                    continue

            i += 1

        # Calculate wave statistics
        if waves:
            avg_period = np.mean([w['total_years'] for w in waves])
            avg_expansion = np.mean([w['expansion_years'] for w in waves])
            avg_contraction = np.mean([w['contraction_years'] for w in waves])
        else:
            avg_period = None
            avg_expansion = None
            avg_contraction = None

        return {
            'variable': variable,
            'waves': waves,
            'n_waves': len(waves),
            'average_period': avg_period,
            'average_expansion_duration': avg_expansion,
            'average_contraction_duration': avg_contraction,
            'long_wave_series': long_wave,
            'peak_indices': peaks,
            'trough_indices': troughs
        }

    def phase_classification(self, variable: str) -> pd.DataFrame:
        """
        Classify each year as expansion or contraction phase.

        Parameters
        ----------
        variable : str
            Variable to analyze

        Returns
        -------
        pd.DataFrame
            Year-by-year phase classification
        """
        # Get waves
        wave_info = self.identify_kondratiev_waves(variable)
        waves = wave_info['waves']

        df = self.data[['year']].copy()
        df['kondratiev_phase'] = 'Uncertain'

        for wave in waves:
            # Expansion phase
            expansion_mask = (df['year'] >= wave['trough1_year']) & \
                           (df['year'] <= wave['peak_year'])
            df.loc[expansion_mask, 'kondratiev_phase'] = 'Expansion'

            # Contraction phase
            contraction_mask = (df['year'] > wave['peak_year']) & \
                              (df['year'] <= wave['trough2_year'])
            df.loc[contraction_mask, 'kondratiev_phase'] = 'Contraction'

        return df

    def historical_wave_dating(self) -> List[Dict]:
        """
        Return historical dating of Kondratiev waves based on literature.

        Based on various scholars' periodizations:
        - Kondratiev (original)
        - Schumpeter
        - Mandel
        - Freeman & Louçã

        Returns
        -------
        List[Dict]
            Historical long wave periods
        """
        historical_waves = [
            {
                'wave': '1st Kondratiev',
                'expansion_start': 1790,
                'peak': 1815,
                'trough': 1848,
                'key_technologies': 'Steam engine, textiles',
                'hegemon': 'Britain (rising)'
            },
            {
                'wave': '2nd Kondratiev',
                'expansion_start': 1848,
                'peak': 1873,
                'trough': 1893,
                'key_technologies': 'Railways, steel',
                'hegemon': 'Britain (peak)'
            },
            {
                'wave': '3rd Kondratiev',
                'expansion_start': 1893,
                'peak': 1914,
                'trough': 1945,
                'key_technologies': 'Electricity, chemicals, automobiles',
                'hegemon': 'Britain (decline), USA (rising)'
            },
            {
                'wave': '4th Kondratiev',
                'expansion_start': 1945,
                'peak': 1973,
                'trough': 1993,
                'key_technologies': 'Mass production, petrochemicals, consumer durables',
                'hegemon': 'USA (peak)'
            },
            {
                'wave': '5th Kondratiev',
                'expansion_start': 1993,
                'peak': 2008,
                'trough': '2025?',
                'key_technologies': 'ICT, internet, biotechnology',
                'hegemon': 'USA (declining?)'
            }
        ]

        return historical_waves


class SchumpeterianCycles:
    """
    Analyze multiple cyclical frequencies (Schumpeter's framework).

    Schumpeter identified three cycles:
    - Kitchin (inventory) cycles: ~3-4 years
    - Juglar (fixed investment) cycles: ~8-10 years
    - Kondratiev (innovation) cycles: ~50-60 years
    """

    def __init__(self, data: pd.DataFrame):
        """
        Initialize Schumpeterian cycle analyzer.

        Parameters
        ----------
        data : pd.DataFrame
            Time series data
        """
        self.data = data

    def extract_all_cycles(self, variable: str) -> Dict:
        """
        Extract all three cycle types from a variable.

        Parameters
        ----------
        variable : str
            Variable to decompose

        Returns
        -------
        Dict
            All three cycle components
        """
        df = self.data[self.data[variable].notna()].copy()
        y = df[variable].values

        # Detrend
        t = np.arange(len(y))
        coeffs = np.polyfit(t, y, 1)
        trend = coeffs[0] * t + coeffs[1]
        y_detrended = y - trend

        # Extract each cycle using band-pass filters
        def extract_cycle(y_data, period_low, period_high):
            freq_low = 1 / period_high
            freq_high = 1 / period_low
            nyquist = 0.5
            low = freq_low / nyquist
            high = freq_high / nyquist

            # Ensure valid range
            low = max(low, 0.01)
            high = min(high, 0.99)

            if low >= high:
                return np.zeros_like(y_data)

            b, a = butter(N=2, Wn=[low, high], btype='band')
            return filtfilt(b, a, y_data)

        # Kitchin cycle (3-4 years)
        kitchin = extract_cycle(y_detrended, 3, 4)

        # Juglar cycle (7-11 years)
        juglar = extract_cycle(y_detrended, 7, 11)

        # Kondratiev cycle (45-65 years)
        kondratiev = extract_cycle(y_detrended, 45, 65)

        # Create output dataframe
        result_df = df[['year']].copy()
        result_df[f'{variable}_trend'] = trend
        result_df[f'{variable}_kitchin'] = kitchin
        result_df[f'{variable}_juglar'] = juglar
        result_df[f'{variable}_kondratiev'] = kondratiev
        result_df[f'{variable}_residual'] = y_detrended - kitchin - juglar - kondratiev

        return {
            'variable': variable,
            'decomposition': result_df,
            'trend': trend,
            'kitchin_cycle': kitchin,
            'juglar_cycle': juglar,
            'kondratiev_cycle': kondratiev
        }


class TechnologyRevolutions:
    """
    Analyze technology revolutions and techno-economic paradigms.

    Based on Carlota Perez's framework:
    Each revolution has:
    - Installation period (new technology, financial bubble)
    - Turning point (crisis, institutional change)
    - Deployment period (mature growth, widespread adoption)
    """

    @staticmethod
    def get_historical_tech_revolutions() -> List[Dict]:
        """
        Return historical technology revolutions (Perez framework).

        Returns
        -------
        List[Dict]
            Technology revolution periods and characteristics
        """
        revolutions = [
            {
                'revolution': 'Industrial Revolution',
                'installation_start': 1771,
                'turning_point': 1793,
                'deployment_end': 1829,
                'core_country': 'Britain',
                'key_technologies': ['Mechanized cotton', 'Wrought iron', 'Canals'],
                'infrastructure': 'Canals and waterways'
            },
            {
                'revolution': 'Age of Steam and Railways',
                'installation_start': 1829,
                'turning_point': 1848,
                'deployment_end': 1873,
                'core_country': 'Britain',
                'key_technologies': ['Steam power', 'Railways', 'Telegraph'],
                'infrastructure': 'Railway networks'
            },
            {
                'revolution': 'Age of Steel and Heavy Engineering',
                'installation_start': 1875,
                'turning_point': 1893,
                'deployment_end': 1918,
                'core_country': 'USA, Germany',
                'key_technologies': ['Steel', 'Heavy chemicals', 'Electricity'],
                'infrastructure': 'Steel mills, electrical networks'
            },
            {
                'revolution': 'Age of Oil, Automobiles and Mass Production',
                'installation_start': 1908,
                'turning_point': 1929,
                'deployment_end': 1974,
                'core_country': 'USA',
                'key_technologies': ['Internal combustion', 'Petrochemicals', 'Mass production'],
                'infrastructure': 'Roads, highways, universal electricity'
            },
            {
                'revolution': 'Age of Information and Telecommunications',
                'installation_start': 1971,
                'turning_point': 2000,
                'deployment_end': 'Ongoing',
                'core_country': 'USA',
                'key_technologies': ['Microelectronics', 'Internet', 'Biotechnology'],
                'infrastructure': 'Digital networks, fiber optics'
            }
        ]

        return revolutions

    @staticmethod
    def classify_tech_phase(year: int) -> Dict:
        """
        Classify a year according to technology revolution phase.

        Parameters
        ----------
        year : int
            Year to classify

        Returns
        -------
        Dict
            Technology revolution and phase
        """
        revolutions = TechnologyRevolutions.get_historical_tech_revolutions()

        for rev in revolutions:
            install_start = rev['installation_start']
            turning_point = rev['turning_point']
            deploy_end = rev['deployment_end']

            if isinstance(deploy_end, str):  # Ongoing
                deploy_end = 2030

            if install_start <= year < turning_point:
                return {
                    'year': year,
                    'revolution': rev['revolution'],
                    'phase': 'Installation',
                    'technologies': rev['key_technologies']
                }
            elif turning_point <= year <= deploy_end:
                return {
                    'year': year,
                    'revolution': rev['revolution'],
                    'phase': 'Deployment',
                    'technologies': rev['key_technologies']
                }

        return {
            'year': year,
            'revolution': 'Unknown',
            'phase': 'Unknown',
            'technologies': []
        }


if __name__ == '__main__':
    print("Long Wave Analysis module loaded successfully.")
    print("\nAvailable classes:")
    print("- LongWaveAnalyzer: Kondratiev wave detection and analysis")
    print("- SchumpeterianCycles: Multi-frequency cycle decomposition")
    print("- TechnologyRevolutions: Perez techno-economic paradigms")
