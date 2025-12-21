from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import signal
from scipy.interpolate import interp1d
from scipy.spatial import ConvexHull
from scipy.stats import pearsonr

# Optional imports for visualization (will be checked at runtime)
try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

try:
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


@dataclass
class RespiratoryMetrics:
    """Data class for respiratory analysis results."""

    breathing_rate: float
    tidal_volume_change: float
    chest_wall_displacement: float
    abdominal_displacement: float
    ribcage_displacement: float
    breathing_pattern_regularity: float
    inspiratory_time: float
    expiratory_time: float
    ie_ratio: float
    volume_synchrony: float
    respiratory_effort: float
    breathing_efficiency: float
    regional_contribution: Dict[str, float]
    wave_synchrony: Dict[str, Any]
    breathing_correctness: Dict[str, Any]


class GraphPreprocessor:
    def __init__(self):

        rib_cage_points_set_front = [
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            10,
            11,
            12,
            13,
            14,
            15,
            16,
            17,
            18,
            19,
        ]
        rib_cage_points_set_back = [43, 44, 45, 46, 47, 48, 49, 50, 51, 52]
        rib_cage_points_set = rib_cage_points_set_front + rib_cage_points_set_back

        abdominal_points_set_front = [15, 16, 17, 18, 19, 21, 22, 23, 24, 25, 26, 27]
        abdominal_points_set_back = [53, 54, 55, 56, 57, 58, 59, 60, 61]
        abdominal_points_set = abdominal_points_set_front + abdominal_points_set_back

        abdomen_points_set_front = [
            28,
            29,
            30,
            31,
            32,
            33,
            34,
            35,
            36,
            37,
            38,
            39,
            40,
            41,
            42,
        ]
        abdomen_points_set_back = [
            62,
            63,
            64,
            65,
            66,
            67,
            68,
            69,
            70,
            71,
            72,
            73,
            74,
            75,
            76,
            77,
            78,
            79,
            80,
            81,
        ]
        abdomen_points_set = abdomen_points_set_front + abdomen_points_set_back

        self.points_set = {}
        self.points_set["rib_cage"] = rib_cage_points_set
        self.points_set["abdominal"] = abdominal_points_set
        self.points_set["abdomen"] = abdomen_points_set


@pd.api.extensions.register_dataframe_accessor("graph")
class GraphCruncher:
    def __init__(self, pandas_obj):
        self._obj = pandas_obj
        self.preprocessor = GraphPreprocessor()
        self.sampling_frequency = self._estimate_sampling_frequency()

    def _estimate_sampling_frequency(self) -> float:
        """Estimate sampling frequency from time data."""
        if "Time" in self._obj.columns and len(self._obj) > 1:
            time_diff = np.diff(self._obj["Time"])
            avg_time_diff = np.median(time_diff[time_diff > 0])
            return 1.0 / avg_time_diff if avg_time_diff > 0 else 60.0  # Default 60 Hz
        return 60.0  # Default assumption

    def calculate_volume(self, row_):
        """
        Original volume calculation method for backward compatibility.
        row = 0
        pd.DataFrame(point_tracks.iloc[row,2:].values.reshape(89,3), columns = ['X', 'Y', 'Z'])
        """
        df_vol_row = pd.DataFrame(self._obj.iloc[row_, 2:].values.reshape(89, 3), columns=["X", "Y", "Z"])
        hull = ConvexHull(df_vol_row.values)
        return hull.volume

    def compute_volume(self, row, selection_list: list = None):
        """Compute convex hull volume for selected points at a given time step."""
        points = []
        if not selection_list:
            # Default: use all available points (infer from columns)
            point_columns = [col for col in row.index if ".X" in col]
            point_numbers = [int(col.split(".")[0]) for col in point_columns]
            selection_list = sorted(set(point_numbers))

        for i in selection_list:
            x_col = f"{i}.X"
            y_col = f"{i}.Y"
            z_col = f"{i}.Z"

            # Check if columns exist
            if x_col in row.index and y_col in row.index and z_col in row.index:
                x = row[x_col]
                y = row[y_col]
                z = row[z_col]

                if pd.notna(x) and pd.notna(y) and pd.notna(z):  # Handle NaN values
                    points.append([x, y, z])

        if len(points) < 4:  # Need at least 4 points for convex hull
            return np.nan

        points = np.array(points)
        try:
            hull = ConvexHull(points)
            return hull.volume
        except Exception as e:
            print(f"Error computing volume: {e}")
            return np.nan

    def compute_regional_volumes(self) -> Dict[str, np.ndarray]:
        """Compute volumes for different anatomical regions over time."""
        regional_volumes = {}

        for region, point_list in self.preprocessor.points_set.items():
            volumes = []
            for _, row in self._obj.iterrows():
                volume = self.compute_volume(row, point_list)
                volumes.append(volume)
            regional_volumes[region] = np.array(volumes)

        # Also compute total volume
        total_volumes = []
        for _, row in self._obj.iterrows():
            volume = self.compute_volume(row)
            total_volumes.append(volume)
        regional_volumes["total"] = np.array(total_volumes)

        return regional_volumes

    def compute_surface_area(self, row, selection_list: list = None) -> float:
        """Compute surface area of the convex hull."""
        points = []
        if not selection_list:
            # Default: use all available points (infer from columns)
            point_columns = [col for col in row.index if ".X" in col]
            point_numbers = [int(col.split(".")[0]) for col in point_columns]
            selection_list = sorted(set(point_numbers))

        for i in selection_list:
            x_col = f"{i}.X"
            y_col = f"{i}.Y"
            z_col = f"{i}.Z"

            if x_col in row.index and y_col in row.index and z_col in row.index:
                x = row[x_col]
                y = row[y_col]
                z = row[z_col]

                if pd.notna(x) and pd.notna(y) and pd.notna(z):
                    points.append([x, y, z])

        if len(points) < 4:
            return np.nan

        points = np.array(points)
        try:
            hull = ConvexHull(points)
            return hull.area
        except Exception as e:
            print(f"Error computing surface area: {e}")
            return np.nan

    def compute_centroid_displacement(self, selection_list: list = None) -> np.ndarray:
        """Compute displacement of the centroid over time."""
        centroids = []
        if not selection_list:
            # Default: use all available points (infer from columns)
            first_row = self._obj.iloc[0]
            point_columns = [col for col in first_row.index if ".X" in col]
            point_numbers = [int(col.split(".")[0]) for col in point_columns]
            selection_list = sorted(set(point_numbers))

        for _, row in self._obj.iterrows():
            points = []
            for i in selection_list:
                x_col = f"{i}.X"
                y_col = f"{i}.Y"
                z_col = f"{i}.Z"

                if x_col in row.index and y_col in row.index and z_col in row.index:
                    x = row[x_col]
                    y = row[y_col]
                    z = row[z_col]

                    if pd.notna(x) and pd.notna(y) and pd.notna(z):
                        points.append([x, y, z])

            if points:
                centroid = np.mean(points, axis=0)
                centroids.append(centroid)
            else:
                centroids.append([np.nan, np.nan, np.nan])

        centroids = np.array(centroids)

        # Calculate displacement from first valid centroid
        first_valid_idx = np.where(~np.isnan(centroids).any(axis=1))[0]
        if len(first_valid_idx) > 0:
            reference = centroids[first_valid_idx[0]]
            displacements = np.linalg.norm(centroids - reference, axis=1)
            return displacements
        else:
            return np.full(len(centroids), np.nan)

    def detect_breathing_cycles(self, volume_signal: np.ndarray, prominence_factor: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detect inspiratory and expiratory peaks in volume signal.

        Returns:
        --------
        Tuple[np.ndarray, np.ndarray]
            (inspiratory_peaks, expiratory_peaks) - indices of peaks
        """
        # Remove NaN values and interpolate if necessary
        valid_idx = ~np.isnan(volume_signal)
        if not np.any(valid_idx):
            return np.array([]), np.array([])

        # Smooth the signal
        smoothed = signal.savgol_filter(
            volume_signal[valid_idx],
            window_length=min(51, len(volume_signal[valid_idx]) // 4 * 2 + 1),
            polyorder=3,
        )

        # Calculate prominence threshold
        signal_range = np.ptp(smoothed)
        prominence_threshold = prominence_factor * signal_range

        # Find inspiratory peaks (maxima)
        insp_peaks, _ = signal.find_peaks(
            smoothed,
            prominence=prominence_threshold,
            distance=int(self.sampling_frequency * 1.5),
        )  # Min 1.5s between peaks

        # Find expiratory peaks (minima)
        exp_peaks, _ = signal.find_peaks(
            -smoothed,
            prominence=prominence_threshold,
            distance=int(self.sampling_frequency * 1.5),
        )

        # Map back to original indices
        valid_indices = np.where(valid_idx)[0]
        insp_peaks_original = valid_indices[insp_peaks] if len(insp_peaks) > 0 else np.array([])
        exp_peaks_original = valid_indices[exp_peaks] if len(exp_peaks) > 0 else np.array([])

        return insp_peaks_original, exp_peaks_original

    def calculate_breathing_rate(self, volume_signal: np.ndarray) -> float:
        """Calculate breathing rate in breaths per minute."""
        insp_peaks, _ = self.detect_breathing_cycles(volume_signal)

        if len(insp_peaks) < 2:
            return np.nan

        # Calculate time between peaks
        peak_times = self._obj["Time"].iloc[insp_peaks].values
        intervals = np.diff(peak_times)

        if len(intervals) == 0:
            return np.nan

        avg_interval = np.mean(intervals)
        breathing_rate = 60.0 / avg_interval  # Convert to breaths per minute

        return breathing_rate

    def calculate_tidal_volume_changes(self, volume_signal: np.ndarray) -> float:
        """Calculate average tidal volume change."""
        insp_peaks, exp_peaks = self.detect_breathing_cycles(volume_signal)

        if len(insp_peaks) == 0 or len(exp_peaks) == 0:
            return np.nan

        # Match inspiratory and expiratory peaks
        tidal_changes = []
        for insp_peak in insp_peaks:
            # Find closest expiratory peak
            exp_diffs = np.abs(exp_peaks - insp_peak)
            if len(exp_diffs) > 0:
                closest_exp_idx = np.argmin(exp_diffs)
                exp_peak = exp_peaks[closest_exp_idx]

                if abs(insp_peak - exp_peak) < len(volume_signal) * 0.1:  # Within 10% of signal length
                    tidal_change = abs(volume_signal[insp_peak] - volume_signal[exp_peak])
                    tidal_changes.append(tidal_change)

        return np.mean(tidal_changes) if tidal_changes else np.nan

    def calculate_breathing_pattern_regularity(self, volume_signal: np.ndarray) -> float:
        """Calculate regularity of breathing pattern (lower values = more regular)."""
        insp_peaks, _ = self.detect_breathing_cycles(volume_signal)

        if len(insp_peaks) < 3:
            return np.nan

        # Calculate intervals between peaks
        peak_times = self._obj["Time"].iloc[insp_peaks].values
        intervals = np.diff(peak_times)

        if len(intervals) < 2:
            return np.nan

        # Coefficient of variation (std/mean)
        regularity = np.std(intervals) / np.mean(intervals)

        return regularity

    def calculate_ie_ratio(self, volume_signal: np.ndarray) -> Tuple[float, float, float]:
        """
        Calculate inspiratory/expiratory ratio and times.

        Returns:
        --------
        Tuple[float, float, float]
            (inspiratory_time, expiratory_time, ie_ratio)
        """
        insp_peaks, exp_peaks = self.detect_breathing_cycles(volume_signal)

        if len(insp_peaks) == 0 or len(exp_peaks) == 0:
            return np.nan, np.nan, np.nan

        # Create combined peaks array with labels
        all_peaks = []
        for peak in insp_peaks:
            all_peaks.append((peak, "insp"))
        for peak in exp_peaks:
            all_peaks.append((peak, "exp"))

        # Sort by time
        all_peaks.sort(key=lambda x: x[0])

        inspiratory_times = []
        expiratory_times = []

        # Calculate phase durations
        for i in range(len(all_peaks) - 1):
            current_peak, current_type = all_peaks[i]
            next_peak, next_type = all_peaks[i + 1]

            duration = self._obj["Time"].iloc[next_peak] - self._obj["Time"].iloc[current_peak]

            if current_type == "exp" and next_type == "insp":
                inspiratory_times.append(duration)
            elif current_type == "insp" and next_type == "exp":
                expiratory_times.append(duration)

        if not inspiratory_times or not expiratory_times:
            return np.nan, np.nan, np.nan

        avg_insp_time = np.mean(inspiratory_times)
        avg_exp_time = np.mean(expiratory_times)
        ie_ratio = avg_insp_time / avg_exp_time if avg_exp_time > 0 else np.nan

        return avg_insp_time, avg_exp_time, ie_ratio

    def calculate_volume_synchrony(self, regional_volumes: Dict[str, np.ndarray]) -> float:
        """Calculate synchrony between different regional volumes."""
        if "rib_cage" not in regional_volumes or "abdominal" not in regional_volumes:
            return np.nan

        rib_cage_vol = regional_volumes["rib_cage"]
        abdominal_vol = regional_volumes["abdominal"]

        # Remove NaN values
        valid_idx = ~(np.isnan(rib_cage_vol) | np.isnan(abdominal_vol))
        if np.sum(valid_idx) < 10:  # Need sufficient data points
            return np.nan

        try:
            correlation, _ = pearsonr(rib_cage_vol[valid_idx], abdominal_vol[valid_idx])
            return correlation
        except Exception as e:
            print(f"Error calculating volume synchrony: {e}")
            return np.nan

    def calculate_respiratory_effort(self, volume_signal: np.ndarray) -> float:
        """Calculate respiratory effort based on volume variability."""
        if len(volume_signal) < 10 or np.all(np.isnan(volume_signal)):
            return np.nan

        valid_signal = volume_signal[~np.isnan(volume_signal)]
        if len(valid_signal) < 10:
            return np.nan

        # Normalize signal
        normalized_signal = (valid_signal - np.mean(valid_signal)) / np.std(valid_signal)

        # Calculate effort as RMS of the derivative (rate of change)
        derivative = np.diff(normalized_signal)
        effort = np.sqrt(np.mean(derivative**2))

        return effort

    def calculate_breathing_efficiency(self, volume_signal: np.ndarray) -> float:
        """Calculate breathing efficiency (volume change per unit effort)."""
        tidal_volume = self.calculate_tidal_volume_changes(volume_signal)
        effort = self.calculate_respiratory_effort(volume_signal)

        if np.isnan(tidal_volume) or np.isnan(effort) or effort == 0:
            return np.nan

        efficiency = tidal_volume / effort
        return efficiency

    def calculate_regional_contributions(self, regional_volumes: Dict[str, np.ndarray]) -> Dict[str, float]:
        """Calculate relative contribution of each region to total breathing."""
        contributions = {}

        if "total" not in regional_volumes:
            return contributions

        total_variation = np.nanvar(regional_volumes["total"])

        if total_variation == 0 or np.isnan(total_variation):
            return {region: np.nan for region in regional_volumes.keys() if region != "total"}

        for region, volumes in regional_volumes.items():
            if region != "total":
                regional_variation = np.nanvar(volumes)
                contribution = regional_variation / total_variation if not np.isnan(regional_variation) else np.nan
                contributions[region] = contribution

        return contributions

    def calculate_wave_synchrony(self, regional_volumes: Dict[str, np.ndarray]) -> Dict[str, float]:
        """
        Calculate wave synchrony between different body regions during breathing.
        This measures how synchronous the body movement is in waves.

        Returns:
        --------
        Dict[str, float]
            Synchrony metrics including phase coherence and cross-correlations
        """
        synchrony_metrics = {}

        # Get the main regions for analysis
        regions = ["rib_cage", "abdominal", "abdomen", "total"]
        available_regions = {region: volumes for region, volumes in regional_volumes.items() if region in regions and not np.all(np.isnan(volumes))}

        if len(available_regions) < 2:
            return {"overall_synchrony": np.nan, "phase_coherence": np.nan}

        # Calculate cross-correlations between regions
        correlations = {}
        for i, (region1, vol1) in enumerate(available_regions.items()):
            for j, (region2, vol2) in enumerate(available_regions.items()):
                if i < j:  # Avoid duplicate pairs
                    valid_idx = ~(np.isnan(vol1) | np.isnan(vol2))
                    if np.sum(valid_idx) > 10:
                        try:
                            corr, _ = pearsonr(vol1[valid_idx], vol2[valid_idx])
                            correlations[f"{region1}_{region2}"] = corr
                        except Exception as e:
                            print(f"Error calculating correlation between {region1} and {region2}: {e}")
                            correlations[f"{region1}_{region2}"] = np.nan

        # Calculate phase coherence using Hilbert transform
        phase_coherences = {}
        try:
            from scipy.signal import hilbert

            for region_pair, corr in correlations.items():
                if not np.isnan(corr):
                    regions_split = region_pair.split("_")
                    if len(regions_split) >= 2:
                        # Handle compound region names like 'rib_cage'
                        region1 = regions_split[0]
                        region2 = "_".join(regions_split[1:])

                        # Check if regions exist in available_regions
                        if region1 in available_regions and region2 in available_regions:
                            vol1 = available_regions[region1]
                            vol2 = available_regions[region2]
                        else:
                            # Try alternative parsing for compound names
                            for i in range(1, len(regions_split)):
                                alt_region1 = "_".join(regions_split[:i])
                                alt_region2 = "_".join(regions_split[i:])
                                if alt_region1 in available_regions and alt_region2 in available_regions:
                                    region1, region2 = alt_region1, alt_region2
                                    vol1 = available_regions[region1]
                                    vol2 = available_regions[region2]
                                    break
                            else:
                                continue  # Skip this pair if regions not found

                    # Remove NaN and apply Hilbert transform
                    valid_idx = ~(np.isnan(vol1) | np.isnan(vol2))
                    if np.sum(valid_idx) > 50:  # Need sufficient data for phase analysis
                        clean_vol1 = vol1[valid_idx]
                        clean_vol2 = vol2[valid_idx]

                        # Smooth signals before phase analysis
                        clean_vol1 = signal.savgol_filter(
                            clean_vol1,
                            window_length=min(31, len(clean_vol1) // 4 * 2 + 1),
                            polyorder=3,
                        )
                        clean_vol2 = signal.savgol_filter(
                            clean_vol2,
                            window_length=min(31, len(clean_vol2) // 4 * 2 + 1),
                            polyorder=3,
                        )

                        # Calculate instantaneous phases
                        analytic1 = hilbert(clean_vol1)
                        analytic2 = hilbert(clean_vol2)
                        phase1 = np.angle(analytic1)
                        phase2 = np.angle(analytic2)

                        # Calculate phase difference
                        phase_diff = np.angle(np.exp(1j * (phase1 - phase2)))

                        # Phase coherence is consistency of phase relationship
                        # Higher values (closer to 1) indicate better synchrony
                        phase_coherence = np.abs(np.mean(np.exp(1j * phase_diff)))
                        phase_coherences[region_pair] = phase_coherence

        except ImportError:
            # Fallback if scipy.signal.hilbert is not available
            phase_coherences = {}

        # Calculate time-lag cross-correlations to detect synchrony
        lag_correlations = {}
        max_lag = min(int(self.sampling_frequency * 2), 60)  # Max 2 seconds or 60 samples lag

        for region_pair, _ in correlations.items():
            regions_split = region_pair.split("_")
            if len(regions_split) >= 2:
                # Handle compound region names like 'rib_cage'
                region1 = regions_split[0]
                region2 = "_".join(regions_split[1:])

                # Check if regions exist in available_regions
                if region1 in available_regions and region2 in available_regions:
                    vol1 = available_regions[region1]
                    vol2 = available_regions[region2]
                else:
                    # Try alternative parsing for compound names
                    for i in range(1, len(regions_split)):
                        alt_region1 = "_".join(regions_split[:i])
                        alt_region2 = "_".join(regions_split[i:])
                        if alt_region1 in available_regions and alt_region2 in available_regions:
                            region1, region2 = alt_region1, alt_region2
                            vol1 = available_regions[region1]
                            vol2 = available_regions[region2]
                            break
                    else:
                        continue  # Skip this pair if regions not found
            else:
                continue

            valid_idx = ~(np.isnan(vol1) | np.isnan(vol2))
            if np.sum(valid_idx) > 100:  # Need sufficient data
                clean_vol1 = vol1[valid_idx]
                clean_vol2 = vol2[valid_idx]

                # Calculate cross-correlation at different lags
                cross_corr = signal.correlate(clean_vol1, clean_vol2, mode="full")
                lags = signal.correlation_lags(len(clean_vol1), len(clean_vol2), mode="full")

                # Focus on reasonable lag range
                valid_lag_idx = np.abs(lags) <= max_lag
                if np.any(valid_lag_idx):
                    max_corr_idx = np.argmax(np.abs(cross_corr[valid_lag_idx]))
                    max_correlation = cross_corr[valid_lag_idx][max_corr_idx]
                    optimal_lag = lags[valid_lag_idx][max_corr_idx]

                    lag_correlations[region_pair] = {
                        "max_correlation": max_correlation,
                        "optimal_lag": optimal_lag,
                        "lag_samples": optimal_lag,
                        "lag_seconds": optimal_lag / self.sampling_frequency,
                    }

        # Calculate overall synchrony score
        if correlations:
            overall_correlation = np.nanmean(list(correlations.values()))
        else:
            overall_correlation = np.nan

        if phase_coherences:
            overall_phase_coherence = np.nanmean(list(phase_coherences.values()))
        else:
            overall_phase_coherence = np.nan

        # Calculate breathing wave quality
        wave_quality = self._calculate_breathing_wave_quality(available_regions)

        synchrony_metrics.update(
            {
                "overall_synchrony": overall_correlation,
                "phase_coherence": overall_phase_coherence,
                "regional_correlations": correlations,
                "phase_coherences": phase_coherences,
                "lag_correlations": lag_correlations,
                "wave_quality": wave_quality,
            }
        )

        return synchrony_metrics

    def _calculate_breathing_wave_quality(self, regional_volumes: Dict[str, np.ndarray]) -> Dict[str, float]:
        """
        Calculate the quality of breathing waves by analyzing the smoothness
        and periodicity of volume changes.
        """
        wave_quality = {}

        for region, volumes in regional_volumes.items():
            if np.all(np.isnan(volumes)):
                wave_quality[region] = np.nan
                continue

            valid_volumes = volumes[~np.isnan(volumes)]
            if len(valid_volumes) < 50:
                wave_quality[region] = np.nan
                continue

            # Smooth the signal
            smoothed = signal.savgol_filter(
                valid_volumes,
                window_length=min(31, len(valid_volumes) // 4 * 2 + 1),
                polyorder=3,
            )

            # Calculate signal-to-noise ratio
            signal_power = np.var(smoothed)
            noise_power = np.var(valid_volumes - smoothed)
            snr = signal_power / noise_power if noise_power > 0 else np.inf

            # Calculate periodicity using autocorrelation
            autocorr = signal.correlate(smoothed, smoothed, mode="full")
            autocorr = autocorr[autocorr.size // 2 :]
            autocorr = autocorr / autocorr[0]  # Normalize

            # Find periodic peaks in autocorrelation
            min_period_samples = int(self.sampling_frequency * 2)  # Min 2 seconds
            max_period_samples = int(self.sampling_frequency * 8)  # Max 8 seconds

            if len(autocorr) > max_period_samples:
                search_range = autocorr[min_period_samples:max_period_samples]
                peaks, _ = signal.find_peaks(search_range, height=0.3)  # Find significant peaks

                if len(peaks) > 0:
                    # Periodicity strength is the height of the highest peak
                    periodicity = np.max(search_range[peaks])
                else:
                    periodicity = 0
            else:
                periodicity = 0

            # Combined wave quality score (0 to 1, higher is better)
            # Combines signal-to-noise ratio and periodicity
            snr_normalized = min(snr / 10.0, 1.0)  # Cap at 10:1 SNR
            wave_quality_score = (snr_normalized + periodicity) / 2.0

            wave_quality[region] = wave_quality_score

        return wave_quality

    def assess_breathing_correctness(self, threshold_synchrony: float = 0.7, threshold_phase_coherence: float = 0.6) -> Dict[str, Any]:
        """
        Assess whether the person is breathing correctly based on wave synchrony.

        Parameters:
        -----------
        threshold_synchrony : float
            Minimum correlation threshold for good synchrony (default: 0.7)
        threshold_phase_coherence : float
            Minimum phase coherence threshold for correct breathing (default: 0.6)

        Returns:
        --------
        Dict[str, Any]
            Assessment results with interpretation and recommendations
        """
        regional_volumes = self.compute_regional_volumes()
        synchrony_metrics = self.calculate_wave_synchrony(regional_volumes)

        # Extract key metrics
        overall_synchrony = synchrony_metrics.get("overall_synchrony", np.nan)
        phase_coherence = synchrony_metrics.get("phase_coherence", np.nan)
        wave_quality = synchrony_metrics.get("wave_quality", {})

        # Overall synchrony assessment
        if np.isnan(overall_synchrony):
            sync_assessment = "Unable to assess - insufficient data"
            sync_status = "unknown"
        elif overall_synchrony >= threshold_synchrony:
            sync_assessment = "Good synchrony - body regions move together well"
            sync_status = "good"
        elif overall_synchrony >= 0.5:
            sync_assessment = "Moderate synchrony - some coordination between regions"
            sync_status = "moderate"
        else:
            sync_assessment = "Poor synchrony - body regions not well coordinated"
            sync_status = "poor"

        # Phase coherence assessment
        if np.isnan(phase_coherence):
            phase_assessment = "Unable to assess phase relationships"
            phase_status = "unknown"
        elif phase_coherence >= threshold_phase_coherence:
            phase_assessment = "Good phase coherence - breathing waves are well coordinated"
            phase_status = "good"
        elif phase_coherence >= 0.4:
            phase_assessment = "Moderate phase coherence - some timing coordination"
            phase_status = "moderate"
        else:
            phase_assessment = "Poor phase coherence - breathing waves not well timed"
            phase_status = "poor"

        # Wave quality assessment
        avg_wave_quality = np.nanmean(list(wave_quality.values())) if wave_quality else np.nan
        if np.isnan(avg_wave_quality):
            quality_assessment = "Unable to assess wave quality"
            quality_status = "unknown"
        elif avg_wave_quality >= 0.7:
            quality_assessment = "High quality breathing waves - smooth and periodic"
            quality_status = "good"
        elif avg_wave_quality >= 0.5:
            quality_assessment = "Moderate quality breathing waves"
            quality_status = "moderate"
        else:
            quality_assessment = "Low quality breathing waves - irregular or noisy"
            quality_status = "poor"

        # Overall breathing correctness
        status_scores = {"good": 3, "moderate": 2, "poor": 1, "unknown": 0}
        scores = [
            status_scores.get(sync_status, 0),
            status_scores.get(phase_status, 0),
            status_scores.get(quality_status, 0),
        ]

        avg_score = np.mean([s for s in scores if s > 0]) if any(s > 0 for s in scores) else 0

        if avg_score >= 2.5:
            overall_status = "correct"
            overall_assessment = "Breathing pattern appears correct - good synchrony and coordination"
        elif avg_score >= 1.5:
            overall_status = "partially_correct"
            overall_assessment = "Breathing pattern partially correct - some areas for improvement"
        elif avg_score >= 0.5:
            overall_status = "incorrect"
            overall_assessment = "Breathing pattern appears incorrect - poor coordination"
        else:
            overall_status = "unknown"
            overall_assessment = "Unable to assess breathing correctness due to insufficient data"

        # Recommendations
        recommendations = []
        if sync_status == "poor":
            recommendations.append("Focus on coordinating chest and abdominal movement")
        if phase_status == "poor":
            recommendations.append("Work on timing - ensure smooth transitions between inhalation and exhalation")
        if quality_status == "poor":
            recommendations.append("Practice slower, more controlled breathing to improve wave quality")
        if not recommendations:
            recommendations.append("Breathing pattern appears healthy")

        return {
            "overall_status": overall_status,
            "overall_assessment": overall_assessment,
            "detailed_assessment": {
                "synchrony": {
                    "status": sync_status,
                    "value": (round(overall_synchrony, 3) if not np.isnan(overall_synchrony) else None),
                    "assessment": sync_assessment,
                },
                "phase_coherence": {
                    "status": phase_status,
                    "value": (round(phase_coherence, 3) if not np.isnan(phase_coherence) else None),
                    "assessment": phase_assessment,
                },
                "wave_quality": {
                    "status": quality_status,
                    "average_value": (round(avg_wave_quality, 3) if not np.isnan(avg_wave_quality) else None),
                    "assessment": quality_assessment,
                    "regional_quality": {k: round(v, 3) if not np.isnan(v) else None for k, v in wave_quality.items()},
                },
            },
            "recommendations": recommendations,
            "thresholds_used": {
                "synchrony_threshold": threshold_synchrony,
                "phase_coherence_threshold": threshold_phase_coherence,
            },
            "raw_metrics": synchrony_metrics,
        }

    def compute_comprehensive_analysis(self) -> RespiratoryMetrics:
        """
        Perform comprehensive respiratory analysis on the 3D point data.

        Returns:
        --------
        RespiratoryMetrics
            Comprehensive analysis results
        """
        # Compute regional volumes
        regional_volumes = self.compute_regional_volumes()

        # Use total volume for main breathing analysis
        total_volume = regional_volumes["total"]

        # Calculate all metrics
        breathing_rate = self.calculate_breathing_rate(total_volume)
        tidal_volume_change = self.calculate_tidal_volume_changes(total_volume)

        # Displacement metrics
        chest_displacement = np.nanstd(self.compute_centroid_displacement(self.preprocessor.points_set["rib_cage"]))
        abdominal_displacement = np.nanstd(self.compute_centroid_displacement(self.preprocessor.points_set["abdominal"]))
        ribcage_displacement = chest_displacement  # Same as chest for this analysis

        # Pattern analysis
        pattern_regularity = self.calculate_breathing_pattern_regularity(total_volume)
        insp_time, exp_time, ie_ratio = self.calculate_ie_ratio(total_volume)

        # Coordination and efficiency
        volume_synchrony = self.calculate_volume_synchrony(regional_volumes)
        respiratory_effort = self.calculate_respiratory_effort(total_volume)
        breathing_efficiency = self.calculate_breathing_efficiency(total_volume)
        regional_contribution = self.calculate_regional_contributions(regional_volumes)

        # New wave synchrony and breathing correctness analysis
        wave_synchrony = self.calculate_wave_synchrony(regional_volumes)
        breathing_correctness = self.assess_breathing_correctness()

        return RespiratoryMetrics(
            breathing_rate=breathing_rate,
            tidal_volume_change=tidal_volume_change,
            chest_wall_displacement=chest_displacement,
            abdominal_displacement=abdominal_displacement,
            ribcage_displacement=ribcage_displacement,
            breathing_pattern_regularity=pattern_regularity,
            inspiratory_time=insp_time,
            expiratory_time=exp_time,
            ie_ratio=ie_ratio,
            volume_synchrony=volume_synchrony,
            respiratory_effort=respiratory_effort,
            breathing_efficiency=breathing_efficiency,
            regional_contribution=regional_contribution,
            wave_synchrony=wave_synchrony,
            breathing_correctness=breathing_correctness,
        )

    def compute_time_series_analysis(self) -> Dict[str, Any]:
        """
        Compute time series data for frontend visualization.

        Returns:
        --------
        Dict[str, Any]
            Time series data including volumes, displacements, and breathing phases
        """
        # Get time array
        time_array = self._obj["Time"].values

        # Compute regional volumes over time
        regional_volumes = self.compute_regional_volumes()

        # Compute displacements over time
        chest_displacement = self.compute_centroid_displacement(self.preprocessor.points_set["rib_cage"])
        abdominal_displacement = self.compute_centroid_displacement(self.preprocessor.points_set["abdominal"])
        total_displacement = self.compute_centroid_displacement()

        # Detect breathing phases
        insp_peaks, exp_peaks = self.detect_breathing_cycles(regional_volumes["total"])

        # Create breathing phase signal
        breathing_phase = np.zeros(len(time_array))  # 0 = neutral, 1 = inspiration, -1 = expiration

        for peak_idx in insp_peaks:
            if peak_idx < len(breathing_phase):
                breathing_phase[peak_idx] = 1

        for peak_idx in exp_peaks:
            if peak_idx < len(breathing_phase):
                breathing_phase[peak_idx] = -1

        # Compute surface areas over time
        surface_areas = []
        for _, row in self._obj.iterrows():
            surface_area = self.compute_surface_area(row)
            surface_areas.append(surface_area)

        return {
            "time": time_array,
            "volumes": regional_volumes,
            "displacements": {
                "chest": chest_displacement,
                "abdominal": abdominal_displacement,
                "total": total_displacement,
            },
            "surface_areas": np.array(surface_areas),
            "breathing_phases": {
                "phase_signal": breathing_phase,
                "inspiratory_peaks": insp_peaks,
                "expiratory_peaks": exp_peaks,
            },
            "sampling_frequency": self.sampling_frequency,
        }

    def get_analysis_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the respiratory analysis suitable for frontend display.

        Returns:
        --------
        Dict[str, Any]
            Summary statistics and interpretations
        """
        metrics = self.compute_comprehensive_analysis()
        time_series = self.compute_time_series_analysis()

        # Interpret results
        def interpret_breathing_rate(rate):
            if np.isnan(rate):
                return "Unable to determine"
            elif rate < 12:
                return "Below normal (bradypnea)"
            elif rate > 20:
                return "Above normal (tachypnea)"
            else:
                return "Normal"

        def interpret_ie_ratio(ratio):
            if np.isnan(ratio):
                return "Unable to determine"
            elif ratio < 0.8:
                return "Short inspiration relative to expiration"
            elif ratio > 1.2:
                return "Long inspiration relative to expiration"
            else:
                return "Normal ratio"

        def interpret_regularity(regularity):
            if np.isnan(regularity):
                return "Unable to determine"
            elif regularity < 0.1:
                return "Very regular"
            elif regularity < 0.3:
                return "Regular"
            elif regularity < 0.5:
                return "Moderately irregular"
            else:
                return "Irregular"

        summary = {
            "metrics": {
                "breathing_rate_bpm": (round(metrics.breathing_rate, 1) if not np.isnan(metrics.breathing_rate) else None),
                "breathing_rate_interpretation": interpret_breathing_rate(metrics.breathing_rate),
                "tidal_volume_change": (round(metrics.tidal_volume_change, 2) if not np.isnan(metrics.tidal_volume_change) else None),
                "inspiratory_time_sec": (round(metrics.inspiratory_time, 2) if not np.isnan(metrics.inspiratory_time) else None),
                "expiratory_time_sec": (round(metrics.expiratory_time, 2) if not np.isnan(metrics.expiratory_time) else None),
                "ie_ratio": (round(metrics.ie_ratio, 2) if not np.isnan(metrics.ie_ratio) else None),
                "ie_ratio_interpretation": interpret_ie_ratio(metrics.ie_ratio),
                "pattern_regularity": (round(metrics.breathing_pattern_regularity, 3) if not np.isnan(metrics.breathing_pattern_regularity) else None),
                "regularity_interpretation": interpret_regularity(metrics.breathing_pattern_regularity),
                "volume_synchrony": (round(metrics.volume_synchrony, 2) if not np.isnan(metrics.volume_synchrony) else None),
                "breathing_efficiency": (round(metrics.breathing_efficiency, 2) if not np.isnan(metrics.breathing_efficiency) else None),
            },
            "breathing_correctness": {
                "overall_status": metrics.breathing_correctness.get("overall_status", "unknown"),
                "overall_assessment": metrics.breathing_correctness.get("overall_assessment", "Unable to assess"),
                "wave_synchrony_score": metrics.wave_synchrony.get("overall_synchrony", None),
                "phase_coherence_score": metrics.wave_synchrony.get("phase_coherence", None),
                "recommendations": metrics.breathing_correctness.get("recommendations", []),
                "detailed_scores": {
                    "synchrony": metrics.breathing_correctness.get("detailed_assessment", {}).get("synchrony", {}),
                    "phase_coherence": metrics.breathing_correctness.get("detailed_assessment", {}).get("phase_coherence", {}),
                    "wave_quality": metrics.breathing_correctness.get("detailed_assessment", {}).get("wave_quality", {}),
                },
            },
            "regional_analysis": {
                "chest_wall_displacement": (round(metrics.chest_wall_displacement, 2) if not np.isnan(metrics.chest_wall_displacement) else None),
                "abdominal_displacement": (round(metrics.abdominal_displacement, 2) if not np.isnan(metrics.abdominal_displacement) else None),
                "regional_contributions": {region: round(contrib, 3) if not np.isnan(contrib) else None for region, contrib in metrics.regional_contribution.items()},
            },
            "time_series_data": time_series,
            "data_quality": {
                "total_duration_sec": (float(time_series["time"][-1] - time_series["time"][0]) if len(time_series["time"]) > 0 else 0),
                "sampling_frequency_hz": time_series["sampling_frequency"],
                "total_samples": len(time_series["time"]),
                "detected_breaths": len(time_series["breathing_phases"]["inspiratory_peaks"]),
            },
        }

        return summary

    def export_for_frontend(self) -> Dict[str, Any]:
        """
        Export all analysis results in a format optimized for frontend visualization.

        Returns:
        --------
        Dict[str, Any]
            Complete analysis results formatted for frontend consumption including
            clinical report data and averaged breath cycles
        """
        analysis_summary = self.get_analysis_summary()

        # Add visualization-ready data
        frontend_data = analysis_summary.copy()

        # Format time series data for plotting
        time_series = analysis_summary["time_series_data"]

        # Basic time series plots (original functionality)
        frontend_data["plots"] = {
            "volume_over_time": {
                "x": time_series["time"].tolist(),
                "y_total": time_series["volumes"]["total"].tolist(),
                "y_ribcage": time_series["volumes"]["rib_cage"].tolist(),
                "y_abdominal": time_series["volumes"]["abdominal"].tolist(),
                "y_abdomen": time_series["volumes"]["abdomen"].tolist(),
                "breathing_phases": {
                    "inspiratory_peaks": {
                        "x": [time_series["time"][i] for i in time_series["breathing_phases"]["inspiratory_peaks"]],
                        "y": [time_series["volumes"]["total"][i] for i in time_series["breathing_phases"]["inspiratory_peaks"]],
                    },
                    "expiratory_peaks": {
                        "x": [time_series["time"][i] for i in time_series["breathing_phases"]["expiratory_peaks"]],
                        "y": [time_series["volumes"]["total"][i] for i in time_series["breathing_phases"]["expiratory_peaks"]],
                    },
                },
            },
            "displacement_over_time": {
                "x": time_series["time"].tolist(),
                "y_chest": time_series["displacements"]["chest"].tolist(),
                "y_abdominal": time_series["displacements"]["abdominal"].tolist(),
                "y_total": time_series["displacements"]["total"].tolist(),
            },
            "surface_area_over_time": {
                "x": time_series["time"].tolist(),
                "y": time_series["surface_areas"].tolist(),
            },
        }

        # Add clinical report data (new functionality)
        try:
            regional_volumes = time_series["volumes"]
            time_data = time_series["time"]

            # Generate averaged breath cycle data for all compartments
            clinical_data = {
                "averaged_cycles": {},
                "flow_volume_loop": None,
                "regional_contributions": None,
                "cycle_statistics": {},
                "colors": {
                    "rib_cage": "#00BFFF",
                    "abdominal": "#9ACD32",
                    "abdomen": "#FFA500",
                    "total": "#9370DB",
                },
                "display_names": {
                    "rib_cage": "Pulmonary Rib Cage",
                    "abdominal": "Abdominal Rib Cage",
                    "abdomen": "Abdomen",
                    "total": "Chest Wall",
                },
            }

            # Generate averaged cycles for each compartment
            total_cycles = 0
            for key in ["rib_cage", "abdominal", "abdomen", "total"]:
                vol_data = regional_volumes.get(key)
                if vol_data is not None and not np.all(np.isnan(vol_data)):
                    try:
                        perc, mean, std, cycle_count = self._get_averaged_breath_cycle(vol_data, time_data)
                        if perc is not None and mean is not None:
                            clinical_data["averaged_cycles"][key] = {
                                "cycle_percentage": perc.tolist(),
                                "mean_values": mean.tolist(),
                                "std_values": std.tolist(),
                                "cycle_count": cycle_count,
                            }
                            if key == "total":
                                total_cycles = cycle_count
                    except Exception as e:
                        print(f"Warning: Could not generate averaged cycle for {key}: {e}")

            # Generate flow-volume loop data
            if "total" in clinical_data["averaged_cycles"]:
                try:
                    total_vol = regional_volumes.get("total")
                    flow = np.gradient(total_vol, time_data)
                    flow_perc, flow_mean, flow_std, _ = self._get_averaged_breath_cycle(flow, time_data)

                    if flow_perc is not None and flow_mean is not None:
                        # Create tidal volume (normalized to start from 0)
                        vol_mean = clinical_data["averaged_cycles"]["total"]["mean_values"]
                        tidal_vol = np.array(vol_mean) - np.min(vol_mean)

                        clinical_data["flow_volume_loop"] = {
                            "tidal_volume": tidal_vol.tolist(),
                            "flow_rate": flow_mean.tolist(),
                            "flow_std": flow_std.tolist(),
                        }
                except Exception as e:
                    print(f"Warning: Could not generate flow-volume loop: {e}")

            # Add regional contribution data
            try:
                metrics = self.compute_comprehensive_analysis()
                if hasattr(metrics, "regional_contribution") and metrics.regional_contribution:
                    contributions = {}
                    for key, value in metrics.regional_contribution.items():
                        if key in clinical_data["display_names"] and not np.isnan(value):
                            contributions[key] = {
                                "percentage": value * 100,
                                "display_name": clinical_data["display_names"][key],
                                "color": clinical_data["colors"][key],
                            }
                    clinical_data["regional_contributions"] = contributions
            except Exception as e:
                print(f"Warning: Could not generate regional contributions: {e}")

            # Add cycle statistics
            clinical_data["cycle_statistics"] = {
                "total_cycles_detected": total_cycles,
                "analysis_duration_seconds": (float(time_data[-1] - time_data[0]) if len(time_data) > 0 else 0),
                "average_cycle_duration": ((time_data[-1] - time_data[0]) / total_cycles if total_cycles > 0 else 0),
            }

            frontend_data["clinical_report"] = clinical_data

        except Exception as e:
            print(f"Warning: Could not generate clinical report data: {e}")
            frontend_data["clinical_report"] = None

        return frontend_data

    def generate_clinical_report_html(self) -> Optional[str]:
        """
        Generate HTML content for clinical report visualization.

        Returns:
        --------
        str or None
            HTML content for the interactive clinical report, or None if plotly unavailable
        """
        if not PLOTLY_AVAILABLE:
            print("Warning: plotly is required for clinical report HTML generation")
            return None

        try:
            fig = self.create_report_style_interactive_plot()
            if fig is not None:
                return fig.to_html(include_plotlyjs="cdn")
        except Exception as e:
            print(f"Error generating clinical report HTML: {e}")
            return None

        return None

    def visualize_breathing_correctness(self, figsize=(16, 12), save_path=None):
        """
        Create comprehensive visualizations for breathing correctness validation.

        Parameters:
        -----------
        figsize : tuple
            Figure size (width, height)
        save_path : str, optional
            Path to save the visualization

        Returns:
        --------
        matplotlib.figure.Figure
            The created figure
        """
        if not MATPLOTLIB_AVAILABLE:
            print("Error: matplotlib is required for visualization. Install with: pip install matplotlib")
            return None

        # Get analysis data
        regional_volumes = self.compute_regional_volumes()
        wave_synchrony = self.calculate_wave_synchrony(regional_volumes)
        breathing_correctness = self.assess_breathing_correctness()
        time_series = self.compute_time_series_analysis()

        # Create figure with custom grid layout
        fig = plt.figure(figsize=figsize)
        gs = GridSpec(4, 3, figure=fig, hspace=0.35, wspace=0.3)

        # Title with overall assessment
        overall_status = breathing_correctness.get("overall_status", "unknown")
        status_color = {
            "correct": "green",
            "partially_correct": "orange",
            "incorrect": "red",
            "unknown": "gray",
        }
        fig.suptitle(
            f"Breathing Correctness Analysis - Status: {overall_status.upper()}",
            fontsize=16,
            fontweight="bold",
            color=status_color.get(overall_status, "black"),
        )

        time = time_series["time"]
        volumes = time_series["volumes"]

        # Plot 1: Regional Volume Synchrony (spans 2 columns)
        ax1 = fig.add_subplot(gs[0, :2])
        ax1.plot(time, volumes["total"], "k-", linewidth=2, label="Total Volume", alpha=0.8)
        ax1.plot(time, volumes["rib_cage"], "r-", label="Rib Cage", alpha=0.7)
        ax1.plot(time, volumes["abdominal"], "g-", label="Abdominal", alpha=0.7)
        ax1.plot(time, volumes["abdomen"], "b-", label="Abdomen", alpha=0.7)

        # Mark breathing phases
        insp_peaks = time_series["breathing_phases"]["inspiratory_peaks"]
        exp_peaks = time_series["breathing_phases"]["expiratory_peaks"]
        if len(insp_peaks) > 0:
            ax1.scatter(
                time[insp_peaks],
                volumes["total"][insp_peaks],
                color="red",
                s=50,
                marker="^",
                label="Inspiration",
                zorder=5,
            )
        if len(exp_peaks) > 0:
            ax1.scatter(
                time[exp_peaks],
                volumes["total"][exp_peaks],
                color="blue",
                s=50,
                marker="v",
                label="Expiration",
                zorder=5,
            )

        ax1.set_title("Regional Volume Synchrony Over Time")
        ax1.set_ylabel("Volume (arbitrary units)")
        ax1.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        ax1.grid(True, alpha=0.3)

        # Plot 2: Wave Synchrony Metrics (top right)
        ax2 = fig.add_subplot(gs[0, 2])
        synchrony_score = wave_synchrony.get("overall_synchrony", 0)
        phase_coherence = wave_synchrony.get("phase_coherence", 0)
        wave_quality = wave_synchrony.get("wave_quality", {})
        avg_wave_quality = np.nanmean(list(wave_quality.values())) if wave_quality else 0

        metrics_data = [synchrony_score, phase_coherence, avg_wave_quality]
        metrics_labels = ["Synchrony", "Phase\nCoherence", "Wave\nQuality"]
        colors = ["skyblue", "lightgreen", "lightcoral"]

        bars = ax2.bar(
            metrics_labels,
            [x if not np.isnan(x) else 0 for x in metrics_data],
            color=colors,
            alpha=0.7,
            edgecolor="black",
        )

        # Add threshold lines
        ax2.axhline(y=0.7, color="green", linestyle="--", alpha=0.8, label="Good (≥0.7)")
        ax2.axhline(y=0.5, color="orange", linestyle="--", alpha=0.8, label="Moderate (≥0.5)")

        # Add value labels on bars
        for bar, value in zip(bars, metrics_data):
            if not np.isnan(value):
                ax2.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.02,
                    f"{value:.3f}",
                    ha="center",
                    va="bottom",
                    fontweight="bold",
                )

        ax2.set_title("Wave Synchrony Metrics")
        ax2.set_ylabel("Score")
        ax2.set_ylim(0, 1.1)
        ax2.legend(fontsize="small")
        ax2.grid(True, alpha=0.3, axis="y")

        # Plot 3: Cross-Correlation Analysis
        ax3 = fig.add_subplot(gs[1, :])
        regional_correlations = wave_synchrony.get("regional_correlations", {})
        if regional_correlations:
            correlations_list = []

            for pair, corr in regional_correlations.items():
                if not np.isnan(corr):
                    correlations_list.append((pair, corr))

            if correlations_list:
                correlations_list.sort(key=lambda x: abs(x[1]), reverse=True)
                pairs = [item[0] for item in correlations_list]
                corr_values = [item[1] for item in correlations_list]

                bars = ax3.barh(
                    pairs,
                    corr_values,
                    color=["green" if c > 0.7 else "orange" if c > 0.5 else "red" for c in corr_values],
                )

                # Add value labels
                for i, (bar, value) in enumerate(zip(bars, corr_values)):
                    ax3.text(
                        value + 0.01 if value >= 0 else value - 0.01,
                        bar.get_y() + bar.get_height() / 2,
                        f"{value:.3f}",
                        ha="left" if value >= 0 else "right",
                        va="center",
                        fontweight="bold",
                    )

                ax3.axvline(x=0.7, color="green", linestyle="--", alpha=0.8, label="Good (≥0.7)")
                ax3.axvline(
                    x=0.5,
                    color="orange",
                    linestyle="--",
                    alpha=0.8,
                    label="Moderate (≥0.5)",
                )
                ax3.axvline(x=0, color="black", linestyle="-", alpha=0.5)

                ax3.set_title("Regional Cross-Correlations (Higher = Better Synchrony)")
                ax3.set_xlabel("Correlation Coefficient")
                ax3.legend()
                ax3.grid(True, alpha=0.3, axis="x")
                ax3.set_xlim(-1, 1)
        else:
            ax3.text(
                0.5,
                0.5,
                "No correlation data available",
                ha="center",
                va="center",
                transform=ax3.transAxes,
                fontsize=12,
            )
            ax3.set_title("Regional Cross-Correlations")

        # Plot 4: Phase Coherence Analysis
        ax4 = fig.add_subplot(gs[2, 0])
        phase_coherences = wave_synchrony.get("phase_coherences", {})
        if phase_coherences:
            coherence_values = [v for v in phase_coherences.values() if not np.isnan(v)]
            if coherence_values:
                ax4.hist(
                    coherence_values,
                    bins=10,
                    alpha=0.7,
                    color="purple",
                    edgecolor="black",
                )
                ax4.axvline(
                    x=np.mean(coherence_values),
                    color="red",
                    linestyle="--",
                    linewidth=2,
                    label=f"Mean: {np.mean(coherence_values):.3f}",
                )
                ax4.axvline(
                    x=0.6,
                    color="green",
                    linestyle="--",
                    alpha=0.8,
                    label="Threshold (0.6)",
                )
                ax4.set_title("Phase Coherence Distribution")
                ax4.set_xlabel("Phase Coherence")
                ax4.set_ylabel("Frequency")
                ax4.legend()
                ax4.grid(True, alpha=0.3)
        else:
            ax4.text(
                0.5,
                0.5,
                "No phase coherence\ndata available",
                ha="center",
                va="center",
                transform=ax4.transAxes,
            )
            ax4.set_title("Phase Coherence Distribution")

        # Plot 5: Wave Quality by Region
        ax5 = fig.add_subplot(gs[2, 1])
        if wave_quality:
            regions = list(wave_quality.keys())
            qualities = [wave_quality[r] if not np.isnan(wave_quality[r]) else 0 for r in regions]
            colors_wq = ["green" if q > 0.7 else "orange" if q > 0.5 else "red" for q in qualities]

            bars = ax5.bar(regions, qualities, color=colors_wq, alpha=0.7, edgecolor="black")

            # Add value labels
            for bar, value in zip(bars, qualities):
                if value > 0:
                    ax5.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.02,
                        f"{value:.3f}",
                        ha="center",
                        va="bottom",
                        fontweight="bold",
                    )

            ax5.axhline(y=0.7, color="green", linestyle="--", alpha=0.8, label="Good")
            ax5.axhline(y=0.5, color="orange", linestyle="--", alpha=0.8, label="Moderate")
            ax5.set_title("Regional Wave Quality")
            ax5.set_ylabel("Quality Score")
            ax5.set_ylim(0, 1.1)
            ax5.legend(fontsize="small")
            ax5.grid(True, alpha=0.3, axis="y")
            plt.setp(ax5.get_xticklabels(), rotation=45, ha="right")
        else:
            ax5.text(
                0.5,
                0.5,
                "No wave quality\ndata available",
                ha="center",
                va="center",
                transform=ax5.transAxes,
            )
            ax5.set_title("Regional Wave Quality")

        # Plot 6: Assessment Summary (text box)
        ax6 = fig.add_subplot(gs[2, 2])
        ax6.axis("off")

        # Create assessment text
        assessment_text = []
        assessment_text.append(f"OVERALL STATUS: {overall_status.upper()}")
        assessment_text.append("")
        assessment_text.append("KEY METRICS:")
        assessment_text.append(f"• Synchrony: {synchrony_score:.3f}" if not np.isnan(synchrony_score) else "• Synchrony: N/A")
        assessment_text.append(f"• Phase Coherence: {phase_coherence:.3f}" if not np.isnan(phase_coherence) else "• Phase Coherence: N/A")
        assessment_text.append(f"• Wave Quality: {avg_wave_quality:.3f}" if not np.isnan(avg_wave_quality) else "• Wave Quality: N/A")
        assessment_text.append("")
        assessment_text.append("RECOMMENDATIONS:")
        recommendations = breathing_correctness.get("recommendations", [])
        for i, rec in enumerate(recommendations[:3], 1):  # Show first 3 recommendations
            assessment_text.append(f"{i}. {rec}")

        text_content = "\n".join(assessment_text)

        # Add colored background based on status
        bbox_props = dict(
            boxstyle="round,pad=0.5",
            facecolor=status_color.get(overall_status, "lightgray"),
            alpha=0.3,
            edgecolor="black",
        )
        ax6.text(
            0.05,
            0.95,
            text_content,
            transform=ax6.transAxes,
            fontsize=10,
            verticalalignment="top",
            bbox=bbox_props,
            fontfamily="monospace",
        )
        ax6.set_title("Assessment Summary")

        # Plot 7: Time-Lag Analysis (bottom row)
        ax7 = fig.add_subplot(gs[3, :])
        lag_correlations = wave_synchrony.get("lag_correlations", {})
        if lag_correlations:
            lag_pairs = []
            lag_values = []
            max_corrs = []

            for pair, lag_data in lag_correlations.items():
                if isinstance(lag_data, dict) and "lag_seconds" in lag_data:
                    lag_pairs.append(pair)
                    lag_values.append(lag_data["lag_seconds"])
                    max_corrs.append(lag_data.get("max_correlation", 0))

            if lag_pairs:
                # Create scatter plot of lag vs correlation
                scatter = ax7.scatter(
                    lag_values,
                    max_corrs,
                    c=max_corrs,
                    cmap="RdYlGn",
                    s=100,
                    alpha=0.7,
                    edgecolors="black",
                )

                # Add labels for each point
                for i, (lag, corr, pair) in enumerate(zip(lag_values, max_corrs, lag_pairs)):
                    ax7.annotate(
                        pair.replace("_", "-"),
                        (lag, corr),
                        xytext=(5, 5),
                        textcoords="offset points",
                        fontsize=8,
                        ha="left",
                    )

                ax7.axvline(x=0, color="black", linestyle="-", alpha=0.5, label="No Lag")
                ax7.axhline(
                    y=0.7,
                    color="green",
                    linestyle="--",
                    alpha=0.8,
                    label="Good Correlation",
                )

                plt.colorbar(scatter, ax=ax7, label="Max Correlation")
                ax7.set_xlabel("Time Lag (seconds)")
                ax7.set_ylabel("Max Cross-Correlation")
                ax7.set_title("Time Lag Analysis Between Regions")
                ax7.legend()
                ax7.grid(True, alpha=0.3)
        else:
            ax7.text(
                0.5,
                0.5,
                "No time-lag data available",
                ha="center",
                va="center",
                transform=ax7.transAxes,
                fontsize=12,
            )
            ax7.set_title("Time Lag Analysis Between Regions")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
            print(f"✅ Visualization saved to: {save_path}")

        return fig

    def create_interactive_breathing_plot(self, save_path=None):
        """
        Create an interactive 3D visualization of breathing patterns using plotly.

        Parameters:
        -----------
        save_path : str, optional
            Path to save the HTML file

        Returns:
        --------
        plotly.graph_objects.Figure or None
            Interactive plotly figure, or None if plotly not available
        """
        if not PLOTLY_AVAILABLE:
            print("Error: plotly is required for interactive visualization. Install with: pip install plotly")
            return None

        # Get analysis data
        time_series = self.compute_time_series_analysis()
        wave_synchrony = self.calculate_wave_synchrony(self.compute_regional_volumes())
        breathing_correctness = self.assess_breathing_correctness()

        time = time_series["time"]
        volumes = time_series["volumes"]

        # Create subplots
        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=(
                "Regional Volumes Over Time",
                "Volume Cross-Correlation Heatmap",
                "Breathing Phase Visualization",
                "Wave Quality Analysis",
            ),
            specs=[
                [{"secondary_y": False}, {"type": "heatmap"}],
                [{"secondary_y": True}, {"type": "bar"}],
            ],
            vertical_spacing=0.12,
            horizontal_spacing=0.1,
        )

        # Plot 1: Regional volumes
        colors = {
            "total": "black",
            "rib_cage": "red",
            "abdominal": "green",
            "abdomen": "blue",
        }
        for region, volume_data in volumes.items():
            fig.add_trace(
                go.Scatter(
                    x=time,
                    y=volume_data,
                    name=region.replace("_", " ").title(),
                    line=dict(color=colors.get(region, "gray"), width=2),
                    opacity=0.8 if region == "total" else 0.7,
                ),
                row=1,
                col=1,
            )

        # Add breathing phase markers
        insp_peaks = time_series["breathing_phases"]["inspiratory_peaks"]
        exp_peaks = time_series["breathing_phases"]["expiratory_peaks"]

        if len(insp_peaks) > 0:
            fig.add_trace(
                go.Scatter(
                    x=time[insp_peaks],
                    y=volumes["total"][insp_peaks],
                    mode="markers",
                    name="Inspiration",
                    marker=dict(color="red", size=8, symbol="triangle-up"),
                ),
                row=1,
                col=1,
            )

        if len(exp_peaks) > 0:
            fig.add_trace(
                go.Scatter(
                    x=time[exp_peaks],
                    y=volumes["total"][exp_peaks],
                    mode="markers",
                    name="Expiration",
                    marker=dict(color="blue", size=8, symbol="triangle-down"),
                ),
                row=1,
                col=1,
            )

        # Plot 2: Correlation heatmap
        regional_correlations = wave_synchrony.get("regional_correlations", {})
        if regional_correlations:
            # Create correlation matrix
            regions = ["rib_cage", "abdominal", "abdomen", "total"]
            corr_matrix = np.eye(len(regions))  # Start with identity matrix

            for i, region1 in enumerate(regions):
                for j, region2 in enumerate(regions):
                    if i != j:
                        pair_key = f"{region1}_{region2}"
                        reverse_key = f"{region2}_{region1}"
                        if pair_key in regional_correlations:
                            corr_matrix[i, j] = regional_correlations[pair_key]
                        elif reverse_key in regional_correlations:
                            corr_matrix[i, j] = regional_correlations[reverse_key]
                        else:
                            corr_matrix[i, j] = np.nan

            fig.add_trace(
                go.Heatmap(
                    z=corr_matrix,
                    x=regions,
                    y=regions,
                    colorscale="RdYlGn",
                    zmid=0,
                    zmin=-1,
                    zmax=1,
                    text=[[f"{val:.3f}" if not np.isnan(val) else "N/A" for val in row] for row in corr_matrix],
                    texttemplate="%{text}",
                    textfont={"size": 10},
                    showscale=True,
                    colorbar=dict(title="Correlation"),
                ),
                row=1,
                col=2,
            )

        # Plot 3: Breathing pattern with phase information
        fig.add_trace(
            go.Scatter(
                x=time,
                y=volumes["total"],
                name="Total Volume",
                line=dict(color="black", width=2),
            ),
            row=2,
            col=1,
        )

        # Add phase coherence as secondary y-axis
        phase_coherences = wave_synchrony.get("phase_coherences", {})
        if phase_coherences:
            coherence_values = list(phase_coherences.values())
            coherence_time = np.linspace(time[0], time[-1], len(coherence_values))
            fig.add_trace(
                go.Scatter(
                    x=coherence_time,
                    y=coherence_values,
                    name="Phase Coherence",
                    line=dict(color="purple", width=2, dash="dash"),
                    yaxis="y2",
                ),
                row=2,
                col=1,
                secondary_y=True,
            )

        # Plot 4: Wave quality by region
        wave_quality = wave_synchrony.get("wave_quality", {})
        if wave_quality:
            regions = list(wave_quality.keys())
            qualities = [wave_quality[r] if not np.isnan(wave_quality[r]) else 0 for r in regions]
            colors_wq = ["green" if q > 0.7 else "orange" if q > 0.5 else "red" for q in qualities]

            fig.add_trace(
                go.Bar(
                    x=regions,
                    y=qualities,
                    name="Wave Quality",
                    marker=dict(color=colors_wq, line=dict(color="black", width=1)),
                    text=[f"{q:.3f}" for q in qualities],
                    textposition="outside",
                ),
                row=2,
                col=2,
            )

            # Add threshold lines
            fig.add_hline(
                y=0.7,
                line_dash="dash",
                line_color="green",
                annotation_text="Good (≥0.7)",
                row=2,
                col=2,
            )
            fig.add_hline(
                y=0.5,
                line_dash="dash",
                line_color="orange",
                annotation_text="Moderate (≥0.5)",
                row=2,
                col=2,
            )

        # Update layout
        overall_status = breathing_correctness.get("overall_status", "unknown")
        title_color = {
            "correct": "green",
            "partially_correct": "orange",
            "incorrect": "red",
            "unknown": "gray",
        }

        fig.update_layout(
            title=dict(
                text=f"Interactive Breathing Correctness Analysis - Status: {overall_status.upper()}",
                font=dict(size=16, color=title_color.get(overall_status, "black")),
            ),
            height=800,
            showlegend=True,
            hovermode="closest",
        )

        # Update axes labels
        fig.update_xaxes(title_text="Time (seconds)", row=1, col=1)
        fig.update_yaxes(title_text="Volume (arbitrary units)", row=1, col=1)
        fig.update_xaxes(title_text="Time (seconds)", row=2, col=1)
        fig.update_yaxes(title_text="Volume (arbitrary units)", row=2, col=1)
        fig.update_yaxes(title_text="Phase Coherence", row=2, col=1, secondary_y=True)
        fig.update_xaxes(title_text="Region", row=2, col=2)
        fig.update_yaxes(title_text="Quality Score", row=2, col=2)

        if save_path:
            if not save_path.endswith(".html"):
                save_path += ".html"
            fig.write_html(save_path)
            print(f"✅ Interactive visualization saved to: {save_path}")

        return fig

    def _get_averaged_breath_cycle(self, data_signal: np.ndarray, time_signal: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], int]:
        """
        Segments a signal into breath cycles and computes the average cycle.
        This is a helper method for creating cycle-based visualizations.

        Returns a tuple of:
        - 1D array of cycle percentage (0-100)
        - 1D array of mean signal value at each percentage point
        - 1D array of standard deviation at each percentage point
        - number of cycles found
        """
        # Use troughs (expiratory peaks) as cycle delimiters for consistency
        _, exp_peaks = self.detect_breathing_cycles(data_signal)

        if len(exp_peaks) < 2:
            return None, None, None, 0

        cycles = []
        cycle_points = 101  # for 0 to 100%
        cycle_perc = np.linspace(0, 100, cycle_points)

        for i in range(len(exp_peaks) - 1):
            start_idx, end_idx = exp_peaks[i], exp_peaks[i + 1]
            if start_idx >= end_idx:
                continue

            cycle_time = time_signal[start_idx : end_idx + 1]
            cycle_data = data_signal[start_idx : end_idx + 1]

            duration = cycle_time[-1] - cycle_time[0]
            if duration <= 0:
                continue

            normalized_time = (cycle_time - cycle_time[0]) / duration * 100

            # Interpolate to the standard cycle percentage scale
            f = interp1d(
                normalized_time,
                cycle_data,
                bounds_error=False,
                fill_value="extrapolate",
            )
            resampled_cycle = f(cycle_perc)
            cycles.append(resampled_cycle)

        if not cycles:
            return None, None, None, 0

        cycles_arr = np.array(cycles)

        mean_cycle = np.nanmean(cycles_arr, axis=0)
        std_cycle = np.nanstd(cycles_arr, axis=0)

        return cycle_perc, mean_cycle, std_cycle, len(cycles)

    def create_report_style_interactive_plot(self, save_path: Optional[str] = None):
        """
        Generates an interactive Plotly dashboard replicating the graphs from the PDF report.

        Parameters:
        -----------
        save_path : str, optional
            Path to save the interactive HTML file. If None, the plot is not saved.

        Returns:
        --------
        plotly.graph_objects.Figure
            The interactive plotly figure object.
        """
        if not PLOTLY_AVAILABLE:
            print("Error: plotly is required for this visualization. Install with: pip install plotly")
            return None

        print("Generating interactive Plotly report...")

        # --- 1. Data Preparation ---
        try:
            time_series_data = self.compute_time_series_analysis()
            metrics = self.compute_comprehensive_analysis()
            regional_volumes = time_series_data["volumes"]
            time = time_series_data["time"]
        except Exception as e:
            print(f"Error preparing data for report: {e}")
            return None

        colors = {
            "Pulmonary Rib Cage": "#00BFFF",
            "Abdominal Rib Cage": "#9ACD32",
            "Abdomen": "#FFA500",
            "Chest Wall": "#9370DB",
        }
        name_map = {
            "rib_cage": "Pulmonary Rib Cage",
            "abdominal": "Abdominal Rib Cage",
            "abdomen": "Abdomen",
            "total": "Chest Wall",
        }

        # --- 2. Create Subplot Layout ---
        try:
            fig = make_subplots(
                rows=3,
                cols=2,
                subplot_titles=(
                    "<b>Volume Over Time</b> (All Compartments)",
                    "<b>Averaged Breath Cycles</b> (by Compartment)",
                    "<b>Averaged Chest Wall Cycle</b>",
                    "<b>Averaged Tidal Flow Cycle</b>",
                    "<b>Average Flow-Volume Loop</b>",
                    "<b>Compartment Volume Contribution (%)</b>",
                ),
                specs=[
                    [{}, {}],
                    [{}, {}],
                    [
                        {"type": "xy"},
                        {"type": "domain"},
                    ],  # Last cell is for a pie chart
                ],
                vertical_spacing=0.15,
                horizontal_spacing=0.08,
            )
        except Exception as e:
            print(f"Error creating subplot layout: {e}")
            return None

        # --- 3. Populate Plots ---
        n_cycles = 0  # Initialize cycle count

        try:
            # Plot 1: Time Series (Page 1)
            for key, data in regional_volumes.items():
                if key in name_map and not np.all(np.isnan(data)):
                    fig.add_trace(
                        go.Scatter(
                            x=time,
                            y=data,
                            name=name_map[key],
                            line=dict(color=colors[name_map[key]]),
                            legendgroup="ts",
                        ),
                        row=1,
                        col=1,
                    )

            # Plot 2: Averaged Compartment Cycles (Page 3, Top)
            for key in ["rib_cage", "abdominal", "abdomen"]:
                vol_data = regional_volumes.get(key)
                name = name_map.get(key)
                if vol_data is not None and name and not np.all(np.isnan(vol_data)):
                    try:
                        perc, mean, std, cycle_count = self._get_averaged_breath_cycle(vol_data, time)
                        if perc is not None:
                            fig.add_trace(
                                go.Scatter(
                                    x=perc,
                                    y=mean,
                                    name=name,
                                    mode="lines",
                                    line=dict(color=colors[name]),
                                    legendgroup="avg",
                                ),
                                row=1,
                                col=2,
                            )
                            fig.add_trace(
                                go.Scatter(
                                    x=np.concatenate([perc, perc[::-1]]),
                                    y=np.concatenate([mean + std, (mean - std)[::-1]]),
                                    fill="toself",
                                    fillcolor=colors[name],
                                    opacity=0.2,
                                    line=dict(color="rgba(255,255,255,0)"),
                                    showlegend=False,
                                    legendgroup="avg",
                                ),
                                row=1,
                                col=2,
                            )
                    except Exception as e:
                        print(f"Warning: Could not create averaged cycle for {key}: {e}")

            # Plot 3 & 4: Averaged Total Volume and Flow (Page 2)
            total_vol = regional_volumes.get("total")
            if total_vol is not None and not np.all(np.isnan(total_vol)):
                try:
                    # Averaged Volume Cycle
                    (
                        vol_perc,
                        vol_mean,
                        vol_std,
                        n_cycles,
                    ) = self._get_averaged_breath_cycle(total_vol, time)
                    if vol_perc is not None and vol_mean is not None:
                        fig.add_trace(
                            go.Scatter(
                                x=vol_perc,
                                y=vol_mean,
                                name="Chest Wall Volume",
                                mode="lines",
                                line=dict(color=colors["Chest Wall"]),
                                showlegend=False,
                            ),
                            row=2,
                            col=1,
                        )
                        fig.add_trace(
                            go.Scatter(
                                x=np.concatenate([vol_perc, vol_perc[::-1]]),
                                y=np.concatenate([vol_mean + vol_std, (vol_mean - vol_std)[::-1]]),
                                fill="toself",
                                fillcolor=colors["Chest Wall"],
                                opacity=0.2,
                                line=dict(color="rgba(255,255,255,0)"),
                                showlegend=False,
                            ),
                            row=2,
                            col=1,
                        )

                        # Averaged Flow Cycle
                        flow = np.gradient(total_vol, time)
                        (
                            flow_perc,
                            flow_mean,
                            flow_std,
                            _,
                        ) = self._get_averaged_breath_cycle(flow, time)
                        if flow_perc is not None and flow_mean is not None:
                            fig.add_trace(
                                go.Scatter(
                                    x=flow_perc,
                                    y=flow_mean,
                                    name="Tidal Flow",
                                    mode="lines",
                                    line=dict(color=colors["Chest Wall"]),
                                    showlegend=False,
                                ),
                                row=2,
                                col=2,
                            )
                            fig.add_trace(
                                go.Scatter(
                                    x=np.concatenate([flow_perc, flow_perc[::-1]]),
                                    y=np.concatenate(
                                        [
                                            flow_mean + flow_std,
                                            (flow_mean - flow_std)[::-1],
                                        ]
                                    ),
                                    fill="toself",
                                    fillcolor=colors["Chest Wall"],
                                    opacity=0.2,
                                    line=dict(color="rgba(255,255,255,0)"),
                                    showlegend=False,
                                ),
                                row=2,
                                col=2,
                            )
                            fig.add_hline(
                                y=0,
                                line_width=1,
                                line_dash="dash",
                                line_color="black",
                                row=2,
                                col=2,
                            )

                            # Plot 5: Flow-Volume Loop (Page 2)
                            # Normalize volume to start from 0 for a typical TV loop
                            tidal_vol_cycle = vol_mean - np.min(vol_mean)
                            fig.add_trace(
                                go.Scatter(
                                    x=tidal_vol_cycle,
                                    y=flow_mean,
                                    name="Flow-Volume Loop",
                                    line=dict(color=colors["Chest Wall"]),
                                    showlegend=False,
                                    fill="toself",
                                    fillcolor=colors["Chest Wall"],
                                    opacity=0.2,
                                ),
                                row=3,
                                col=1,
                            )
                            fig.add_hline(
                                y=0,
                                line_width=1,
                                line_dash="dash",
                                line_color="black",
                                row=3,
                                col=1,
                            )
                            fig.update_yaxes(autorange="reversed", row=3, col=1)  # Inspiration is positive flow
                except Exception as e:
                    print(f"Warning: Could not create volume/flow cycles: {e}")

            # Plot 6: Regional Contributions Pie Chart (Page 3, Table)
            try:
                if hasattr(metrics, "regional_contribution") and metrics.regional_contribution:
                    contribs = metrics.regional_contribution
                    labels = [name_map.get(k, k) for k in contribs.keys() if k in name_map]
                    values = [v * 100 for k, v in contribs.items() if k in name_map and not np.isnan(v)]
                    pie_colors = [colors.get(name_map.get(k, k), "#808080") for k in contribs.keys() if k in name_map]

                    if labels and values:
                        fig.add_trace(
                            go.Pie(
                                labels=labels,
                                values=values,
                                marker_colors=pie_colors,
                                hole=0.3,
                                textinfo="percent+label",
                            ),
                            row=3,
                            col=2,
                        )
            except Exception as e:
                print(f"Warning: Could not create regional contributions pie chart: {e}")

        except Exception as e:
            print(f"Error populating plots: {e}")
            return None

        # --- 4. Final Layout Updates ---
        try:
            fig.update_layout(
                title_text=f"<b>Interactive Breathing Analysis Report</b> (Detected {n_cycles} breaths)",
                height=1000,
                showlegend=True,
                template="plotly_white",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            )

            # Update all axes
            fig.update_xaxes(title_text="Time [s]", row=1, col=1)
            fig.update_yaxes(title_text="Volume [L]", row=1, col=1)
            fig.update_xaxes(title_text="Cycle [%]", row=1, col=2)
            fig.update_yaxes(title_text="Volume [L]", row=1, col=2)
            fig.update_xaxes(title_text="Cycle [%]", row=2, col=1)
            fig.update_yaxes(title_text="Volume [L]", row=2, col=1)
            fig.update_xaxes(title_text="Cycle [%]", row=2, col=2)
            fig.update_yaxes(title_text="Flow [L/s]", row=2, col=2)
            fig.update_xaxes(title_text="Tidal Volume [L]", row=3, col=1)
            fig.update_yaxes(title_text="Flow [L/s]", row=3, col=1)
        except Exception as e:
            print(f"Error updating layout: {e}")
            return None

        # Save the plot
        if save_path:
            try:
                if not save_path.endswith(".html"):
                    save_path += ".html"
                fig.write_html(save_path)
                print(f"✅ Interactive report saved to: {save_path}")
            except Exception as e:
                print(f"Error saving plot: {e}")

        return fig
