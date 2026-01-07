from dataclasses import dataclass
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
from scipy import signal
from scipy.spatial import ConvexHull
from scipy.stats import pearsonr


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


class GraphPreprocessor:
    def __init__(self):

        rib_cage_points_set_front = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
        rib_cage_points_set_back = [43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 54, 53]
        rib_cage_points_set = rib_cage_points_set_front + rib_cage_points_set_back

        abdominal_points_set_front = [15, 16, 17, 18, 19, 20, 21, 22, 23, 24]
        abdominal_points_set_back = [55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75]
        abdominal_points_set = abdominal_points_set_front + abdominal_points_set_back

        abdomen_points_set_front = [25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42]
        abdomen_points_set_back = [76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89]
        abdomen_points_set = abdomen_points_set_front + abdomen_points_set_back

        self.points_set = {}
        self.points_set["rib_cage"] = rib_cage_points_set
        self.points_set["abdominal"] = abdominal_points_set
        self.points_set["abdomen"] = abdomen_points_set


class GraphCruncher:
    # Unit conversion constants
    MM3_TO_LITERS = 1_000_000  # 1 liter = 1,000,000 mm³

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
            print(f"Volume computation error: {e}")
            return np.nan

    def compute_regional_volumes(self, units: str = "mm3") -> Dict[str, np.ndarray]:
        """Compute volumes for different anatomical regions over time.

        Parameters:
        -----------
        units : str
            Volume units - 'mm3' for cubic millimeters or 'liters' for liters

        Returns:
        --------
        Dict[str, np.ndarray]
            Dictionary with regional volumes in specified units
        """
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

        # Convert to liters if requested
        if units.lower() == "liters":
            for region in regional_volumes:
                regional_volumes[region] = regional_volumes[region] / self.MM3_TO_LITERS

        return regional_volumes

    def get_regional_volumes_in_liters(self) -> Dict[str, np.ndarray]:
        """
        Convenience method to get regional volumes in liters.

        Returns:
        --------
        Dict[str, np.ndarray]
            Dictionary with regional volumes in liters
        """
        return self.compute_regional_volumes(units="liters")

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
            print(f"Surface area computation error: {e}")
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

    def detect_breathing_cycles(
        self,
        volume_signal: np.ndarray,
        min_breath_duration: float = 1.0,
        smooth_seconds: float = 0.2,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detect inspiratory and expiratory peaks from a lung volume signal.

        Uses light smoothing and adaptive, statistics-based thresholds
        with scipy.signal.find_peaks, similar to ECG peak detection
        but tuned for respiratory volume signals.

        Parameters
        ----------
        volume_signal : np.ndarray
            Lung volume signal over time.
        min_breath_duration : float, default=1.0
            Minimum expected breath duration in seconds.
        smooth_seconds : float, default=0.2
            Smoothing window length in seconds.

        Returns
        -------
        inspiratory_peaks : np.ndarray
            Indices of inspiratory (end-inhalation) peaks.
        expiratory_peaks : np.ndarray
            Indices of expiratory (end-exhalation) peaks.
        """

        fs = self.sampling_frequency
        vol = np.asarray(volume_signal, float)

        valid = ~np.isnan(vol)
        if valid.sum() < fs:
            return np.array([]), np.array([])

        idx_map = np.where(valid)[0]
        vol = vol[valid]

        win = int(smooth_seconds * fs)
        win = max(5, win | 1)  # odd window
        vol_s = signal.savgol_filter(vol, win, polyorder=2)

        v_median = np.median(vol_s)
        v_range = np.percentile(vol_s, 95) - np.percentile(vol_s, 5)

        prominence = 0.25 * v_range
        min_distance = int(min_breath_duration * fs * 0.6)

        insp_height = v_median + 0.2 * v_range
        exp_height = v_median - 0.2 * v_range

        insp_peaks, _ = signal.find_peaks(
            vol_s,
            distance=min_distance,
            prominence=prominence,
            height=insp_height,
        )

        exp_peaks, _ = signal.find_peaks(
            -vol_s,
            distance=min_distance,
            prominence=prominence,
            height=-exp_height,
        )

        insp_final = []
        exp_final = []

        i, j = 0, 0
        while i < len(insp_peaks) and j < len(exp_peaks):
            if insp_peaks[i] < exp_peaks[j]:
                insp_final.append(insp_peaks[i])
                exp_final.append(exp_peaks[j])
                i += 1
                j += 1
            else:
                j += 1

        insp_final = np.asarray(insp_final)
        exp_final = np.asarray(exp_final)

        return idx_map[insp_final], idx_map[exp_final]

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
            print(f"Volume synchrony computation error: {e}")
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

    def compute_comprehensive_analysis(self, volume_units: str = "liters") -> RespiratoryMetrics:
        """
        Perform comprehensive respiratory analysis on the 3D point data.

        Parameters:
        -----------
        volume_units : str
            Units for volume measurements - 'mm3' or 'liters' (default: 'liters')

        Returns:
        --------
        RespiratoryMetrics
            Comprehensive analysis results
        """
        # Compute regional volumes in specified units
        regional_volumes = self.compute_regional_volumes(units=volume_units)

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
        )

    def compute_time_series_analysis(self, volume_units: str = "liters") -> Dict[str, Any]:
        """
        Compute time series data for frontend visualization.

        Parameters:
        -----------
        volume_units : str
            Units for volume measurements - 'mm3' or 'liters' (default: 'liters')

        Returns:
        --------
        Dict[str, Any]
            Time series data including volumes, displacements, and breathing phases
        """
        # Get time array
        time_array = self._obj["Time"].values

        # Compute regional volumes over time in specified units
        regional_volumes = self.compute_regional_volumes(units=volume_units)

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
            "displacements": {"chest": chest_displacement, "abdominal": abdominal_displacement, "total": total_displacement},
            "surface_areas": np.array(surface_areas),
            "breathing_phases": {"phase_signal": breathing_phase, "inspiratory_peaks": insp_peaks, "expiratory_peaks": exp_peaks},
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
                "breathing_rate_bpm": round(metrics.breathing_rate, 1) if not np.isnan(metrics.breathing_rate) else None,
                "breathing_rate_interpretation": interpret_breathing_rate(metrics.breathing_rate),
                "tidal_volume_change": round(metrics.tidal_volume_change, 2) if not np.isnan(metrics.tidal_volume_change) else None,
                "inspiratory_time_sec": round(metrics.inspiratory_time, 2) if not np.isnan(metrics.inspiratory_time) else None,
                "expiratory_time_sec": round(metrics.expiratory_time, 2) if not np.isnan(metrics.expiratory_time) else None,
                "ie_ratio": round(metrics.ie_ratio, 2) if not np.isnan(metrics.ie_ratio) else None,
                "ie_ratio_interpretation": interpret_ie_ratio(metrics.ie_ratio),
                "pattern_regularity": round(metrics.breathing_pattern_regularity, 3) if not np.isnan(metrics.breathing_pattern_regularity) else None,
                "regularity_interpretation": interpret_regularity(metrics.breathing_pattern_regularity),
                "volume_synchrony": round(metrics.volume_synchrony, 2) if not np.isnan(metrics.volume_synchrony) else None,
                "breathing_efficiency": round(metrics.breathing_efficiency, 2) if not np.isnan(metrics.breathing_efficiency) else None,
            },
            "regional_analysis": {
                "chest_wall_displacement": round(metrics.chest_wall_displacement, 2) if not np.isnan(metrics.chest_wall_displacement) else None,
                "abdominal_displacement": round(metrics.abdominal_displacement, 2) if not np.isnan(metrics.abdominal_displacement) else None,
                "regional_contributions": {region: round(contrib, 3) if not np.isnan(contrib) else None for region, contrib in metrics.regional_contribution.items()},
            },
            "time_series_data": time_series,
            "data_quality": {
                "total_duration_sec": float(time_series["time"][-1] - time_series["time"][0]) if len(time_series["time"]) > 0 else 0,
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
            Complete analysis results formatted for frontend consumption
        """
        analysis_summary = self.get_analysis_summary()

        # Add visualization-ready data
        frontend_data = analysis_summary.copy()

        # Format time series data for plotting
        time_series = analysis_summary["time_series_data"]

        frontend_data["plots"] = {
            "volume_over_time": {
                "x": time_series["time"].tolist(),
                "y_total": time_series["volumes"]["total"].tolist(),
                "y_ribcage": time_series["volumes"]["rib_cage"].tolist(),
                "y_abdominal": time_series["volumes"]["abdominal"].tolist(),
                "y_abdomen": time_series["volumes"]["abdomen"].tolist(),
                "breathing_phases": {
                    "inspiratory_peaks": {"x": [time_series["time"][i] for i in time_series["breathing_phases"]["inspiratory_peaks"]], "y": [time_series["volumes"]["total"][i] for i in time_series["breathing_phases"]["inspiratory_peaks"]]},
                    "expiratory_peaks": {"x": [time_series["time"][i] for i in time_series["breathing_phases"]["expiratory_peaks"]], "y": [time_series["volumes"]["total"][i] for i in time_series["breathing_phases"]["expiratory_peaks"]]},
                },
            },
            "displacement_over_time": {"x": time_series["time"].tolist(), "y_chest": time_series["displacements"]["chest"].tolist(), "y_abdominal": time_series["displacements"]["abdominal"].tolist(), "y_total": time_series["displacements"]["total"].tolist()},
            "surface_area_over_time": {"x": time_series["time"].tolist(), "y": time_series["surface_areas"].tolist()},
        }

        return frontend_data
