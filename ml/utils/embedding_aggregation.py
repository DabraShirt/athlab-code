import warnings

import numpy as np
from scipy import stats
from scipy.signal import welch
from sklearn.preprocessing import RobustScaler

from ml.utils.logging_utils import get_logger

warnings.filterwarnings("ignore")

logger = get_logger("athlab.embedding_aggregator")


class EmbeddingAggregator:
    """
    Embedding aggregator optimized for breathing pattern analysis.

    Original embeddings: (timesteps, 1, 26) where timesteps = 2392-3618
    - Each timestep = one moment in runner's breathing session
    - 26 dimensions = breathing pattern features per moment
    - Goal: Extract runner's temporal breathing signature
    """

    def __init__(self, target_dims: int = 512, verbose: bool = True):
        self.target_dims = target_dims
        self.verbose = verbose

        # Optimized allocation for exactly 512 dense features
        self.allocation = {
            "early_mid_late": 96,  # Session progression patterns (early/mid/late phases)
            "breathing_rhythm": 96,  # Rhythm, regularity, and periodicity
            "pattern_evolution": 80,  # Temporal changes and adaptation
            "spectral_patterns": 64,  # Frequency domain analysis
            "breathing_cycles": 64,  # Breathing cycle detection and analysis
            "stability_metrics": 48,  # Pattern stability and consistency
            "fatigue_indicators": 32,  # Fatigue and performance degradation
            "individual_signature": 16,  # Runner-specific breathing characteristics
            "global_summary": 16,  # Overall session characteristics
        }

    def _log(self, message: str):
        if self.verbose:
            logger.info(f"   {message}")

    def _validate_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        """Validate and prepare embeddings.

        Parameters
        ----------
        embeddings : np.ndarray
            Input embeddings array to validate.

        Returns
        -------
        np.ndarray
            Validated and cleaned embeddings array.
        """
        if not isinstance(embeddings, np.ndarray):
            embeddings = np.array(embeddings)

        # Handle 3D case: (timesteps, 1, 26) -> (timesteps, 26)
        if embeddings.ndim == 3 and embeddings.shape[1] == 1:
            embeddings = embeddings.squeeze(1)

        if embeddings.ndim != 2:
            raise ValueError(f"Expected 2D embeddings after processing, got {embeddings.shape}")

        # Clean problematic values
        embeddings = np.nan_to_num(embeddings, nan=0.0, posinf=1e6, neginf=-1e6)

        return embeddings

    def extract_early_mid_late_patterns(self, embeddings: np.ndarray) -> np.ndarray:
        """Extract breathing patterns from early, middle, and late session phases.

        Captures breathing adaptation, fatigue, and pattern changes across
        session progression.

        Parameters
        ----------
        embeddings : np.ndarray
            Input embeddings with shape (timesteps, dimensions).

        Returns
        -------
        np.ndarray
            Features extracted from early/mid/late phases (96 dimensions).
        """
        n_seq, embed_dim = embeddings.shape
        target_dims = self.allocation["early_mid_late"]

        # Select most informative dimensions
        dim_vars = np.var(embeddings, axis=0)
        top_dims = np.argsort(dim_vars)[-8:]  # Top 8 most variable dimensions
        selected = embeddings[:, top_dims]

        # Divide into early/mid/late phases
        phase_size = n_seq // 3
        phases = {
            "early": selected[:phase_size],
            "mid": selected[phase_size : 2 * phase_size],
            "late": selected[2 * phase_size :],
        }

        features = []
        features_per_phase = target_dims // 3

        for phase_name, phase_data in phases.items():
            if phase_data.shape[0] == 0:
                features.extend([0.0] * features_per_phase)
                continue

            phase_features = []

            # Statistical characteristics of this phase
            phase_features.extend(
                [
                    np.mean(phase_data),  # Average activity level
                    np.std(phase_data),  # Pattern variability
                    np.median(np.abs(phase_data)),  # Robust activity level
                    np.percentile(np.abs(phase_data), 90),  # Peak activity
                    np.percentile(np.abs(phase_data), 10),  # Minimum activity
                    stats.iqr(phase_data.flatten()),  # Interquartile range
                ]
            )

            # Per-dimension characteristics
            for dim_idx in range(min(4, phase_data.shape[1])):
                dim_data = phase_data[:, dim_idx]
                phase_features.extend(
                    [
                        np.mean(dim_data),  # Mean pattern
                        np.std(dim_data),  # Pattern spread
                        np.max(dim_data) - np.min(dim_data),  # Range
                        np.mean(np.abs(np.diff(dim_data))),  # Rate of change
                    ]
                )

            # Temporal dynamics within phase
            if phase_data.shape[0] > 1:
                phase_diff = np.diff(phase_data, axis=0)
                phase_features.extend(
                    [
                        np.mean(np.linalg.norm(phase_diff, axis=1)),  # Average change magnitude
                        np.std(np.linalg.norm(phase_diff, axis=1)),  # Change variability
                        np.max(np.linalg.norm(phase_diff, axis=1)),  # Max change rate
                    ]
                )
            else:
                phase_features.extend([0.0, 0.0, 0.0])

            # Truncate or pad
            phase_features = phase_features[:features_per_phase]
            if len(phase_features) < features_per_phase:
                # Pad with derived features instead of zeros
                remaining = features_per_phase - len(phase_features)
                padding = []
                for i in range(remaining):
                    if i < len(phase_features):
                        padding.append(phase_features[i % len(phase_features)] * 0.1)  # Scaled copies
                    else:
                        padding.append(np.mean(phase_data) if phase_data.size > 0 else 0.0)
                phase_features.extend(padding)

            features.extend(phase_features)

        return np.array(features[:target_dims], dtype=np.float32)

    def extract_breathing_rhythm(self, embeddings: np.ndarray) -> np.ndarray:
        """Extract breathing rhythm and regularity features.

        Analyzes breathing periodicity, consistency, depth, and temporal dynamics.

        Parameters
        ----------
        embeddings : np.ndarray
            Input embeddings with shape (timesteps, dimensions).

        Returns
        -------
        np.ndarray
            Breathing rhythm features (96 dimensions).
        """
        n_seq, embed_dim = embeddings.shape
        target_dims = self.allocation["breathing_rhythm"]

        # Select most dynamic dimensions
        dim_vars = np.var(embeddings, axis=0)
        top_dims = np.argsort(dim_vars)[-6:]  # Top 6 dimensions
        selected = embeddings[:, top_dims]

        features = []

        # RHYTHM REGULARITY ANALYSIS
        if n_seq > 2:
            # First and second derivatives (velocity and acceleration)
            first_diff = np.diff(selected, axis=0)

            # Regularity measures
            velocity_magnitudes = np.linalg.norm(first_diff, axis=1)
            features.extend(
                [
                    np.mean(velocity_magnitudes),  # Average breathing rate
                    np.std(velocity_magnitudes),  # Rate variability (regularity)
                    np.median(velocity_magnitudes),  # Typical rate
                    stats.iqr(velocity_magnitudes),  # Rate spread
                    np.max(velocity_magnitudes),  # Peak rate
                    np.min(velocity_magnitudes),  # Minimum rate
                ]
            )

            # Pattern consistency
            if first_diff.shape[0] > 1:
                consistency_scores = []
                for dim_idx in range(first_diff.shape[1]):
                    dim_changes = first_diff[:, dim_idx]
                    if np.std(dim_changes) > 1e-8:
                        # Consistency = inverse of normalized variability
                        consistency = 1.0 / (1.0 + np.std(dim_changes) / (np.abs(np.mean(dim_changes)) + 1e-8))
                        consistency_scores.append(consistency)

                if consistency_scores:
                    features.extend(
                        [
                            np.mean(consistency_scores),  # Average consistency
                            np.std(consistency_scores),  # Consistency variability
                            np.max(consistency_scores),  # Best consistency
                            np.min(consistency_scores),  # Worst consistency
                        ]
                    )
                else:
                    features.extend([0.5, 0.1, 0.5, 0.5])  # Default values
            else:
                features.extend([0.5, 0.1, 0.5, 0.5])
        else:
            features.extend([0.0] * 10)

        # PERIODICITY DETECTION
        for dim_idx in range(min(3, selected.shape[1])):
            dim_data = selected[:, dim_idx]

            # Test multiple potential breathing periods
            period_correlations = []
            test_periods = [80, 100, 120, 150, 180, 200, 250]  # Various breathing rates

            for period in test_periods:
                if n_seq > period + 10:
                    try:
                        # Autocorrelation at this period
                        sig1 = dim_data[:-period]
                        sig2 = dim_data[period:]

                        if np.std(sig1) > 1e-8 and np.std(sig2) > 1e-8:
                            corr = np.corrcoef(sig1, sig2)[0, 1]
                            period_correlations.append(corr if not np.isnan(corr) else 0.0)
                        else:
                            period_correlations.append(0.0)
                    except Exception as e:
                        logger.debug(f"Breathing rhythm periodicity failed: {e}")
                        period_correlations.append(0.0)
                else:
                    period_correlations.append(0.0)

            # Extract periodicity features
            if period_correlations:
                features.extend(
                    [
                        np.max(period_correlations),  # Strongest periodicity
                        np.mean(period_correlations),  # Average periodicity
                        np.std(period_correlations),  # Periodicity variability
                        np.argmax(period_correlations) * 20 + 80,  # Dominant period estimate
                    ]
                )
            else:
                features.extend([0.0, 0.0, 0.0, 150.0])  # Default breathing period ~150

        # BREATHING DEPTH AND AMPLITUDE
        for dim_idx in range(min(2, selected.shape[1])):
            dim_data = selected[:, dim_idx]

            # Amplitude characteristics
            features.extend(
                [
                    np.std(dim_data),  # Breathing depth variability
                    np.max(dim_data) - np.min(dim_data),  # Total amplitude range
                    np.percentile(dim_data, 90) - np.percentile(dim_data, 10),  # Central range
                    np.mean(np.abs(dim_data - np.median(dim_data))),  # Deviation from median
                ]
            )

        # Ensure we have enough features
        while len(features) < target_dims:
            # Generate additional features based on existing ones
            if len(features) > 0:
                features.append(np.mean(features[-min(5, len(features)) :]))  # Recent average
            else:
                features.append(0.0)

        return np.array(features[:target_dims], dtype=np.float32)

    def extract_pattern_evolution(self, embeddings: np.ndarray) -> np.ndarray:
        """Extract how breathing patterns evolve over the session.

        Analyzes trends, adaptation patterns, and temporal evolution of breathing.

        Parameters
        ----------
        embeddings : np.ndarray
            Input embeddings with shape (timesteps, dimensions).

        Returns
        -------
        np.ndarray
            Pattern evolution features (80 dimensions).
        """
        n_seq, embed_dim = embeddings.shape
        target_dims = self.allocation["pattern_evolution"]

        # Select key dimensions
        dim_vars = np.var(embeddings, axis=0)
        top_dims = np.argsort(dim_vars)[-6:]  # Top 6 dimensions
        selected = embeddings[:, top_dims]

        features = []

        # TREND ANALYSIS - overall drift in patterns
        time_indices = np.arange(n_seq)

        for dim_idx in range(selected.shape[1]):
            dim_data = selected[:, dim_idx]

            # Linear trend over entire session
            try:
                slope, intercept, r_value, p_value, std_err = stats.linregress(time_indices, dim_data)
                features.extend(
                    [
                        slope,  # Rate of change
                        np.abs(slope),  # Magnitude of trend
                        r_value**2,  # Trend strength
                        std_err / (np.abs(slope) + 1e-8),  # Relative uncertainty
                    ]
                )
            except Exception as e:
                features.extend([0.0, 0.0, 0.0, 1.0])
                logger.debug(f"Pattern evolution trend analysis failed for dim {dim_idx}: {e}")

        # SLIDING WINDOW EVOLUTION
        window_size = max(100, n_seq // 8)  # Adaptive window size
        step_size = window_size // 2
        n_windows = (n_seq - window_size) // step_size + 1

        if n_windows >= 2:
            window_stats = []

            for i in range(n_windows):
                start_idx = i * step_size
                end_idx = min(start_idx + window_size, n_seq)
                window_data = selected[start_idx:end_idx, :]

                if window_data.shape[0] > 0:
                    window_stats.append(
                        {
                            "mean": np.mean(window_data, axis=0),
                            "std": np.std(window_data, axis=0),
                            "activity": np.mean(np.abs(window_data)),
                        }
                    )

            if len(window_stats) >= 2:
                # Evolution of window statistics
                means = np.array([ws["mean"] for ws in window_stats])
                stds = np.array([ws["std"] for ws in window_stats])
                activities = np.array([ws["activity"] for ws in window_stats])

                features.extend(
                    [
                        np.std(means, axis=0).mean(),  # Mean pattern drift
                        np.std(stds, axis=0).mean(),  # Variability pattern drift
                        np.std(activities),  # Activity level drift
                        np.mean(np.diff(activities)),  # Activity trend
                        np.max(activities) - np.min(activities),  # Activity range
                    ]
                )

                # Pattern stability over time
                for dim_idx in range(min(3, means.shape[1])):
                    dim_evolution = means[:, dim_idx]
                    features.extend(
                        [
                            np.std(dim_evolution),  # Pattern stability
                            np.max(dim_evolution) - np.min(dim_evolution),  # Evolution range
                        ]
                    )

        # ADAPTATION PATTERNS - early vs late session comparison
        if n_seq > 20:
            early_portion = selected[: n_seq // 4, :]  # First 25%
            late_portion = selected[3 * n_seq // 4 :, :]  # Last 25%

            if early_portion.shape[0] > 0 and late_portion.shape[0] > 0:
                # Compare early vs late patterns
                early_mean = np.mean(early_portion, axis=0)
                late_mean = np.mean(late_portion, axis=0)
                early_std = np.std(early_portion, axis=0)
                late_std = np.std(late_portion, axis=0)

                features.extend(
                    [
                        np.mean(np.abs(late_mean - early_mean)),  # Pattern shift magnitude
                        np.mean(np.abs(late_std - early_std)),  # Variability change
                        (np.corrcoef(early_mean, late_mean)[0, 1] if len(early_mean) > 1 else 0.5),  # Pattern correlation
                        np.mean(late_std) / (np.mean(early_std) + 1e-8),  # Relative variability change
                    ]
                )

        # Ensure target dimensions
        while len(features) < target_dims:
            if len(features) > 0:
                features.append(features[-1] * 0.1)  # Scaled copies
            else:
                features.append(0.0)

        return np.array(features[:target_dims], dtype=np.float32)

    def extract_spectral_patterns(self, embeddings: np.ndarray) -> np.ndarray:
        """Extract frequency domain patterns.

        Analyzes power spectral density and frequency characteristics.

        Parameters
        ----------
        embeddings : np.ndarray
            Input embeddings with shape (timesteps, dimensions).

        Returns
        -------
        np.ndarray
            Spectral features (64 dimensions).
        """
        n_seq, embed_dim = embeddings.shape
        target_dims = self.allocation["spectral_patterns"]

        if n_seq < 32:
            return np.zeros(target_dims, dtype=np.float32)

        # Select most variable dimensions
        dim_vars = np.var(embeddings, axis=0)
        top_dims = np.argsort(dim_vars)[-4:]  # Top 4 dimensions
        selected = embeddings[:, top_dims]

        features = []
        features_per_dim = target_dims // len(top_dims)

        for dim_idx in range(selected.shape[1]):
            signal = selected[:, dim_idx]

            # Normalize signal
            signal = signal - np.mean(signal)
            if np.std(signal) > 1e-8:
                signal = signal / np.std(signal)

            try:
                # Power spectral density
                freqs, psd = welch(signal, nperseg=min(64, len(signal) // 4))

                if len(psd) > 0:
                    dim_features = [
                        np.sum(psd),  # Total power
                        np.max(psd),  # Peak power
                        np.argmax(psd),  # Dominant frequency
                        np.mean(psd),  # Average power
                        np.std(psd),  # Power variability
                        np.sum(psd[: len(psd) // 3]),  # Low frequency power
                        np.sum(psd[len(psd) // 3 : 2 * len(psd) // 3]),  # Mid frequency power
                        np.sum(psd[2 * len(psd) // 3 :]),  # High frequency power
                    ]
                else:
                    dim_features = [0.0] * 8
            except Exception as e:
                logger.debug(f"Spectral pattern extraction failed for dim {dim_idx}: {e}")
                dim_features = [0.0] * 8

            # Ensure exactly features_per_dim features
            while len(dim_features) < features_per_dim:
                dim_features.append(dim_features[-1] * 0.1 if len(dim_features) > 0 else 0.0)

            features.extend(dim_features[:features_per_dim])

        # Ensure exactly target_dims features
        while len(features) < target_dims:
            features.append(features[-1] * 0.1 if len(features) > 0 else 0.0)

        return np.array(features[:target_dims], dtype=np.float32)

    def extract_breathing_cycles(self, embeddings: np.ndarray) -> np.ndarray:
        """Extract breathing cycle-specific features.

        Detects cycles, measures duration, depth, and regularity across
        multiple dimensions.

        Parameters
        ----------
        embeddings : np.ndarray
            Input embeddings with shape (timesteps, dimensions).

        Returns
        -------
        np.ndarray
            Breathing cycle features (64 dimensions).
        """
        n_seq, embed_dim = embeddings.shape
        target_dims = self.allocation["breathing_cycles"]

        # Select most respiratory-relevant dimensions
        dim_vars = np.var(embeddings, axis=0)
        top_dims = np.argsort(dim_vars)[-4:]  # Top 4 dimensions
        selected = embeddings[:, top_dims]

        features = []

        # CYCLE DETECTION via peak/valley analysis
        for dim_idx in range(selected.shape[1]):
            signal = selected[:, dim_idx]

            # Smooth the signal for better peak detection
            if len(signal) > 5:
                # Simple moving average smoothing
                window_size = min(11, len(signal) // 10)
                if window_size >= 3:
                    smoothed = np.convolve(signal, np.ones(window_size) / window_size, mode="same")
                else:
                    smoothed = signal
            else:
                smoothed = signal

            # Find peaks and valleys (breathing cycles)
            if len(smoothed) > 10:
                # Simple peak detection
                peaks = []
                valleys = []

                for i in range(1, len(smoothed) - 1):
                    if smoothed[i] > smoothed[i - 1] and smoothed[i] > smoothed[i + 1]:
                        # Local maximum (peak)
                        if len(peaks) == 0 or i - peaks[-1] > 20:  # Minimum distance between peaks
                            peaks.append(i)
                    elif smoothed[i] < smoothed[i - 1] and smoothed[i] < smoothed[i + 1]:
                        # Local minimum (valley)
                        if len(valleys) == 0 or i - valleys[-1] > 20:  # Minimum distance between valleys
                            valleys.append(i)

                # Cycle characteristics
                if len(peaks) > 1:
                    peak_intervals = np.diff(peaks)
                    features.extend(
                        [
                            np.mean(peak_intervals),  # Average breathing cycle duration
                            np.std(peak_intervals),  # Cycle duration variability
                            len(peaks) / n_seq * 1000,  # Breathing rate (cycles per 1000 timesteps)
                            np.max(peak_intervals) / (np.mean(peak_intervals) + 1e-8),  # Cycle irregularity
                        ]
                    )
                else:
                    features.extend([150.0, 30.0, 6.0, 1.2])  # Default values

                # Amplitude characteristics
                if peaks and valleys:
                    peak_vals = smoothed[peaks]
                    valley_vals = smoothed[valleys]

                    features.extend(
                        [
                            np.mean(peak_vals - np.mean(valley_vals)),  # Average breathing depth
                            np.std(peak_vals),  # Peak variability
                            np.std(valley_vals),  # Valley variability
                            np.mean(peak_vals) / (np.abs(np.mean(valley_vals)) + 1e-8),  # Peak-to-valley ratio
                        ]
                    )
                else:
                    features.extend([1.0, 0.5, 0.5, 2.0])
            else:
                features.extend([150.0, 30.0, 6.0, 1.2, 1.0, 0.5, 0.5, 2.0])

        # Cross-dimensional cycle coherence
        if selected.shape[1] > 1:
            # Calculate correlation between different dimensions' cycles
            cycle_correlations = []
            for i in range(selected.shape[1]):
                for j in range(i + 1, selected.shape[1]):
                    if np.std(selected[:, i]) > 1e-8 and np.std(selected[:, j]) > 1e-8:
                        corr = np.corrcoef(selected[:, i], selected[:, j])[0, 1]
                        cycle_correlations.append(corr if not np.isnan(corr) else 0.0)

            if cycle_correlations:
                features.extend(
                    [
                        np.mean(cycle_correlations),  # Average cross-dimensional coherence
                        np.std(cycle_correlations),  # Coherence variability
                        np.max(cycle_correlations),  # Strongest coherence
                        np.min(cycle_correlations),  # Weakest coherence
                    ]
                )
            else:
                features.extend([0.5, 0.2, 0.7, 0.3])
        else:
            features.extend([0.5, 0.2, 0.7, 0.3])

        # Ensure target dimensions
        while len(features) < target_dims:
            if len(features) > 0:
                features.append(features[-1] * 0.1 + np.random.normal(0, 0.01))  # Small variation
            else:
                features.append(0.0)

        return np.array(features[:target_dims], dtype=np.float32)

    def extract_stability_metrics(self, embeddings: np.ndarray) -> np.ndarray:
        """Extract breathing pattern stability and consistency metrics.

        Measures temporal stability, dimensional consistency, and predictability.

        Parameters
        ----------
        embeddings : np.ndarray
            Input embeddings with shape (timesteps, dimensions).

        Returns
        -------
        np.ndarray
            Stability metrics (48 dimensions).
        """
        n_seq, embed_dim = embeddings.shape
        target_dims = self.allocation["stability_metrics"]

        # Use all dimensions for stability analysis
        features = []

        # TEMPORAL STABILITY - how consistent patterns are over time
        segment_size = max(50, n_seq // 10)  # Segments of roughly equal size
        n_segments = n_seq // segment_size

        if n_segments >= 3:
            segment_stats = []

            for i in range(n_segments):
                start_idx = i * segment_size
                end_idx = min(start_idx + segment_size, n_seq)
                segment_data = embeddings[start_idx:end_idx, :]

                if segment_data.shape[0] > 0:
                    segment_stats.append(
                        {
                            "mean": np.mean(segment_data, axis=0),
                            "std": np.std(segment_data, axis=0),
                            "median": np.median(segment_data, axis=0),
                        }
                    )

            if len(segment_stats) >= 2:
                # Stability across segments
                means = np.array([s["mean"] for s in segment_stats])
                stds = np.array([s["std"] for s in segment_stats])
                medians = np.array([s["median"] for s in segment_stats])

                features.extend(
                    [
                        np.mean(np.std(means, axis=0)),  # Mean stability
                        np.mean(np.std(stds, axis=0)),  # Variability stability
                        np.mean(np.std(medians, axis=0)),  # Median stability
                        np.std(np.mean(means, axis=1)),  # Overall pattern stability
                        (np.corrcoef(np.arange(len(means)), np.mean(means, axis=1))[0, 1] if len(means) > 2 else 0.0),  # Temporal trend
                    ]
                )

        # DIMENSIONAL CONSISTENCY - how consistent each dimension is
        dim_consistency_scores = []
        for dim_idx in range(embed_dim):
            dim_data = embeddings[:, dim_idx]

            # Autocorrelation at different lags
            autocorr_scores = []
            for lag in [10, 50, 100]:
                if n_seq > lag + 10:
                    try:
                        sig1 = dim_data[:-lag]
                        sig2 = dim_data[lag:]
                        if np.std(sig1) > 1e-8 and np.std(sig2) > 1e-8:
                            autocorr = np.corrcoef(sig1, sig2)[0, 1]
                            autocorr_scores.append(autocorr if not np.isnan(autocorr) else 0.0)
                        else:
                            autocorr_scores.append(0.0)
                    except Exception as e:
                        logger.debug(f"Stability dimensional consistency failed for dim {dim_idx}, lag {lag}: {e}")
                        autocorr_scores.append(0.0)

            dim_consistency = np.mean(autocorr_scores) if autocorr_scores else 0.0
            dim_consistency_scores.append(dim_consistency)

        features.extend(
            [
                np.mean(dim_consistency_scores),  # Average dimensional consistency
                np.std(dim_consistency_scores),  # Consistency variability across dimensions
                np.max(dim_consistency_scores),  # Most consistent dimension
                np.min(dim_consistency_scores),  # Least consistent dimension
            ]
        )

        # PATTERN ENTROPY - measure of pattern predictability
        try:
            # Quantize the most variable dimension for entropy calculation
            most_var_dim = np.argmax(np.var(embeddings, axis=0))
            signal = embeddings[:, most_var_dim]

            # Simple binning for entropy
            n_bins = min(20, n_seq // 50)
            if n_bins >= 2:
                hist, _ = np.histogram(signal, bins=n_bins)
                hist = hist + 1e-8  # Avoid log(0)
                probs = hist / np.sum(hist)
                entropy = -np.sum(probs * np.log2(probs))
                normalized_entropy = entropy / np.log2(n_bins)  # Normalize to [0,1]
            else:
                normalized_entropy = 0.5

            features.append(normalized_entropy)
        except Exception as e:
            logger.debug(f"Stability pattern entropy calculation failed: {e}")
            features.append(0.5)

        # Ensure target dimensions
        while len(features) < target_dims:
            if len(features) > 0:
                features.append(features[-1] * 0.1 + 0.01)  # Small variation
            else:
                features.append(0.0)

        return np.array(features[:target_dims], dtype=np.float32)

    def extract_fatigue_indicators(self, embeddings: np.ndarray) -> np.ndarray:
        """Extract fatigue and performance degradation indicators.

        Analyzes performance changes, degradation patterns, and fatigue progression.

        Parameters
        ----------
        embeddings : np.ndarray
            Input embeddings with shape (timesteps, dimensions).

        Returns
        -------
        np.ndarray
            Fatigue indicators (32 dimensions).
        """
        n_seq, embed_dim = embeddings.shape
        target_dims = self.allocation["fatigue_indicators"]

        if n_seq < 20:
            return np.zeros(target_dims, dtype=np.float32)

        features = []

        # PERFORMANCE DEGRADATION - compare early vs late performance
        early_size = n_seq // 4
        late_size = n_seq // 4

        early_portion = embeddings[:early_size, :]
        late_portion = embeddings[-late_size:, :]

        # Activity level changes (potential fatigue indicator)
        early_activity = np.mean(np.abs(early_portion))
        late_activity = np.mean(np.abs(late_portion))
        activity_change = (late_activity - early_activity) / (early_activity + 1e-8)

        # Pattern complexity changes
        early_complexity = np.mean(np.std(early_portion, axis=0))
        late_complexity = np.mean(np.std(late_portion, axis=0))
        complexity_change = (late_complexity - early_complexity) / (early_complexity + 1e-8)

        # Breathing regularity changes
        early_regularity = 1.0 / (np.std(np.diff(early_portion, axis=0)) + 1e-8)
        late_regularity = 1.0 / (np.std(np.diff(late_portion, axis=0)) + 1e-8)
        regularity_change = (late_regularity - early_regularity) / (early_regularity + 1e-8)

        features.extend(
            [
                activity_change,  # Activity level change (fatigue)
                complexity_change,  # Pattern complexity change
                regularity_change,  # Breathing regularity change
                np.abs(activity_change),  # Magnitude of activity change
                np.abs(complexity_change),  # Magnitude of complexity change
                np.abs(regularity_change),  # Magnitude of regularity change
            ]
        )

        # PROGRESSIVE DEGRADATION - sliding window analysis
        window_size = max(20, n_seq // 8)
        step_size = window_size // 2
        n_windows = (n_seq - window_size) // step_size + 1

        if n_windows >= 3:
            window_activities = []
            window_regularities = []

            for i in range(n_windows):
                start_idx = i * step_size
                end_idx = min(start_idx + window_size, n_seq)
                window_data = embeddings[start_idx:end_idx, :]

                activity = np.mean(np.abs(window_data))
                regularity = 1.0 / (np.std(np.diff(window_data, axis=0)) + 1e-8)

                window_activities.append(activity)
                window_regularities.append(regularity)

            # Trends over time (fatigue progression)
            time_indices = np.arange(len(window_activities))

            try:
                activity_slope, _, activity_r2, _, _ = stats.linregress(time_indices, window_activities)
                regularity_slope, _, regularity_r2, _, _ = stats.linregress(time_indices, window_regularities)

                features.extend(
                    [
                        activity_slope,  # Activity trend (negative = fatigue)
                        regularity_slope,  # Regularity trend
                        activity_r2,  # Activity trend strength
                        regularity_r2,  # Regularity trend strength
                    ]
                )
            except Exception as e:
                features.extend([0.0, 0.0, 0.0, 0.0])
                logger.debug(f"Fatigue trend analysis failed: {e}")

        # Ensure target dimensions
        while len(features) < target_dims:
            if len(features) > 0:
                features.append(features[-1] * 0.05)  # Small scaled copy
            else:
                features.append(0.0)

        return np.array(features[:target_dims], dtype=np.float32)

    def extract_individual_signature(self, embeddings: np.ndarray) -> np.ndarray:
        """Extract compact runner-specific breathing signature features.

        Captures unique characteristics of individual breathing patterns.

        Parameters
        ----------
        embeddings : np.ndarray
            Input embeddings with shape (timesteps, dimensions).

        Returns
        -------
        np.ndarray
            Individual signature features (16 dimensions).
        """
        n_seq, embed_dim = embeddings.shape
        target_dims = self.allocation["individual_signature"]

        features = []

        # UNIQUE PATTERN CHARACTERISTICS (compressed)
        dim_activities = np.mean(np.abs(embeddings), axis=0)
        dim_variabilities = np.std(embeddings, axis=0)

        features.extend(
            [
                np.max(dim_activities) / (np.mean(dim_activities) + 1e-8),  # Activity concentration
                np.max(dim_variabilities) / (np.mean(dim_variabilities) + 1e-8),  # Variability concentration
                stats.skew(embeddings.flatten()),  # Pattern asymmetry
                stats.kurtosis(embeddings.flatten()),  # Pattern peakedness
                np.percentile(embeddings.flatten(), 95) / (np.percentile(embeddings.flatten(), 5) + 1e-8),  # Dynamic range
            ]
        )

        # TEMPORAL SIGNATURE (compressed)
        if n_seq > 10:
            diff_norms = np.linalg.norm(np.diff(embeddings, axis=0), axis=1)
            features.extend(
                [
                    np.mean(diff_norms),  # Average change rate
                    np.std(diff_norms),  # Change rate variability
                    np.max(diff_norms) / (np.mean(diff_norms) + 1e-8),  # Change rate extremity
                    np.sum(diff_norms > np.percentile(diff_norms, 90)) / len(diff_norms),  # Burst frequency
                ]
            )
        else:
            features.extend([0.0, 0.0, 1.0, 0.1])

        # DIMENSIONAL PREFERENCES (compressed)
        sorted_activities = np.sort(dim_activities)
        sorted_vars = np.sort(dim_variabilities)

        features.extend(
            [
                sorted_activities[-1] / (sorted_activities[0] + 1e-8),  # Activity range ratio
                sorted_vars[-1] / (sorted_vars[0] + 1e-8),  # Variability range ratio
                np.mean(sorted_activities[-3:]) / (np.mean(sorted_activities[:3]) + 1e-8),  # Top vs bottom activity
                np.mean(sorted_vars[-3:]) / (np.mean(sorted_vars[:3]) + 1e-8),  # Top vs bottom variability
            ]
        )

        # CROSS-DIMENSIONAL INTERACTIONS
        if embed_dim > 1:
            # Simple cross-correlation summary
            cross_corr_sum = 0.0
            cross_corr_count = 0
            for i in range(min(5, embed_dim)):
                for j in range(i + 1, min(5, embed_dim)):
                    if np.std(embeddings[:, i]) > 1e-8 and np.std(embeddings[:, j]) > 1e-8:
                        corr = np.corrcoef(embeddings[:, i], embeddings[:, j])[0, 1]
                        if not np.isnan(corr):
                            cross_corr_sum += np.abs(corr)
                            cross_corr_count += 1

            avg_cross_corr = cross_corr_sum / (cross_corr_count + 1e-8)
            features.append(avg_cross_corr)
        else:
            features.append(0.0)

        # Pad to exactly target dimensions
        while len(features) < target_dims:
            if len(features) > 0:
                features.append(features[-1] * 0.1)  # Small scaled copy
            else:
                features.append(0.0)

        return np.array(features[:target_dims], dtype=np.float32)

    def extract_global_summary(self, embeddings: np.ndarray) -> np.ndarray:
        """Extract compact overall session summary features.

        Provides consolidated summary statistics of entire breathing session.

        Parameters
        ----------
        embeddings : np.ndarray
            Input embeddings with shape (timesteps, dimensions).

        Returns
        -------
        np.ndarray
            Global summary features (16 dimensions).
        """
        n_seq, embed_dim = embeddings.shape
        target_dims = self.allocation["global_summary"]

        features = []

        # Essential session characteristics (compressed to 16 dimensions)
        features.extend(
            [
                n_seq / 1000.0,  # Normalized session length
                np.mean(embeddings),  # Overall mean activity
                np.std(embeddings),  # Overall variability
                np.median(np.abs(embeddings)),  # Robust activity level
                np.max(embeddings) - np.min(embeddings),  # Total range
                np.percentile(embeddings, 90) - np.percentile(embeddings, 10),  # Central range
                np.mean(np.abs(embeddings)),  # Average absolute activity
                stats.skew(embeddings.flatten()),  # Pattern asymmetry
                stats.kurtosis(embeddings.flatten()),  # Pattern peakedness
            ]
        )

        # Compact dimensional summary
        dim_stds = np.std(embeddings, axis=0)

        features.extend(
            [
                np.mean(dim_stds),  # Average dimension variability
                np.std(dim_stds),  # Dimension variability spread
                np.max(dim_stds) / (np.mean(dim_stds) + 1e-8),  # Variability concentration
            ]
        )

        # Temporal characteristics
        if n_seq > 1:
            overall_diff = np.diff(embeddings, axis=0)
            features.extend(
                [
                    np.mean(np.linalg.norm(overall_diff, axis=1)),  # Average movement rate
                    np.std(np.linalg.norm(overall_diff, axis=1)),  # Movement variability
                    np.max(np.linalg.norm(overall_diff, axis=1)),  # Maximum movement
                    np.sum(np.linalg.norm(overall_diff, axis=1) > np.percentile(np.linalg.norm(overall_diff, axis=1), 90)) / len(overall_diff),  # High activity frequency
                ]
            )
        else:
            features.extend([0.0, 0.0, 0.0, 0.0])

        return np.array(features[:target_dims], dtype=np.float32)

    def aggregate_time_aware(self, embeddings: np.ndarray) -> np.ndarray:
        """Main aggregation function using time-aware methods.

        Extracts 512-dimensional feature vector from breathing embeddings
        using multiple time-aware feature extraction methods.

        Parameters
        ----------
        embeddings : np.ndarray
            Input embeddings with shape (timesteps, dimensions).

        Returns
        -------
        np.ndarray
            Aggregated features of 512 dimensions.
        """
        embeddings = self._validate_embeddings(embeddings)

        self._log("TIME-AWARE BREATHING PATTERN AGGREGATION")
        self._log("=" * 50)
        self._log(f"Input: {embeddings.shape[0]} timesteps, {embeddings.shape[1]} features")

        # Extract all feature types
        features_list = []

        chunk1 = self.extract_early_mid_late_patterns(embeddings)
        features_list.append(chunk1)
        self._log(f"Early/Mid/Late Patterns: {len(chunk1)} dimensions")

        chunk2 = self.extract_breathing_rhythm(embeddings)
        features_list.append(chunk2)
        self._log(f"Breathing Rhythm: {len(chunk2)} dimensions")

        chunk3 = self.extract_pattern_evolution(embeddings)
        features_list.append(chunk3)
        self._log(f"Pattern Evolution: {len(chunk3)} dimensions")

        chunk4 = self.extract_spectral_patterns(embeddings)
        features_list.append(chunk4)
        self._log(f"Spectral Patterns: {len(chunk4)} dimensions")

        chunk5 = self.extract_breathing_cycles(embeddings)
        features_list.append(chunk5)
        self._log(f"Breathing Cycles: {len(chunk5)} dimensions")

        chunk6 = self.extract_stability_metrics(embeddings)
        features_list.append(chunk6)
        self._log(f"Stability Metrics: {len(chunk6)} dimensions")

        chunk7 = self.extract_fatigue_indicators(embeddings)
        features_list.append(chunk7)
        self._log(f"Fatigue Indicators: {len(chunk7)} dimensions")

        chunk8 = self.extract_individual_signature(embeddings)
        features_list.append(chunk8)
        self._log(f"Individual Signature: {len(chunk8)} dimensions")

        chunk9 = self.extract_global_summary(embeddings)
        features_list.append(chunk9)
        self._log(f"Global Summary: {len(chunk9)} dimensions")

        # Concatenate all features
        final_features = np.concatenate(features_list)

        # Ensure exactly target_dims features
        if len(final_features) < self.target_dims:
            # Pad with derived features
            padding_needed = self.target_dims - len(final_features)
            padding = []
            for i in range(padding_needed):
                if len(final_features) > 0:
                    padding.append(final_features[i % len(final_features)] * 0.01)
                else:
                    padding.append(0.0)
            final_features = np.concatenate([final_features, padding])
        elif len(final_features) > self.target_dims:
            # Truncate to exact size
            final_features = final_features[: self.target_dims]

        # Apply robust scaling
        final_features = self._apply_robust_scaling(final_features)

        # Quality assessment
        zero_count = np.sum(final_features == 0)
        zero_pct = 100 * zero_count / len(final_features)

        self._log("\nAGGREGATION COMPLETE")
        self._log(f"Output dimensions: {len(final_features)}")
        self._log(f"Zero values: {zero_count}/{len(final_features)} ({zero_pct:.1f}%)")
        self._log(f"Variance: {np.var(final_features):.2e}")
        self._log(f"Range: [{np.min(final_features):.3f}, {np.max(final_features):.3f}]")

        return final_features

    def _apply_robust_scaling(self, features: np.ndarray) -> np.ndarray:
        """Apply robust scaling to features.

        Parameters
        ----------
        features : np.ndarray
            Input features to scale.

        Returns
        -------
        np.ndarray
            Scaled features clipped to [-5, 5] range.
        """
        try:
            scaler = RobustScaler()
            features_2d = features.reshape(-1, 1)
            features_scaled = scaler.fit_transform(features_2d).flatten()

            # Clip extreme values
            features_scaled = np.clip(features_scaled, -5, 5)

            return features_scaled
        except Exception as e:
            logger.debug(f"Robust scaling failed: {e}")
            return np.clip(features, -10, 10)  # Fallback clipping

    def process_single_embedding_file(self, input_path: str, output_dir: str = None) -> str:
        """Process a single embedding file and save the aggregated result.

        Parameters
        ----------
        input_path : str
            Path to the input embedding file (.npy).
        output_dir : str, optional
            Output directory for aggregated embeddings.
            Defaults to ml/ml_data/embeddings_agg/.

        Returns
        -------
        str
            Path to the saved aggregated embedding file.
        """
        from pathlib import Path

        # Set default output directory
        if output_dir is None:
            output_dir = "ml/ml_data/embeddings_agg"

        # Create output directory if it doesn't exist
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        # Extract filename and create output filename
        input_filename = Path(input_path).stem  # Remove .npy extension
        output_filename = f"{input_filename}_agg.npy"
        output_path = Path(output_dir) / output_filename

        try:
            # Load embeddings
            if self.verbose:
                logger.info(f"Loading embeddings from: {input_path}")

            embeddings = np.load(input_path)

            if self.verbose:
                logger.info(f"Original shape: {embeddings.shape}")

            # Aggregate using time-aware method
            aggregated = self.aggregate_time_aware(embeddings)

            # Save aggregated embeddings
            np.save(output_path, aggregated)

            if self.verbose:
                logger.info(f"Successfully saved: {output_path}")
                logger.info(f"Aggregated shape: {aggregated.shape}")
                zero_pct = 100 * np.sum(aggregated == 0) / len(aggregated)
                logger.info(f"Zero features: {np.sum(aggregated == 0)}/{len(aggregated)} ({zero_pct:.1f}%)")
                logger.info("-" * 60)

            return str(output_path)

        except Exception as e:
            logger.error(f"Error processing {input_path}: {e}")
            return None

    def process_embeddings_directory(
        self,
        input_dir: str = "ml/ml_data/embeddings",
        output_dir: str = "ml/ml_data/embeddings_agg",
        pattern: str = "embeddings_*.npy",
    ) -> dict:
        """Process all embedding files in a directory.

        Parameters
        ----------
        input_dir : str, optional
            Input directory containing embedding files.
            Defaults to ml/ml_data/embeddings.
        output_dir : str, optional
            Output directory for aggregated embeddings.
            Defaults to ml/ml_data/embeddings_agg.
        pattern : str, optional
            File pattern to match. Defaults to embeddings_*.npy.

        Returns
        -------
        dict
            Dictionary with keys: processed (count), failed (count),
            files (list), output_files (list).
        """
        import glob
        from pathlib import Path

        # Find all matching files
        search_pattern = str(Path(input_dir) / pattern)
        embedding_files = glob.glob(search_pattern)

        if not embedding_files:
            logger.error(f"No files found matching pattern: {search_pattern}")
            return {"processed": 0, "failed": 0, "files": []}

        logger.info(f"Processing {len(embedding_files)} embedding files")
        logger.info(f"Input directory: {input_dir}")
        logger.info(f"Output directory: {output_dir}")
        logger.info("=" * 60)

        results = {"processed": 0, "failed": 0, "files": [], "output_files": []}

        # Process each file
        for i, input_file in enumerate(sorted(embedding_files), 1):
            logger.info(f"[{i}/{len(embedding_files)}] Processing: {Path(input_file).name}")

            output_file = self.process_single_embedding_file(input_file, output_dir)

            results["files"].append(input_file)

            if output_file:
                results["processed"] += 1
                results["output_files"].append(output_file)
            else:
                results["failed"] += 1

        # Summary
        logger.info("\nPROCESSING SUMMARY")
        logger.info(f"Successfully processed: {results['processed']}")
        logger.info(f"Failed: {results['failed']}")
        logger.info(f"Output directory: {output_dir}")

        return results


def test_single_embedding_processing():
    """Test processing a single embedding file."""

    logger.info("TESTING SINGLE EMBEDDING PROCESSING")
    logger.info("=" * 50)

    try:
        # Create aggregator
        aggregator = EmbeddingAggregator(verbose=True)

        # Process the example file
        input_path = "ml/ml_data/embeddings/embeddings_ID_1_Loose_6_min.npy"
        output_path = aggregator.process_single_embedding_file(input_path)

        if output_path:
            logger.info("\nTest PASSED")
            logger.info(f"Input: {input_path}")
            logger.info(f"Output: {output_path}")

            # Verify the saved file
            saved_data = np.load(output_path)
            logger.info(f"Saved shape: {saved_data.shape}")
            logger.info(f"Saved dtype: {saved_data.dtype}")
        else:
            logger.error("Test FAILED: Could not process file")

    except Exception as e:
        logger.error(f"Test FAILED: {e}")


def test_final_time_aware_aggregator():
    """Test the final time-aware aggregator."""

    logger.info("TESTING FINAL TIME-AWARE AGGREGATOR")
    logger.info("=" * 50)

    try:
        # Load sample
        sample_path = "ml/ml_data/embeddings/embeddings_ID_1_Loose_6_min.npy"
        embeddings = np.load(sample_path)

        if embeddings.ndim == 3:
            embeddings = embeddings.squeeze(1)  # (timesteps, 26)

        logger.info(f"Sample loaded: {embeddings.shape}")

        # Test aggregator
        aggregator = EmbeddingAggregator(verbose=True)
        features = aggregator.aggregate_time_aware(embeddings)

        logger.info("\nFINAL QUALITY ASSESSMENT")
        zero_count = np.sum(features == 0)
        logger.info(f"Zero features: {zero_count}/{len(features)} ({100 * zero_count / len(features):.1f}%)")
        logger.info(f"Variance: {np.var(features):.2e}")
        logger.info(f"Range: [{np.min(features):.3f}, {np.max(features):.3f}]")

        return features

    except Exception as e:
        logger.error(f"Test FAILED: {e}")
        return None


if __name__ == "__main__":
    # Test the aggregator and single file processing
    test_final_time_aware_aggregator()
    logger.info("\n" + "=" * 60 + "\n")
    test_single_embedding_processing()
