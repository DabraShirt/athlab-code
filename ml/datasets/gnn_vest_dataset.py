# %%
import warnings
from pathlib import Path
from typing import Dict, List, Literal, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from ml.utils.emt_pandas import Emt3DAccessor, EmtAccessor
from ml.utils.logging_utils import get_logger

logger = get_logger("athlab.gnn_vest_dataset")
logger.info(f"Emt3DAccessor: {Emt3DAccessor}")
logger.info(f"EmtAccessor: {EmtAccessor}")


class GNNVestDataset:
    """
    Enhanced GNN Dataset for Vest Motion Capture Data

    Supports multiple graph creation modes:
    - 'whole_file': Entire file as one graph with temporal connections
    - 'timestep': Each row (timestep) as separate graph
    - 'sequence': Multiple consecutive rows as one graph (sliding window)

    Features:
    - Automatic data validation and cleaning
    - Multiple scaling strategies
    - Recording-aware splits to prevent data leakage
    - Comprehensive visualization tools
    - Automatic condition detection (rest vs exercise)
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
        mode: Literal["whole_file", "timestep", "sequence"] = "timestep",
        sequence_length: int = 10,
        sequence_step: int = 1,
        test_size: float = 0.2,
        val_size: float = 0.1,
        scaler_type: Literal["standard", "minmax", "robust", "none"] = "standard",
        validate_data: bool = True,
    ):
        """
        Initialize GNN Vest Dataset.

        Parameters:
        -----------
        dataframe : pd.DataFrame
            Input dataframe with vest motion capture data
        mode : str
            Graph creation mode ('whole_file', 'timestep', 'sequence')
        sequence_length : int
            Length of sequences for 'sequence' mode
        sequence_step : int
            Step size for sequence sliding window (1=consecutive, >1=skip rows)
        test_size : float
            Proportion of data for testing
        val_size : float
            Proportion of data for validation
        scaler_type : str
            Type of scaler ('standard', 'minmax', 'robust', 'none')
        validate_data : bool
            Whether to validate input data
        """

        assert isinstance(dataframe, pd.DataFrame), "Input must be a pandas DataFrame"

        # Validate mode
        valid_modes = ["timestep", "sequence", "whole_file"]
        if mode not in valid_modes:
            raise ValueError(f"Mode '{mode}' not supported. Valid modes: {valid_modes}")

        # Validate sequence mode parameters
        if mode == "sequence" and sequence_length is None:
            raise ValueError("sequence_length must be specified for 'sequence' mode")

        self.original_dataframe = dataframe.copy()
        self.dataframe = dataframe
        self.mode = mode
        self.sequence_length = sequence_length
        self.sequence_step = sequence_step
        self.test_size = test_size
        self.val_size = val_size
        self.scaler_type = scaler_type

        # Initialize scaler
        self.scaler = self._create_scaler(scaler_type)

        # Data validation
        if validate_data:
            self._validate_data()

        # Store metadata if available
        self.has_metadata = any(col in dataframe.columns for col in ["person_id", "recording_id"])
        if self.has_metadata:
            self.metadata_cols = [
                col
                for col in [
                    "person_id",
                    "recording_id",
                    "time_condition",
                    "filename",
                    "condition",
                ]
                if col in dataframe.columns
            ]
        else:
            self.metadata_cols = []

        # Initialize containers
        self.node_features = None
        self.data = None
        self.graphs = None

        logger.info("GNNVestDataset initialized:")
        logger.info(f"   - Mode: {self.mode}")
        logger.info(f"   - Data shape: {self.dataframe.shape}")
        logger.info(f"   - Scaler: {scaler_type}")
        if self.mode == "sequence":
            logger.info(f"   - Sequence length: {sequence_length}")
            logger.info(f"   - Sequence step: {sequence_step}")
        if self.has_metadata:
            logger.info(f"   - Metadata columns: {self.metadata_cols}")

    def _create_scaler(self, scaler_type: str):
        """Create the appropriate scaler."""
        if scaler_type == "standard":
            return StandardScaler()
        elif scaler_type == "minmax":
            return MinMaxScaler()
        elif scaler_type == "robust":
            return RobustScaler()
        elif scaler_type == "none":
            return None  # No scaling
        else:
            raise ValueError(f"Unknown scaler type: {scaler_type}")

    def _validate_data(self):
        """Validate the input dataframe."""
        logger.info("Validating data...")

        # Check for required columns
        required_patterns = [f"{i}.X" for i in range(1, 90)] + [f"{i}.Y" for i in range(1, 90)] + [f"{i}.Z" for i in range(1, 90)]
        missing_patterns = [p for p in required_patterns if p not in self.dataframe.columns]

        if missing_patterns:
            warnings.warn(f"Missing {len(missing_patterns)} coordinate columns. First few: {missing_patterns[:5]}")
            logger.warning(f"Missing {len(missing_patterns)} coordinate columns. First few: {missing_patterns[:5]}")

        # Check for NaN values
        nan_counts = self.dataframe.isnull().sum()
        if nan_counts.sum() > 0:
            logger.warning(f"Found NaN values in {(nan_counts > 0).sum()} columns")
            warnings.warn(f"Found missing values in {(nan_counts > 0).sum()} columns")
            top_nan_cols = nan_counts.nlargest(5)
            logger.info(f"   Top NaN columns: {dict(top_nan_cols)}")

        # Check data ranges
        numeric_cols = self.dataframe.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            data_stats = self.dataframe[numeric_cols].describe()
            extreme_vals = []
            for col in numeric_cols:
                if abs(data_stats.loc["min", col]) > 10000 or abs(data_stats.loc["max", col]) > 10000:
                    extreme_vals.append(col)

            if extreme_vals:
                logger.warning(f"Found {len(extreme_vals)} columns with extreme values (>10000)")
                warnings.warn(f"Found extreme values in {len(extreme_vals)} columns")
                if len(extreme_vals) <= 5:
                    logger.info(f"   Columns: {extreme_vals}")

        logger.info("Data validation completed")

    def get_data_summary(self) -> Dict:
        """Get comprehensive summary of the dataset."""
        summary = {
            "basic_info": {
                "shape": self.dataframe.shape,
                "mode": self.mode,
                "sequence_length": (self.sequence_length if self.mode == "sequence" else None),
                "has_metadata": self.has_metadata,
                "scaler_type": self.scaler_type,
            },
            "data_quality": {},
            "vest_points": {},
            "temporal_info": {},
        }

        # Data quality metrics
        summary["data_quality"]["missing_values"] = self.dataframe.isnull().sum().sum()
        summary["data_quality"]["duplicate_rows"] = self.dataframe.duplicated().sum()

        # Vest point information
        coord_cols = [col for col in self.dataframe.columns if any(col.endswith(suffix) for suffix in [".X", ".Y", ".Z"])]
        summary["vest_points"]["total_coordinates"] = len(coord_cols)
        summary["vest_points"]["expected_points"] = 89
        summary["vest_points"]["complete_points"] = len(coord_cols) // 3

        # Temporal information
        if "Time" in self.dataframe.columns:
            time_data = self.dataframe["Time"]
            summary["temporal_info"]["duration"] = float(time_data.max() - time_data.min())
            summary["temporal_info"]["fps"] = len(time_data) / (time_data.max() - time_data.min()) if time_data.max() > time_data.min() else 0
            summary["temporal_info"]["total_timesteps"] = len(time_data)

        return summary

    def preprocess_data(self, apply_scaling: bool = True):
        """
        Extract and preprocess x, y, z coordinates as node features.

        Parameters:
        -----------
        apply_scaling : bool
            Whether to apply scaling to the data
        """
        logger.info(f"Preprocessing data in '{self.mode}' mode...")

        # Get all coordinate columns
        feature_cols = [col for col in self.dataframe.columns if col not in ["Frame", "Time"] + self.metadata_cols]
        coord_cols = [col for col in feature_cols if any(col.endswith(suffix) for suffix in [".X", ".Y", ".Z"])]

        # Initialize lists to store coordinates
        x_coords = []
        y_coords = []
        z_coords = []

        # Extract coordinates for each point (1-89)
        available_points = set()
        for col in coord_cols:
            if "." in col:
                point_num = int(col.split(".")[0])
                available_points.add(point_num)

        available_points = sorted(available_points)
        logger.info(f"   - Found coordinates for {len(available_points)} vest points")

        for point in available_points:
            x_col = f"{point}.X"
            y_col = f"{point}.Y"
            z_col = f"{point}.Z"

            if all(col in self.dataframe.columns for col in [x_col, y_col, z_col]):
                x_coords.append(self.dataframe[x_col].values)
                y_coords.append(self.dataframe[y_col].values)
                z_coords.append(self.dataframe[z_col].values)

        if not x_coords:
            raise ValueError("No valid coordinate data found")

        # Stack coordinates to create node features
        x_stack = np.stack(x_coords, axis=1)  # Shape: [num_timesteps, num_points]
        y_stack = np.stack(y_coords, axis=1)
        z_stack = np.stack(z_coords, axis=1)

        # Combine into final node features: [num_timesteps, num_points, 3]
        self.node_features = np.stack([x_stack, y_stack, z_stack], axis=2)

        # Handle NaN values
        nan_count = np.isnan(self.node_features).sum()
        if nan_count > 0:
            logger.warning(f"   Found {nan_count} NaN values, replacing with zeros")
            self.node_features = np.nan_to_num(self.node_features, nan=0.0)

        # Apply scaling if requested
        if apply_scaling:
            self._apply_scaling()

        logger.info(f"   Node features shape: {self.node_features.shape}")
        return self.node_features

    def _apply_scaling(self):
        """Apply scaling to the node features."""
        if self.node_features is None:
            raise ValueError("Node features not computed. Call preprocess_data() first.")

        # Skip scaling if scaler is None
        if self.scaler is None:
            logger.info("   No scaling applied (scaler_type='none')")
            return

        original_shape = self.node_features.shape

        # Reshape for scaling: [num_timesteps * num_points, 3]
        features_reshaped = self.node_features.reshape(-1, 3)

        # Fit and transform
        features_scaled = self.scaler.fit_transform(features_reshaped)

        # Reshape back to original shape
        self.node_features = features_scaled.reshape(original_shape)

        logger.info(f"   Applied {self.scaler_type} scaling")

        # Log scaling statistics
        if hasattr(self.scaler, "mean_"):
            logger.info(f"      - Mean: [{self.scaler.mean_[0]:.2f}, {self.scaler.mean_[1]:.2f}, {self.scaler.mean_[2]:.2f}]")
            logger.info(f"      - Std: [{self.scaler.scale_[0]:.2f}, {self.scaler.scale_[1]:.2f}, {self.scaler.scale_[2]:.2f}]")

    def create_graphs(self) -> List[Data]:
        """
        Create graphs based on the specified mode.

        Returns:
        --------
        List[Data]
            List of PyTorch Geometric Data objects
        """
        if self.node_features is None:
            self.preprocess_data()

        if self.mode == "whole_file":
            self.graphs = self._create_whole_file_graph()
        elif self.mode == "timestep":
            self.graphs = self._create_timestep_graphs()
        elif self.mode == "sequence":
            self.graphs = self._create_sequence_graphs()
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

        logger.info(f"Created {len(self.graphs)} graphs in '{self.mode}' mode")
        return self.graphs

    def _create_whole_file_graph(self) -> List[Data]:
        """Create a single graph representing the entire file with temporal connections."""
        logger.info("   Creating whole file graph with temporal connections...")

        # Create nodes: each timestep creates a set of vest points
        all_node_features = []
        node_timesteps = []
        node_point_ids = []

        num_timesteps, num_points, num_features = self.node_features.shape

        for t in range(num_timesteps):
            for p in range(num_points):
                all_node_features.append(self.node_features[t, p])
                node_timesteps.append(t)
                node_point_ids.append(p)

        node_features_tensor = torch.tensor(all_node_features, dtype=torch.float)

        # Create edges: spatial (within timestep) + temporal (across timesteps)
        edge_list = []

        # Spatial edges (within each timestep)
        spatial_connections = self.node_connections
        for t in range(num_timesteps):
            base_idx = t * num_points
            for src, dst in spatial_connections:
                if src <= num_points and dst <= num_points:  # Valid point indices
                    src_idx = base_idx + (src - 1)  # Convert to 0-based
                    dst_idx = base_idx + (dst - 1)
                    edge_list.extend([[src_idx, dst_idx], [dst_idx, src_idx]])  # Bidirectional

        # Temporal edges (same point across consecutive timesteps)
        for t in range(num_timesteps - 1):
            for p in range(num_points):
                curr_idx = t * num_points + p
                next_idx = (t + 1) * num_points + p
                edge_list.extend([[curr_idx, next_idx], [next_idx, curr_idx]])

        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()

        # Calculate edge attributes (distances)
        edge_attr = []
        for src_idx, dst_idx in edge_list:
            src_coords = node_features_tensor[src_idx]
            dst_coords = node_features_tensor[dst_idx]
            distance = torch.sqrt(torch.sum((src_coords - dst_coords) ** 2))
            edge_attr.append(distance.item())

        edge_attr_tensor = torch.tensor(edge_attr, dtype=torch.float).view(-1, 1)

        # Create the graph
        graph = Data(x=node_features_tensor, edge_index=edge_index, edge_attr=edge_attr_tensor)

        # Add metadata
        graph.num_timesteps = num_timesteps
        graph.num_points_per_timestep = num_points
        graph.mode = "whole_file"

        return [graph]

    def _create_timestep_graphs(self) -> List[Data]:
        """Create one graph per timestep."""
        logger.info("   Creating timestep graphs...")

        graphs = []
        spatial_connections = self.node_connections

        # Precompute edge connections (same for all timesteps)
        edge_index = [[src - 1, dst - 1] for src, dst in spatial_connections]  # Convert to 0-based
        bidirectional_edges = edge_index + [[dst, src] for src, dst in edge_index]
        edge_index_tensor = torch.tensor(bidirectional_edges, dtype=torch.long).t().contiguous()

        for timestep in tqdm(range(self.node_features.shape[0]), desc="Creating timestep graphs"):
            # Node features for this timestep
            node_features = torch.tensor(self.node_features[timestep], dtype=torch.float)

            # Calculate edge attributes (distances)
            edge_attr = []
            for src_idx, dst_idx in bidirectional_edges:
                src_coords = node_features[src_idx]
                dst_coords = node_features[dst_idx]
                distance = torch.sqrt(torch.sum((src_coords - dst_coords) ** 2))
                edge_attr.append(distance.item())

            edge_attr_tensor = torch.tensor(edge_attr, dtype=torch.float).view(-1, 1)

            # Create graph
            graph = Data(
                x=node_features,
                edge_index=edge_index_tensor,
                edge_attr=edge_attr_tensor,
            )

            # Add metadata
            graph.timestep = timestep
            graph.mode = "timestep"
            if timestep < len(self.dataframe):
                for col in self.metadata_cols:
                    if col in self.dataframe.columns:
                        setattr(graph, col, self.dataframe.iloc[timestep][col])
                # Add subject_id and file_index for splitting
                if "subject_id" in self.dataframe.columns:
                    graph.subject_id = self.dataframe.iloc[timestep]["subject_id"]
                if "file_index" in self.dataframe.columns:
                    graph.file_index = self.dataframe.iloc[timestep]["file_index"]

            graphs.append(graph)

        return graphs

    def _create_sequence_graphs(self) -> List[Data]:
        """Create graphs from sequences of consecutive timesteps with optional step size."""
        logger.info(f"   Creating sequence graphs (length={self.sequence_length}, step={self.sequence_step})...")

        if self.sequence_length >= self.node_features.shape[0]:
            logger.warning("   Sequence length >= data length, creating single sequence graph")
            return [self._create_single_sequence_graph(0, self.node_features.shape[0])]

        graphs = []
        # Calculate number of sequences considering step size
        max_start = self.node_features.shape[0] - self.sequence_length
        sequence_starts = list(range(0, max_start + 1, self.sequence_step))

        for seq_start in tqdm(sequence_starts, desc="Creating sequence graphs"):
            seq_end = seq_start + self.sequence_length
            graph = self._create_single_sequence_graph(seq_start, seq_end)
            graphs.append(graph)

        return graphs

    def _create_single_sequence_graph(self, start_idx: int, end_idx: int) -> Data:
        """Create a single sequence graph from timesteps [start_idx:end_idx]."""
        sequence_data = self.node_features[start_idx:end_idx]  # Shape: [seq_len, num_points, 3]
        seq_len, num_points, num_features = sequence_data.shape

        # Flatten sequence into nodes: each (timestep, point) becomes a node
        all_node_features = []
        node_timesteps = []
        node_point_ids = []

        for t in range(seq_len):
            for p in range(num_points):
                all_node_features.append(sequence_data[t, p])
                node_timesteps.append(t)
                node_point_ids.append(p)

        node_features_tensor = torch.tensor(all_node_features, dtype=torch.float)

        # Create edges: spatial + temporal
        edge_list = []
        spatial_connections = self.node_connections

        # Spatial edges (within each timestep)
        for t in range(seq_len):
            base_idx = t * num_points
            for src, dst in spatial_connections:
                if src <= num_points and dst <= num_points:
                    src_idx = base_idx + (src - 1)
                    dst_idx = base_idx + (dst - 1)
                    edge_list.extend([[src_idx, dst_idx], [dst_idx, src_idx]])

        # Temporal edges (same point across consecutive timesteps)
        for t in range(seq_len - 1):
            for p in range(num_points):
                curr_idx = t * num_points + p
                next_idx = (t + 1) * num_points + p
                edge_list.extend([[curr_idx, next_idx], [next_idx, curr_idx]])

        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()

        # Calculate edge attributes
        edge_attr = []
        for src_idx, dst_idx in edge_list:
            src_coords = node_features_tensor[src_idx]
            dst_coords = node_features_tensor[dst_idx]
            distance = torch.sqrt(torch.sum((src_coords - dst_coords) ** 2))
            edge_attr.append(distance.item())

        edge_attr_tensor = torch.tensor(edge_attr, dtype=torch.float).view(-1, 1)

        # Create graph
        graph = Data(x=node_features_tensor, edge_index=edge_index, edge_attr=edge_attr_tensor)

        # Add metadata
        graph.sequence_start = start_idx
        graph.sequence_end = end_idx
        graph.sequence_length = seq_len
        graph.num_points_per_timestep = num_points
        graph.mode = "sequence"

        # Add subject_id and file_index from the first timestep in sequence
        if start_idx < len(self.dataframe):
            if "subject_id" in self.dataframe.columns:
                graph.subject_id = self.dataframe.iloc[start_idx]["subject_id"]
            if "file_index" in self.dataframe.columns:
                graph.file_index = self.dataframe.iloc[start_idx]["file_index"]

        return graph

    def create_dataloaders(
        self,
        batch_size: int = 32,
        split_ratio: List[float] = [0.7, 0.15, 0.15],
        recording_aware: bool = True,
        subject_aware: bool = False,
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Create train/validation/test DataLoaders.

        Parameters:
        -----------
        batch_size : int
            Batch size for DataLoaders
        split_ratio : List[float]
            Ratios for [train, val, test] splits
        recording_aware : bool
            Whether to split by recordings to prevent data leakage
        subject_aware : bool
            Whether to split by subjects to prevent subject-level data leakage

        Returns:
        --------
        Tuple[DataLoader, DataLoader, DataLoader]
            Train, validation, and test DataLoaders
        """
        if self.graphs is None:
            self.create_graphs()

        # CRITICAL: Enforce batch_size=1 for timestep mode to prevent temporal leakage
        if self.mode == "timestep" and batch_size > 1:
            if getattr(self, "force_batch_size_1_for_timestep", False):
                logger.warning(f"TEMPORAL LEAKAGE PREVENTION: Forcing batch_size=1 for timestep mode (was {batch_size})")
                batch_size = 1
            else:
                logger.warning(f"WARNING: batch_size={batch_size} in timestep mode may cause temporal leakage!")
                logger.warning("Adjacent timesteps from same recording may appear in same batch.")
                logger.warning("Consider using batch_size=1 or switch to 'sequence' mode.")

        if subject_aware and self.has_metadata and "subject_id" in self.dataframe.columns:
            train_graphs, val_graphs, test_graphs = self._subject_aware_split(split_ratio)
        elif recording_aware and self.has_metadata and "recording_id" in self.dataframe.columns:
            train_graphs, val_graphs, test_graphs = self._recording_aware_split(split_ratio)
        else:
            train_graphs, val_graphs, test_graphs = self._random_split(split_ratio)

        # Create DataLoaders
        train_loader = DataLoader(train_graphs, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_graphs, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_graphs, batch_size=batch_size, shuffle=False)

        logger.info("DataLoaders created:")
        logger.info(f"   Training: {len(train_graphs)} graphs ({len(train_loader)} batches)")
        logger.info(f"   Validation: {len(val_graphs)} graphs ({len(val_loader)} batches)")
        logger.info(f"   Test: {len(test_graphs)} graphs ({len(test_loader)} batches)")

        return train_loader, val_loader, test_loader

    def _recording_aware_split(self, split_ratio: List[float]) -> Tuple[List[Data], List[Data], List[Data]]:
        """Split graphs by recordings to prevent data leakage."""
        logger.info("   Using recording-aware splits to prevent recording-level data leakage...")

        if self.mode == "whole_file":
            logger.warning("Recording-aware split not applicable for whole_file mode, using random split")
            return self._random_split(split_ratio)

        # Group graphs by recording_id
        recording_groups = {}
        recording_subject_info = {}  # Track which subject each recording belongs to

        for i, graph in enumerate(self.graphs):
            if hasattr(graph, "recording_id"):
                rec_id = graph.recording_id
            else:
                # Fallback: use index to determine recording
                if self.mode == "timestep":
                    rec_id = self.dataframe.iloc[i]["recording_id"] if i < len(self.dataframe) else "unknown"
                else:
                    rec_id = "unknown"

            if rec_id not in recording_groups:
                recording_groups[rec_id] = []
            recording_groups[rec_id].append(graph)

            # Track subject information for logging
            if hasattr(graph, "subject_id"):
                recording_subject_info[rec_id] = graph.subject_id
            elif i < len(self.dataframe) and "subject_id" in self.dataframe.columns:
                recording_subject_info[rec_id] = self.dataframe.iloc[i]["subject_id"]

        # Split recordings
        unique_recordings = list(recording_groups.keys())
        n_recordings = len(unique_recordings)

        if n_recordings < 3:
            logger.warning("Too few recordings for proper splits, using random split")
            return self._random_split(split_ratio)

        np.random.seed(42)
        shuffled_recordings = np.random.permutation(unique_recordings)

        n_train = max(1, int(split_ratio[0] * n_recordings))
        n_val = max(1, int(split_ratio[1] * n_recordings))

        train_recordings = shuffled_recordings[:n_train]
        val_recordings = shuffled_recordings[n_train : n_train + n_val]
        test_recordings = shuffled_recordings[n_train + n_val :]

        # Log detailed split information
        logger.info("\n   RECORDING-AWARE SPLIT DETAILS:")
        logger.info(f"   Total recordings: {n_recordings}")
        logger.info(f"   Split ratios: {split_ratio} -> Recordings: [{len(train_recordings)}, {len(val_recordings)}, {len(test_recordings)}]")
        logger.warning("   Same subject may appear in multiple splits!")

        # Collect graphs and log recording assignments
        train_graphs = []
        val_graphs = []
        test_graphs = []

        logger.info(f"\n   TRAINING SET RECORDINGS ({len(train_recordings)} recordings):")
        for rec_id in train_recordings:
            graphs = recording_groups[rec_id]
            train_graphs.extend(graphs)
            subject_id = recording_subject_info.get(rec_id, "unknown")
            logger.info(f"      Recording {rec_id}: {len(graphs)} graphs from Subject {subject_id}")

        logger.info(f"\n   VALIDATION SET RECORDINGS ({len(val_recordings)} recordings):")
        for rec_id in val_recordings:
            graphs = recording_groups[rec_id]
            val_graphs.extend(graphs)
            subject_id = recording_subject_info.get(rec_id, "unknown")
            logger.info(f"      Recording {rec_id}: {len(graphs)} graphs from Subject {subject_id}")

        logger.info(f"\n   TEST SET RECORDINGS ({len(test_recordings)} recordings):")
        for rec_id in test_recordings:
            graphs = recording_groups[rec_id]
            test_graphs.extend(graphs)
            subject_id = recording_subject_info.get(rec_id, "unknown")
            logger.info(f"      Recording {rec_id}: {len(graphs)} graphs from Subject {subject_id}")

        logger.info("\n   Recording-aware split complete:")
        logger.info(f"   Train: {len(train_graphs)} graphs from {len(train_recordings)} recordings")
        logger.info(f"   Val: {len(val_graphs)} graphs from {len(val_recordings)} recordings")
        logger.info(f"   Test: {len(test_graphs)} graphs from {len(test_recordings)} recordings")

        return train_graphs, val_graphs, test_graphs

    def _subject_aware_split(self, split_ratio: List[float]) -> Tuple[List[Data], List[Data], List[Data]]:
        """Split graphs by subjects to prevent subject-level data leakage."""
        logger.info("   Using subject-aware splits to prevent subject-level data leakage...")

        if self.mode == "whole_file":
            logger.warning("Subject-aware split not applicable for whole_file mode, using random split")
            return self._random_split(split_ratio)

        # Group graphs by subject_id
        subject_groups = {}
        subject_file_info = {}  # Track which files belong to each subject

        for i, graph in enumerate(self.graphs):
            if hasattr(graph, "subject_id"):
                subject_id = graph.subject_id
            else:
                # Fallback: use dataframe to determine subject
                if self.mode == "timestep" and i < len(self.dataframe):
                    subject_id = self.dataframe.iloc[i]["subject_id"]
                elif self.mode == "sequence" and hasattr(graph, "sequence_start"):
                    start_idx = graph.sequence_start
                    if start_idx < len(self.dataframe):
                        subject_id = self.dataframe.iloc[start_idx]["subject_id"]
                    else:
                        subject_id = "unknown"
                else:
                    subject_id = "unknown"

            if subject_id not in subject_groups:
                subject_groups[subject_id] = []
                subject_file_info[subject_id] = set()

            subject_groups[subject_id].append(graph)

            # Track file information for logging
            if hasattr(graph, "file_index") and hasattr(self, "metadata"):
                file_idx = graph.file_index
                if file_idx < len(self.metadata):
                    file_path = self.metadata[file_idx]["path"]
                    subject_file_info[subject_id].add(file_path)

        # Split subjects between train/val/test
        unique_subjects = list(subject_groups.keys())
        n_subjects = len(unique_subjects)

        if n_subjects < 3:
            logger.warning(f"Too few subjects ({n_subjects}) for proper splits, using random split")
            return self._random_split(split_ratio)

        np.random.seed(42)  # For reproducible splits
        shuffled_subjects = np.random.permutation(unique_subjects)

        n_train = max(1, int(split_ratio[0] * n_subjects))
        n_val = max(1, int(split_ratio[1] * n_subjects))

        train_subjects = shuffled_subjects[:n_train]
        val_subjects = shuffled_subjects[n_train : n_train + n_val]
        test_subjects = shuffled_subjects[n_train + n_val :]

        # Log detailed split information
        logger.info("\n   SUBJECT-AWARE SPLIT DETAILS:")
        logger.info(f"   Total subjects: {n_subjects}")
        logger.info(f"   Split ratios: {split_ratio} -> Subjects: [{len(train_subjects)}, {len(val_subjects)}, {len(test_subjects)}]")

        # Collect graphs and log subject assignments
        train_graphs = []
        val_graphs = []
        test_graphs = []

        logger.info(f"\n   TRAINING SET SUBJECTS ({len(train_subjects)} subjects):")
        for subject_id in train_subjects:
            graphs = subject_groups[subject_id]
            train_graphs.extend(graphs)
            files = subject_file_info.get(subject_id, {"unknown"})
            logger.info(f"      Subject {subject_id}: {len(graphs)} graphs from {len(files)} files")
            for file_path in sorted(files):
                logger.info(f"        {file_path}")

        logger.info(f"\n   VALIDATION SET SUBJECTS ({len(val_subjects)} subjects):")
        for subject_id in val_subjects:
            graphs = subject_groups[subject_id]
            val_graphs.extend(graphs)
            files = subject_file_info.get(subject_id, {"unknown"})
            logger.info(f"      Subject {subject_id}: {len(graphs)} graphs from {len(files)} files")
            for file_path in sorted(files):
                logger.info(f"        {file_path}")

        logger.info(f"\n   TEST SET SUBJECTS ({len(test_subjects)} subjects):")
        for subject_id in test_subjects:
            graphs = subject_groups[subject_id]
            test_graphs.extend(graphs)
            files = subject_file_info.get(subject_id, {"unknown"})
            logger.info(f"      Subject {subject_id}: {len(graphs)} graphs from {len(files)} files")
            for file_path in sorted(files):
                logger.info(f"        {file_path}")

        logger.info("\n   Subject-aware split complete:")
        logger.info(f"   Train: {len(train_graphs)} graphs from {len(train_subjects)} subjects")
        logger.info(f"   Val: {len(val_graphs)} graphs from {len(val_subjects)} subjects")
        logger.info(f"   Test: {len(test_graphs)} graphs from {len(test_subjects)} subjects")

        return train_graphs, val_graphs, test_graphs

    def _random_split(self, split_ratio: List[float]) -> Tuple[List[Data], List[Data], List[Data]]:
        """Random split of graphs."""
        logger.info("   Using random splits...")

        n_total = len(self.graphs)
        indices = np.random.permutation(n_total)

        n_train = int(split_ratio[0] * n_total)
        n_val = int(split_ratio[1] * n_total)

        train_idx = indices[:n_train]
        val_idx = indices[n_train : n_train + n_val]
        test_idx = indices[n_train + n_val :]

        train_graphs = [self.graphs[i] for i in train_idx]
        val_graphs = [self.graphs[i] for i in val_idx]
        test_graphs = [self.graphs[i] for i in test_idx]

        logger.info("\n   RANDOM SPLIT DETAILS:")
        logger.info(f"   Total graphs: {n_total}")
        logger.info(f"   Split ratios: {split_ratio} -> Graphs: [{len(train_graphs)}, {len(val_graphs)}, {len(test_graphs)}]")
        logger.warning("   Random splits may have subject-level data leakage!")

        return train_graphs, val_graphs, test_graphs

    def detect_conditions_from_motion(self, threshold_percentile: float = 75) -> np.ndarray:
        """Detect rest vs exercise conditions based on motion patterns."""
        if self.node_features is None:
            self.preprocess_data()

        logger.info("Detecting conditions from motion patterns...")

        # Calculate motion magnitude for each timestep
        motion_magnitudes = []

        for timestep in range(self.node_features.shape[0]):
            if timestep == 0:
                motion_magnitudes.append(0.0)
            else:
                curr_coords = self.node_features[timestep]
                prev_coords = self.node_features[timestep - 1]
                movements = np.sqrt(np.sum((curr_coords - prev_coords) ** 2, axis=1))
                avg_movement = np.mean(movements)
                motion_magnitudes.append(avg_movement)

        motion_magnitudes = np.array(motion_magnitudes)

        # Smooth with sliding window
        window_size = min(30, len(motion_magnitudes) // 10)
        if window_size > 1:
            smoothed_motion = np.convolve(motion_magnitudes, np.ones(window_size) / window_size, mode="same")
        else:
            smoothed_motion = motion_magnitudes

        # Create binary labels
        threshold = np.percentile(smoothed_motion, threshold_percentile)
        conditions = (smoothed_motion > threshold).astype(int)

        logger.info(f"   Motion threshold: {threshold:.3f}")
        logger.info(f"   Rest periods: {np.sum(conditions == 0)} timesteps ({100 * np.mean(conditions == 0):.1f}%)")
        logger.info(f"   Exercise periods: {np.sum(conditions == 1)} timesteps ({100 * np.mean(conditions == 1):.1f}%)")

        return conditions

    def visualize_graphs(self, num_graphs: int = 3, save_path: str = None) -> plt.Figure:
        """Visualize sample graphs from the dataset."""
        if self.graphs is None:
            self.create_graphs()

        num_graphs = min(num_graphs, len(self.graphs))
        fig, axes = plt.subplots(1, num_graphs, figsize=(6 * num_graphs, 6))
        if num_graphs == 1:
            axes = [axes]

        fig.suptitle(f"Sample Graphs ({self.mode} mode)", fontsize=16, fontweight="bold")

        for i in range(num_graphs):
            graph = self.graphs[i]
            ax = axes[i]

            # Extract coordinates for visualization
            coords = graph.x.numpy()

            if self.mode == "timestep":
                # For timestep mode, we have [num_points, 3]
                if coords.shape[1] >= 3:
                    x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]
                    scatter = ax.scatter(x, y, c=z, cmap="viridis", alpha=0.7)
                    ax.set_title(f"Graph {i} (Timestep mode)")
                    plt.colorbar(scatter, ax=ax, label="Z coordinate")

            elif self.mode in ["whole_file", "sequence"]:
                # For other modes, we have flattened nodes
                # Color by node index for visualization
                if len(coords) > 0:
                    node_indices = np.arange(len(coords))
                    if coords.shape[1] >= 2:
                        scatter = ax.scatter(
                            coords[:, 0],
                            coords[:, 1],
                            c=node_indices,
                            cmap="tab10",
                            alpha=0.7,
                        )
                        ax.set_title(f"Graph {i} ({self.mode} mode)")
                        plt.colorbar(scatter, ax=ax, label="Node index")

            ax.set_xlabel("X coordinate")
            ax.set_ylabel("Y coordinate")
            ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info(f"Graph visualization saved to: {save_path}")

        return fig

    @classmethod
    def from_multiple_files(
        cls,
        file_paths: List[str],
        person_ids: List[str] = None,
        conditions: List[str] = None,
        **kwargs,
    ):
        """Create dataset from multiple EMT files."""
        logger.info(f"Loading {len(file_paths)} files...")

        combined_data = []

        for i, file_path in enumerate(file_paths):
            try:
                logger.info(f"   Loading file {i + 1}/{len(file_paths)}: {Path(file_path).name}")
                df = pd.DataFrame.emt3d.from_emt(file_path)

                # Add metadata
                df["recording_id"] = f"recording_{i}"
                df["filename"] = str(Path(file_path).name)

                if person_ids and i < len(person_ids):
                    df["person_id"] = person_ids[i]

                if conditions and i < len(conditions):
                    df["condition"] = conditions[i]

                # Auto-detect time condition from filename
                filename_lower = str(file_path).lower()
                if "rest" in filename_lower:
                    df["time_condition"] = "rest"
                elif any(x in filename_lower for x in ["min", "exercise"]):
                    df["time_condition"] = "exercise"
                else:
                    df["time_condition"] = "unknown"

                combined_data.append(df)

            except Exception as e:
                logger.warning(f"Error loading {file_path}: {e}")

        if not combined_data:
            raise ValueError("No files could be loaded successfully")

        combined_df = pd.concat(combined_data, ignore_index=True)
        logger.info(f"Combined dataset: {combined_df.shape[0]} timesteps, {combined_df['recording_id'].nunique()} recordings")

        return cls(combined_df, **kwargs)

    @property
    def node_connections(self):
        """Get the node connections as a list of edges (vest topology).

        If we have the expected 89 nodes, use the full vest topology.
        Otherwise, create a simplified grid-like topology.
        """
        num_nodes = self.node_features.shape[1]

        if num_nodes == 89:
            # Full vest topology
            edge_connections = [
                [1, 2],
                [1, 8],
                [1, 46],
                [1, 47],
                [2, 3],
                [2, 9],
                [2, 13],
                [2, 46],
                [3, 4],
                [3, 9],
                [3, 45],
                [4, 5],
                [4, 10],
                [4, 45],
                [5, 6],
                [5, 11],
                [5, 45],
                [6, 7],
                [6, 11],
                [6, 44],
                [7, 12],
                [7, 14],
                [7, 43],
                [7, 44],
                [8, 13],
                [8, 15],
                [8, 47],
                [8, 52],
                [9, 10],
                [9, 16],
                [9, 17],
                [10, 17],
                [11, 10],
                [11, 14],
                [11, 17],
                [11, 18],
                [12, 19],
                [12, 43],
                [12, 48],
                [13, 9],
                [13, 15],
                [13, 16],
                [14, 18],
                [14, 19],
                [15, 16],
                [15, 20],
                [15, 61],
                [16, 17],
                [16, 21],
                [17, 18],
                [17, 22],
                [18, 19],
                [18, 23],
                [19, 24],
                [19, 55],
                [20, 21],
                [20, 28],
                [20, 68],
                [20, 75],
                [21, 22],
                [21, 25],
                [22, 23],
                [22, 26],
                [23, 24],
                [23, 27],
                [24, 32],
                [24, 62],
                [24, 69],
                [25, 28],
                [25, 26],
                [25, 29],
                [26, 30],
                [26, 27],
                [27, 31],
                [27, 32],
                [28, 29],
                [28, 33],
                [28, 82],
                [29, 30],
                [29, 34],
                [30, 31],
                [30, 35],
                [31, 32],
                [31, 36],
                [32, 37],
                [32, 76],
                [33, 38],
                [33, 34],
                [33, 89],
                [34, 35],
                [34, 39],
                [35, 36],
                [36, 37],
                [37, 83],
                [38, 39],
                [40, 35],
                [40, 39],
                [40, 41],
                [41, 36],
                [41, 42],
                [42, 37],
                [43, 44],
                [43, 48],
                [44, 45],
                [44, 49],
                [45, 46],
                [45, 50],
                [46, 47],
                [46, 51],
                [47, 52],
                [48, 49],
                [48, 53],
                [48, 55],
                [49, 53],
                [49, 50],
                [50, 51],
                [50, 58],
                [51, 52],
                [51, 54],
                [52, 54],
                [52, 61],
                [53, 55],
                [53, 56],
                [54, 60],
                [54, 61],
                [55, 56],
                [55, 62],
                [56, 57],
                [56, 63],
                [57, 58],
                [57, 64],
                [58, 59],
                [58, 65],
                [59, 60],
                [59, 66],
                [60, 61],
                [60, 67],
                [61, 68],
                [62, 63],
                [62, 69],
                [63, 64],
                [63, 70],
                [64, 65],
                [64, 71],
                [65, 66],
                [65, 72],
                [66, 67],
                [66, 73],
                [67, 68],
                [67, 74],
                [68, 75],
                [69, 70],
                [69, 76],
                [70, 71],
                [70, 77],
                [71, 72],
                [71, 78],
                [72, 73],
                [72, 79],
                [73, 74],
                [73, 80],
                [74, 75],
                [74, 81],
                [75, 82],
                [76, 77],
                [76, 83],
                [77, 78],
                [77, 84],
                [78, 79],
                [78, 85],
                [79, 80],
                [79, 86],
                [80, 81],
                [80, 87],
                [81, 82],
                [81, 88],
                [82, 89],
                [83, 84],
                [84, 85],
                [85, 86],
                [86, 87],
                [87, 88],
                [88, 89],
            ]
        else:
            # Simplified grid-like topology for different number of nodes
            edge_connections = []
            for i in range(1, num_nodes + 1):
                # Connect to immediate neighbors in a grid-like pattern
                if i < num_nodes:  # Connect to next node
                    edge_connections.append([i, i + 1])
                if i <= num_nodes - 2:  # Connect to node two positions away
                    edge_connections.append([i, i + 2])

        return edge_connections

    @property
    def num_node_features(self):
        """Get the number of node features (should be 3 for x,y,z)."""
        if self.node_features is not None:
            return self.node_features.shape[2]
        return 3

    @property
    def num_graphs(self):
        """Get the number of graphs in the dataset."""
        return len(self.graphs) if self.graphs else 0

    def __len__(self):
        """Return the number of graphs."""
        return self.num_graphs

    def __getitem__(self, idx):
        """Get a specific graph by index."""
        if self.graphs is None:
            self.create_graphs()
        return self.graphs[idx]

    def save_dataset(self, path: str):
        """Save the processed dataset."""
        data_to_save = {
            "dataframe": self.original_dataframe,
            "node_features": self.node_features,
            "graphs": self.graphs,
            "mode": self.mode,
            "sequence_length": self.sequence_length,
            "scaler": self.scaler,
            "scaler_type": self.scaler_type,
            "metadata_cols": self.metadata_cols,
        }
        torch.save(data_to_save, path)
        logger.info(f"Dataset saved to: {path}")

    @classmethod
    def load_dataset(cls, path: str):
        """Load a saved dataset."""
        data = torch.load(path)

        dataset = cls(
            dataframe=data["dataframe"],
            mode=data["mode"],
            sequence_length=data.get("sequence_length", 10),
            scaler_type=data.get("scaler_type", "standard"),
            validate_data=False,
        )

        dataset.node_features = data["node_features"]
        dataset.graphs = data["graphs"]
        dataset.scaler = data["scaler"]
        dataset.metadata_cols = data.get("metadata_cols", [])

        logger.info(f"Dataset loaded from: {path}")
        return dataset


# %%
# Example usage (uncommented for testing):
#
# # Single file
# df = pd.DataFrame.emt3d.from_emt('data/1201/1201_tightbra/15.min/3D Point Tracks.emt')
# dataset = GNNVestDataset(df, mode='timestep')
# dataset.visualize_data_overview()
# graphs = dataset.create_graphs()
# train_loader, val_loader, test_loader = dataset.create_dataloaders()
#
# # Multiple files
# file_paths = ['file1.emt', 'file2.emt', 'file3.emt']
# dataset = GNNVestDataset.from_multiple_files(file_paths, mode='sequence', sequence_length=5)
# dataset.visualize_graphs()
#
# # Condition detection
# conditions = dataset.detect_conditions_from_motion()
# print(f"Detected conditions: {np.bincount(conditions)}")

# %%
# Visualization example:
# fig = dataset.visualize_data_overview()
# fig.show()
#
# fig = dataset.visualize_graphs(num_graphs=3)
# fig.show()
