import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import yaml

from ml.datasets.gnn_vest_dataset import GNNVestDataset
from ml.utils.emt_pandas import Emt3DAccessor, EmtAccessor  # noqa: F401
from ml.utils.logging_utils import get_logger

logger = get_logger("athlab.data_loader")


def load_data_config(config_path: str = "configs/data_splits.yaml") -> Dict:
    """Load data split configuration."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def validate_data_paths(data_sources: List[Dict]) -> List[Dict]:
    """Validate that all data paths exist."""
    valid_sources = []
    missing_files = []

    for source in data_sources:
        path = Path(source["path"])
        if path.exists():
            valid_sources.append(source)
        else:
            missing_files.append(str(path))
            logging.warning(f"Missing file: {path}")

    if missing_files:
        logging.warning(f"Found {len(missing_files)} missing files out of {len(data_sources)} total")
        logging.info(f"Using {len(valid_sources)} valid files for training")

    return valid_sources


def load_multiple_emt_files(data_sources: List[Dict], max_files: Optional[int] = None) -> Tuple[pd.DataFrame, List[Dict]]:
    """
    Load and combine multiple EMT files into a single DataFrame.

    Parameters
    ----------
    data_sources : list of dict
        List of data source configurations
    max_files : int, optional
        Maximum number of files to load (for testing)

    Returns
    -------
    tuple
        (combined_dataframe, metadata_list)
    """
    logging.info(f"Loading {len(data_sources)} EMT files...")

    all_dataframes = []
    metadata_list = []

    # Limit files if specified (useful for testing)
    if max_files:
        data_sources = data_sources[:max_files]
        logging.info(f"Limited to first {max_files} files for testing")

    for i, source in enumerate(data_sources):
        try:
            # Load the EMT file
            df = pd.DataFrame.emt3d.from_emt(source["path"])
            df = df[:100]

            # Add metadata columns to identify source
            df["subject_id"] = source["subject"]
            df["condition"] = source["condition"]
            df["activity"] = source["activity"]
            df["file_index"] = i
            df["recording_id"] = f"recording_{i}"  # Add unique recording ID

            all_dataframes.append(df)
            metadata_list.append(source)

            logging.info(f"Loaded {source['subject']}/{source['condition']}/{source['activity']}: {df.shape[0]} frames")

        except Exception as e:
            logging.error(f"Failed to load {source['path']}: {e}")
            continue

    if not all_dataframes:
        raise ValueError("No valid EMT files could be loaded!")

    # Combine all dataframes
    combined_df = pd.concat(all_dataframes, ignore_index=True)

    logging.info(f"Combined {len(all_dataframes)} files into dataset with {combined_df.shape[0]} total frames")
    logging.info(f"Subjects: {combined_df['subject_id'].unique()}")
    logging.info(f"Conditions: {combined_df['condition'].unique()}")
    logging.info(f"Activities: {combined_df['activity'].unique()}")

    return combined_df, metadata_list


def create_stratified_dataset(
    combined_df: pd.DataFrame,
    metadata_list: List[Dict],
    config: Dict,
    dataset_mode: str = "timestep",
    sequence_length: int = 10,
    sequence_step: int = 1,
) -> GNNVestDataset:
    """
    Create a GNNVestDataset with proper stratification.

    Parameters
    ----------
    combined_df : pd.DataFrame
        Combined dataframe from multiple sources
    metadata_list : list of dict
        Metadata for each source file
    config : dict
        Configuration dictionary
    dataset_mode : str, default='timestep'
        Dataset mode ('timestep', 'sequence', 'whole_file')
    sequence_length : int, default=10
        Sequence length for sequence mode
    sequence_step : int, default=1
        Sequence step for sequence mode. How many timesteps to skip between sequences.

    Returns
    -------
    GNNVestDataset
        Dataset ready for training
    """
    logging.info(f"Creating stratified GNN dataset in '{dataset_mode}' mode...")

    # Get split configuration
    splits = config.get("training_splits", {})
    test_size = splits.get("test_size", 0.15)
    val_size = splits.get("val_size", 0.15)
    stratify_by = splits.get("stratify_by", "subject")

    # Create the dataset with specified mode
    dataset = GNNVestDataset(
        dataframe=combined_df,
        mode=dataset_mode,
        sequence_length=sequence_length if dataset_mode == "sequence" else None,
        sequence_step=sequence_step if dataset_mode == "sequence" else 1,
        test_size=test_size,
        val_size=val_size,
        scaler_type="standard",
        validate_data=True,
    )

    # Store stratify_by configuration for later use
    dataset.stratify_by = stratify_by
    logging.info(f"   - Stratification strategy: {stratify_by}")

    # Store metadata in dataset for later use
    dataset.metadata = metadata_list
    dataset.file_info = {
        "subjects": sorted(combined_df["subject_id"].unique()),
        "conditions": sorted(combined_df["condition"].unique()),
        "activities": sorted(combined_df["activity"].unique()),
        "total_files": len(metadata_list),
        "total_frames": len(combined_df),
    }

    logging.info("Created dataset with {dataset.file_info['total_frames']} frames in '{dataset_mode}' mode")
    if dataset_mode == "sequence":
        logging.info(f"   - Sequence length: {sequence_length}")
    elif dataset_mode == "whole_file":
        logging.info("   - Using entire files as single graphs")
    elif dataset_mode == "timestep":
        logging.info("   - Each timestep as separate graph")

    logging.info("DATASET SUMMARY:")
    logging.info(f"   - Total subjects: {len(dataset.file_info['subjects'])}")
    logging.info(f"   - Subjects: {dataset.file_info['subjects']}")
    logging.info(f"   - Conditions: {dataset.file_info['conditions']}")
    logging.info(f"   - Activities: {dataset.file_info['activities']}")
    logging.info(f"   - Split strategy: {stratify_by}-aware")
    logging.info(f"   - Split ratios: Train={1.0 - test_size - val_size:.1%}, Val={val_size:.1%}, Test={test_size:.1%}")

    # Configure splitting strategy based on stratify_by setting
    if stratify_by == "subject":
        dataset.use_subject_aware_splits = True
        dataset.use_recording_aware_splits = False
        logging.info("   SUBJECT-AWARE SPLITTING: Subjects will be separated across train/val/test")
        logging.info("   NO SUBJECT-LEVEL DATA LEAKAGE: Model will be tested on completely new subjects")
    elif stratify_by == "recording":
        dataset.use_subject_aware_splits = False
        dataset.use_recording_aware_splits = True
        logging.info("   RECORDING-AWARE SPLITTING: Same subject may appear in multiple splits")
        logging.info("   POTENTIAL SUBJECT-LEVEL DATA LEAKAGE: Results may not generalize to new subjects")
    else:
        dataset.use_subject_aware_splits = False
        dataset.use_recording_aware_splits = False
        logging.warning("   RANDOM SPLITTING: No data leakage protection! This should only be used for debugging.")

    return dataset


def prepare_training_data(
    config_path: str = "configs/data_splits.yaml",
    max_files: Optional[int] = None,
    dataset_mode: str = "timestep",
    sequence_length: int = 10,
    sequence_step: int = 1,
) -> Tuple[GNNVestDataset, List, int]:
    """
    Prepare training data from multiple sources.

    Parameters
    ----------
    config_path : str
        Path to data configuration file
    max_files : int, optional
        Maximum number of files to load (for testing)
    dataset_mode : str, default='timestep'
        Dataset mode ('timestep', 'sequence', 'whole_file')
    sequence_length : int, default=10
        Sequence length for sequence mode
    sequence_step : int, default=1
        Sequence step for sequence mode.

    Returns
    -------
    tuple
        (dataset, graphs, num_nodes)
    """
    # Load configuration
    config = load_data_config(config_path)

    # Get training data sources
    training_sources = config["training_data"]["sources"]

    # If sources is a string 'ALL', scan the data directory for all 3D Point Tracks.emt files
    if isinstance(training_sources, str) and training_sources.strip().upper() == "ALL":
        logging.info("Scanning data directory for all 3D Point Tracks.emt files...")
        data_dir = Path("data")
        emt_files = list(data_dir.glob("**/3D Point Tracks.emt"))
        training_sources = []
        for file_path in emt_files:
            # Try to extract subject, condition, activity from the path
            parts = file_path.parts
            # Heuristic: subject is after 'data', condition next, activity next
            try:
                data_idx = parts.index("data")
                subject = parts[data_idx + 1] if len(parts) > data_idx + 1 else "unknown"
                condition = parts[data_idx + 2] if len(parts) > data_idx + 2 else "unknown"
                activity = parts[data_idx + 3] if len(parts) > data_idx + 3 else "unknown"
            except Exception:
                subject = "unknown"
                condition = "unknown"
                activity = "unknown"
            training_sources.append(
                {
                    "path": str(file_path).replace("\\", "/"),
                    "subject": subject,
                    "condition": condition,
                    "activity": activity,
                }
            )
        logging.info(f"Found {len(training_sources)} EMT files in data directory.")

    # Validate paths
    valid_sources = validate_data_paths(training_sources)

    if not valid_sources:
        raise ValueError("No valid training data files found!")

    # Load and combine data
    combined_df, metadata_list = load_multiple_emt_files(valid_sources, max_files)

    # Create dataset with specified mode
    dataset = create_stratified_dataset(combined_df, metadata_list, config, dataset_mode, sequence_length, sequence_step)

    # Preprocess and create graphs
    dataset.preprocess_data(apply_scaling=True)
    graphs = dataset.create_graphs()  # Creates graphs based on specified mode

    # Get number of nodes
    num_nodes = graphs[0].x.shape[0] if graphs else 89

    logging.info(f"Prepared training data: {len(graphs)} graphs with {num_nodes} nodes each")
    logging.info(f"   - Mode: {dataset_mode}")
    if dataset_mode == "sequence":
        logging.info(f"   - Sequence length: {sequence_length}")

    # Add temporal leakage warning for timestep mode with batching
    if dataset_mode == "timestep":
        logging.warning("CRITICAL TEMPORAL LEAKAGE WARNING:")
        logging.warning("   Timestep mode with batch_size > 1 WILL cause temporal leakage!")
        logging.warning("   Adjacent timesteps from same recording will appear in the same batch.")
        logging.warning("   SOLUTION: Use batch_size=1 for timestep mode or switch to 'sequence' mode.")
        logging.warning("   Current config will force batch_size=1 for timestep mode.")
        dataset.force_batch_size_1_for_timestep = True

    return dataset, graphs, num_nodes


def prepare_exploration_data(config_path: str = "configs/data_splits.yaml", max_files: Optional[int] = None) -> Dict[str, Tuple[pd.DataFrame, List[Dict]]]:
    """
    Prepare exploration data from multiple sources.

    Parameters
    ----------
    config_path : str
        Path to data configuration file
    max_files : int, optional
        Maximum number of files to load (for testing)

    Returns
    -------
    dict
        Dictionary mapping data source types to (dataframe, metadata) tuples
    """
    # Load configuration
    config = load_data_config(config_path)

    # Get exploration data sources
    exploration_sources = config["exploration_data"]["sources"]

    # Validate paths
    valid_sources = validate_data_paths(exploration_sources)

    if not valid_sources:
        raise ValueError("No valid exploration data files found!")

    # Load and combine exploration data
    combined_df, metadata_list = load_multiple_emt_files(valid_sources, max_files)

    # Group by condition for separate analysis
    exploration_data = {}

    for condition in combined_df["condition"].unique():
        condition_df = combined_df[combined_df["condition"] == condition].copy()
        condition_metadata = [m for m in metadata_list if m["condition"] == condition]

        exploration_data[condition] = (condition_df, condition_metadata)
        logging.info(f"Prepared {condition} exploration data: {len(condition_df)} frames")

    return exploration_data


if __name__ == "__main__":
    # Test the data loading functions
    logging.basicConfig(level=logging.INFO)

    logger.info("Testing multi-dataset data loader with different modes...")

    try:
        # Test different dataset modes with limited files for speed
        modes_to_test = [("timestep", None), ("sequence", 5), ("whole_file", None)]

        for mode, seq_len in modes_to_test:
            logger.info(f"Testing {mode} mode...")

            dataset, graphs, num_nodes = prepare_training_data(
                max_files=3,  # Limited for testing
                dataset_mode=mode,
                sequence_length=seq_len,
            )

            logger.info(f"{mode.capitalize()} mode: {len(graphs)} graphs, {num_nodes} nodes per graph")

            # Show a sample graph
            if graphs:
                sample_graph = graphs[0]
                logger.info(f"Sample graph: {sample_graph.x.shape[0]} nodes, {sample_graph.edge_index.shape[1]} edges")
                if hasattr(sample_graph, "mode"):
                    logger.info(f"Graph mode: {sample_graph.mode}")

        logger.info("Testing exploration data...")
        exploration_data = prepare_exploration_data(max_files=2)
        logger.info(f"Exploration data prepared: {len(exploration_data)} conditions")

        logger.info("Multi-dataset loading test successful!")

    except Exception as e:
        logger.error(f"Test failed: {e}")
        import traceback

        traceback.print_exc()
