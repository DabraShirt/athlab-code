from typing import Any, Dict, Set

from torch_geometric.loader import DataLoader

from ml.utils.logging_utils import get_logger

logger = get_logger("athlab.validation")


def validate_subject_separation(
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Validate that no subjects appear in multiple splits.

    Parameters
    ----------
    train_loader : DataLoader
        Training data loader
    val_loader : DataLoader
        Validation data loader
    test_loader : DataLoader
        Test data loader
    verbose : bool, default=True
        Whether to print detailed validation results

    Returns
    -------
    dict
        Validation results including whether splits are valid and details
    """

    def extract_subjects_from_loader(loader: DataLoader, split_name: str) -> Set[str]:
        """Extract unique subject IDs from a data loader."""
        subjects = set()
        for batch in loader:
            for graph in batch:
                if hasattr(graph, "subject_id"):
                    subjects.add(graph.subject_id)
                elif hasattr(graph, "graph_attrs") and "subject_id" in graph.graph_attrs:
                    subjects.add(graph.graph_attrs["subject_id"])

        if verbose:
            logger.info(f"{split_name} set subjects: {sorted(subjects)}")
            logger.info(f"Total unique subjects: {len(subjects)}")

        return subjects

    # Extract subjects from each split
    train_subjects = extract_subjects_from_loader(train_loader, "TRAINING")
    val_subjects = extract_subjects_from_loader(val_loader, "VALIDATION")
    test_subjects = extract_subjects_from_loader(test_loader, "TEST")

    # Check for overlaps
    train_val_overlap = train_subjects.intersection(val_subjects)
    train_test_overlap = train_subjects.intersection(test_subjects)
    val_test_overlap = val_subjects.intersection(test_subjects)

    # Calculate results
    all_overlaps = train_val_overlap.union(train_test_overlap).union(val_test_overlap)
    is_valid = len(all_overlaps) == 0

    results = {
        "is_valid": is_valid,
        "train_subjects": train_subjects,
        "val_subjects": val_subjects,
        "test_subjects": test_subjects,
        "train_val_overlap": train_val_overlap,
        "train_test_overlap": train_test_overlap,
        "val_test_overlap": val_test_overlap,
        "total_subjects": len(train_subjects.union(val_subjects).union(test_subjects)),
        "train_count": len(train_subjects),
        "val_count": len(val_subjects),
        "test_count": len(test_subjects),
    }

    if verbose:
        logger.info("=" * 60)
        logger.info("SUBJECT SEPARATION VALIDATION RESULTS")
        logger.info("=" * 60)

        if is_valid:
            logger.info("VALIDATION PASSED: No subject overlap between splits")
            logger.info("Model will be tested on completely new subjects")
            logger.info("Results will properly reflect generalization capability")
        else:
            logger.warning("VALIDATION FAILED: Subject overlap detected")
            logger.warning("Model may be learning subject-specific patterns")
            logger.warning("Results may not reflect true generalization capability")

            if train_val_overlap:
                logger.warning(f"Train-Val overlap: {sorted(train_val_overlap)}")
            if train_test_overlap:
                logger.warning(f"Train-Test overlap: {sorted(train_test_overlap)}")
            if val_test_overlap:
                logger.warning(f"Val-Test overlap: {sorted(val_test_overlap)}")

        logger.info("SPLIT STATISTICS:")
        logger.info(f"Total subjects across all splits: {results['total_subjects']}")
        logger.info(f"Training subjects: {results['train_count']}")
        logger.info(f"Validation subjects: {results['val_count']}")
        logger.info(f"Test subjects: {results['test_count']}")

        # Calculate split ratios
        total = results["train_count"] + results["val_count"] + results["test_count"]
        if total > 0:
            train_ratio = results["train_count"] / total
            val_ratio = results["val_count"] / total
            test_ratio = results["test_count"] / total
            logger.info(f"Subject split ratios: Train={train_ratio:.2%}, Val={val_ratio:.2%}, Test={test_ratio:.2%}")

        logger.info("=" * 60)

    return results


def validate_temporal_separation(
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    max_batches_to_check: int = 10,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Validate that batches don't contain temporally adjacent samples (for timestep mode).

    Parameters
    ----------
    train_loader : DataLoader
        Training data loader
    val_loader : DataLoader
        Validation data loader
    test_loader : DataLoader
        Test data loader
    max_batches_to_check : int, default=10
        Maximum number of batches to check per split
    verbose : bool, default=True
        Whether to print detailed validation results

    Returns
    -------
    dict
        Temporal validation results
    """

    def check_temporal_adjacency_in_loader(loader: DataLoader, split_name: str) -> Dict[str, Any]:
        """Check for temporal adjacency within batches."""
        batch_violations = []
        total_batches_checked = 0

        for batch_idx, batch in enumerate(loader):
            if batch_idx >= max_batches_to_check:
                break

            total_batches_checked += 1

            # Extract timestamps or frame indices if available
            timestamps = []
            recording_ids = []

            for graph in batch:
                # Try to extract temporal information
                timestamp = None
                recording_id = None

                if hasattr(graph, "timestamp"):
                    timestamp = graph.timestamp
                elif hasattr(graph, "frame_idx"):
                    timestamp = graph.frame_idx

                if hasattr(graph, "recording_id"):
                    recording_id = graph.recording_id

                if timestamp is not None:
                    timestamps.append(timestamp)
                if recording_id is not None:
                    recording_ids.append(recording_id)

            # Check for violations within this batch
            if len(timestamps) > 1 and len(set(recording_ids)) == 1:
                # Same recording, check if timestamps are adjacent
                timestamps.sort()
                for i in range(len(timestamps) - 1):
                    if abs(timestamps[i + 1] - timestamps[i]) == 1:
                        batch_violations.append(
                            {
                                "batch_idx": batch_idx,
                                "adjacent_frames": (timestamps[i], timestamps[i + 1]),
                                "recording_id": recording_ids[0],
                            }
                        )
                        break

        return {
            "violations": batch_violations,
            "batches_checked": total_batches_checked,
            "violation_count": len(batch_violations),
        }

    # Check each split
    train_results = check_temporal_adjacency_in_loader(train_loader, "TRAINING")
    val_results = check_temporal_adjacency_in_loader(val_loader, "VALIDATION")
    test_results = check_temporal_adjacency_in_loader(test_loader, "TEST")

    total_violations = train_results["violation_count"] + val_results["violation_count"] + test_results["violation_count"]

    is_valid = total_violations == 0

    results = {
        "is_valid": is_valid,
        "train_results": train_results,
        "val_results": val_results,
        "test_results": test_results,
        "total_violations": total_violations,
    }

    if verbose:
        logger.info("=" * 60)
        logger.info("TEMPORAL LEAKAGE VALIDATION RESULTS")
        logger.info("=" * 60)

        if is_valid:
            logger.info("TEMPORAL VALIDATION PASSED: No adjacent timesteps in same batch")
            logger.info("No temporal leakage detected")
        else:
            logger.warning("TEMPORAL VALIDATION FAILED: Adjacent timesteps found in batches")
            logger.warning("Model may be learning from temporal proximity")
            logger.warning("Consider using batch_size=1 for timestep mode")

            for split_name, split_results in [
                ("Training", train_results),
                ("Validation", val_results),
                ("Test", test_results),
            ]:
                if split_results["violation_count"] > 0:
                    logger.warning(f"{split_name} violations: {split_results['violation_count']}")
                    for violation in split_results["violations"][:3]:  # Show first 3
                        logger.warning(f"Batch {violation['batch_idx']}: frames {violation['adjacent_frames']} from {violation['recording_id']}")

        logger.info("=" * 60)

    return results


def run_comprehensive_validation(
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    check_temporal: bool = True,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Run comprehensive data leakage validation.

    Parameters
    ----------
    train_loader, val_loader, test_loader : DataLoader
        Data loaders for each split
    check_temporal : bool, default=True
        Whether to check for temporal leakage
    verbose : bool, default=True
        Whether to print detailed results

    Returns
    -------
    dict
        Complete validation results
    """
    if verbose:
        logger.info("RUNNING COMPREHENSIVE DATA LEAKAGE VALIDATION")
        logger.info("=" * 80)

    # Subject separation validation
    subject_results = validate_subject_separation(train_loader, val_loader, test_loader, verbose)

    # Temporal leakage validation
    temporal_results = None
    if check_temporal:
        temporal_results = validate_temporal_separation(train_loader, val_loader, test_loader, verbose=verbose)

    # Overall assessment
    overall_valid = subject_results["is_valid"]
    if check_temporal and temporal_results:
        overall_valid = overall_valid and temporal_results["is_valid"]

    results = {
        "overall_valid": overall_valid,
        "subject_validation": subject_results,
        "temporal_validation": temporal_results,
    }

    if verbose:
        logger.info("=" * 80)
        logger.info("OVERALL VALIDATION SUMMARY")
        logger.info("=" * 80)

        if overall_valid:
            logger.info("ALL VALIDATIONS PASSED")
            logger.info("No data leakage detected")
            logger.info("Results will reflect true model performance")
        else:
            logger.warning("VALIDATION ISSUES DETECTED")
            logger.warning("Data leakage may compromise results")
            logger.warning("Consider adjusting split strategy or batch size")

        logger.info("=" * 80)

    return results


if __name__ == "__main__":
    # Example usage
    logger.info("Data Leakage Validation Utilities")
    logger.info("Import this module and use the validation functions with your data loaders.")
    logger.info("Example:")
    logger.info("from ml.utils.data_leakage_validator import run_comprehensive_validation")
    logger.info("results = run_comprehensive_validation(train_loader, val_loader, test_loader)")
