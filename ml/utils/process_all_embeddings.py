import sys
import time
from pathlib import Path

from ml.utils.embedding_aggregation import EmbeddingAggregator

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))


def process_all_embeddings(input_dir=None, output_dir=None):
    """Process all embedding files in the specified directory."""

    print("PROCESSING ALL EMBEDDING FILES")
    print("=" * 60)

    # Create aggregator with less verbose output for batch processing
    aggregator = EmbeddingAggregator(verbose=False)

    # Input and output directories (relative to project root)
    project_root = Path(__file__).parent.parent.parent

    # Allow custom directories or use defaults
    if input_dir is None:
        input_dir = project_root / "ml/ml_data/embeddings"
    else:
        input_dir = Path(input_dir)
        if not input_dir.is_absolute():
            input_dir = project_root / input_dir

    if output_dir is None:
        output_dir = project_root / "ml/ml_data/embeddings_agg"
    else:
        output_dir = Path(output_dir)
        if not output_dir.is_absolute():
            output_dir = project_root / output_dir

    pattern = "embeddings_*.npy"

    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Pattern: {pattern}")

    # Start processing
    start_time = time.time()

    results = aggregator.process_embeddings_directory(input_dir=str(input_dir), output_dir=str(output_dir), pattern=pattern)

    end_time = time.time()
    processing_time = end_time - start_time

    # Detailed summary
    print("\nFINAL PROCESSING SUMMARY")
    print("=" * 60)
    print(f"   Total files found: {len(results['files'])}")
    print(f"   Successfully processed: {results['processed']}")
    print(f"   Failed: {results['failed']}")
    print(f"   Total processing time: {processing_time:.1f} seconds")

    if results["processed"] > 0:
        print(f"   Average time per file: {processing_time / results['processed']:.2f} seconds")
        print(f"   Output files saved to: {output_dir}")

        # Quick verification of output files
        print("\nQUICK VERIFICATION")
        print("-" * 30)

        import glob

        import numpy as np

        output_files = glob.glob(f"{output_dir}/*.npy")
        if output_files:
            sample_file = output_files[0]
            sample_data = np.load(sample_file)
            print(f"   Sample file: {Path(sample_file).name}")
            print(f"   Output shape: {sample_data.shape}")
            print("   Target dimensions: 512 CORRECT" if sample_data.shape[0] == 512 else f"   ERROR: Wrong dimensions: {sample_data.shape[0]}")
            print(f"   Data type: {sample_data.dtype}")

            # Count total zero features across all files
            total_zeros = 0
            total_features = 0

            for output_file in output_files[:5]:  # Check first 5 files
                data = np.load(output_file)
                total_zeros += np.sum(data == 0)
                total_features += len(data)

            zero_percentage = 100 * total_zeros / total_features if total_features > 0 else 0
            print(f"   Zero features (sample): {total_zeros}/{total_features} ({zero_percentage:.1f}%)")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Process embedding files for aggregation")
    parser.add_argument(
        "--input-dir",
        type=str,
        help="Input directory containing embeddings (default: ml/ml_data/embeddings)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Output directory for aggregated embeddings (default: ml/ml_data/embeddings_agg)",
    )

    args = parser.parse_args()

    results = process_all_embeddings(input_dir=args.input_dir, output_dir=args.output_dir)

    if results["processed"] > 0:
        print(f"\nSUCCESS! Processed {results['processed']} embedding files.")
        output_path = args.output_dir if args.output_dir else "ml/ml_data/embeddings_agg/"
        print(f"All aggregated files saved to: {output_path}")
    else:
        print("\nNo files were processed successfully.")
