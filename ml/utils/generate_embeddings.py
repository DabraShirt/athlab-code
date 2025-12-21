import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ml.datasets.gnn_vest_dataset import GNNVestDataset
from ml.models.GNN_autoencoder import GNNAutoencoder
from ml.utils.emt_pandas import Emt3DAccessor, EmtAccessor  # noqa: F401
from ml.utils.logging_utils import get_logger

# Add project root to path
project_root = Path(__file__).parent.parent.parent  # Go up from ml/utils/ to project root
sys.path.insert(0, str(project_root))

# Configuration
CONFIG = {
    "runners": {
        "ID_1": {"path": "data/ID_1/", "color": "#3498db", "name": "Runner ID_1"},
        "ID_2": {"path": "data/ID_2/", "color": "#e74c3c", "name": "Runner ID_2"},
        "ID_3": {"path": "data/ID_3/", "color": "#9b59b6", "name": "Runner ID_3"},
        "ID_4": {"path": "data/ID_4/", "color": "#f39c12", "name": "Runner ID_4"},
        "ID_5": {"path": "data/ID_5/", "color": "#e67e22", "name": "Runner ID_5"},
        "ID_6": {"path": "data/ID_6/", "color": "#1abc9c", "name": "Runner ID_6"},
        "ID_26": {"path": "data/ID_26/", "color": "#8e44ad", "name": "Runner ID_26"},
        "ID_50": {"path": "data/ID_50/", "color": "#d35400", "name": "Runner ID_50"},
        "ID_74": {"path": "data/ID_74/", "color": "#2c3e50", "name": "Runner ID_74"},
        "ID_94": {"path": "data/ID_94/", "color": "#7f8c8d", "name": "Runner ID_94"},
        "ID_115": {"path": "data/ID_115/", "color": "#16a085", "name": "Runner ID_115"},
        "ID_138": {"path": "data/ID_138/", "color": "#2ecc71", "name": "Runner ID_138"},
        "ID_170": {"path": "data/ID_170/", "color": "#e74c3c", "name": "Runner ID_170"},
        "ID_191": {"path": "data/ID_191/", "color": "#9b59b6", "name": "Runner ID_191"},
        "ID_211": {"path": "data/ID_211/", "color": "#f39c12", "name": "Runner ID_211"},
        "ID_231": {"path": "data/ID_231/", "color": "#27ae60", "name": "Runner ID_231"},
    },
    "model_path": "ml/ml_data/trained_models/best_gnn_autoencoder_20251218_235652.pt",
    "sequence_length": 5,
    "sequence_step": 3,
    "output_dir": "ml/ml_data/embeddings_seq5_step3_20251123",
}


logger = get_logger("athlab.embeddings")


def find_latest_model(model_dir="ml/ml_data/trained_models", pattern="*sequential.pt"):
    """Find the most recently created sequential model."""
    import glob
    import os

    model_files = glob.glob(os.path.join(model_dir, pattern))
    if not model_files:
        # Fall back to any .pt file if no sequential models found
        model_files = glob.glob(os.path.join(model_dir, "*.pt"))

    if not model_files:
        return None

    # Sort by modification time, get the most recent
    latest_model = max(model_files, key=os.path.getmtime)
    logger.info(f"Auto-detected latest model: {latest_model}")

    return latest_model


def load_sequential_model(model_path=None):
    """Load the trained sequential GNN model."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Auto-detect latest model if no path provided
    if model_path is None:
        model_path = find_latest_model()
        if model_path is None:
            model_path = CONFIG["model_path"]  # Fall back to config

    try:
        if os.path.exists(model_path):
            logger.info(f"Loading model from: {model_path}")
            # Load checkpoint
            checkpoint = torch.load(model_path, map_location=device)

            # Get number of nodes from data (89 nodes per timestep * sequence_length)
            num_nodes = 89 * CONFIG["sequence_length"]  # 445 for seq_length=5
            logger.info(f"Number of nodes for sequence length {CONFIG['sequence_length']}: {num_nodes}")

            # Extract architecture from checkpoint to match saved model
            state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint

            # Infer encoder dimensions from state dict
            encoder_dim1 = state_dict["encoder.convs.0.lin.weight"].shape[0]  # First layer output
            encoder_dim2 = state_dict["encoder.convs.1.lin.weight"].shape[0]  # Second layer output
            encoder_dim3 = state_dict["encoder.convs.2.lin.weight"].shape[0]  # Third layer output (embedding)

            # Decoder dimensions are reverse of encoder (excluding final layer which is always 3)
            decoder_dim1 = encoder_dim2  # First decoder layer
            decoder_dim2 = encoder_dim1  # Second decoder layer

            # Check if edge decoder components exist in the saved model
            has_edge_decoder = any(key.startswith("edge_decoder.") for key in state_dict.keys())

            logger.info("Detected architecture from checkpoint:")
            logger.info(f"Encoder: [3, {encoder_dim1}, {encoder_dim2}, {encoder_dim3}]")
            logger.info(f"Decoder: [{encoder_dim2}, {encoder_dim1}, 3]")
            logger.info(f"Edge reconstruction: {has_edge_decoder}")

            # Create model with correct architecture
            model = GNNAutoencoder(
                input_dim=3,
                encoder_hidden_dims=[encoder_dim1, encoder_dim2, encoder_dim3],
                decoder_hidden_dims=[decoder_dim1, decoder_dim2, 3],
                num_nodes=num_nodes,
                reconstruct_edges=has_edge_decoder,  # Only enable if saved model has it
            )

            # Load weights with error handling for missing keys
            if "model_state_dict" in checkpoint:
                model.load_state_dict(checkpoint["model_state_dict"], strict=False)
                training_info = checkpoint
            else:
                model.load_state_dict(checkpoint, strict=False)
                training_info = {}

            model = model.to(device)
            model.eval()

            logger.info("Sequential model loaded successfully")
            logger.info(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
            logger.info(f"Device: {device}")
            logger.info(f"Nodes: {num_nodes}")
            logger.info(f"Embedding dimension: {encoder_dim3}")

            return model, device, training_info

        else:
            logger.error(f"Model file not found: {model_path}")
            return None, None, {}

    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        return None, None, {}


def process_all_emt_files():
    """Process all EMT files and generate embeddings with seq_length=5, seq_step=3."""

    logger.info("\nGENERATING EMBEDDINGS: seq_length=5, seq_step=3")
    logger.info("=" * 55)

    # Load model
    logger.info("Loading sequential GNN autoencoder model...")
    model, device, model_info = load_sequential_model()

    if model is None:
        logger.error("Model not loaded. Cannot process files.")
        return []

    # Find all EMT files
    all_3d_emt_files = []
    for runner_id, runner_info in CONFIG["runners"].items():
        base_path = runner_info["path"]
        if not os.path.exists(base_path):
            logger.warning(f"Path not found: {base_path}")
            continue

        for condition in os.listdir(base_path):
            condition_path = os.path.join(base_path, condition)
            if os.path.isdir(condition_path):
                for duration in os.listdir(condition_path):
                    duration_path = os.path.join(condition_path, duration)
                    if os.path.isdir(duration_path):
                        emt_file = os.path.join(duration_path, "3D Point Tracks.emt")
                        if os.path.exists(emt_file):
                            all_3d_emt_files.append(
                                {
                                    "runner_id": runner_id,
                                    "condition": condition,
                                    "duration": duration,
                                    "file_path": emt_file,
                                }
                            )

    if len(all_3d_emt_files) == 0:
        logger.error("No EMT files found in specified directories.")
        return []

    logger.info(f"Found {len(all_3d_emt_files)} EMT files to process")

    # Ensure output directory exists
    output_dir = CONFIG["output_dir"]
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")

    # Process each file
    processing_results = []
    start_time = time.time()

    for i, file_info in enumerate(all_3d_emt_files):
        logger.info(f"\n[{i + 1}/{len(all_3d_emt_files)}] Processing: {file_info['runner_id']}_{file_info['condition']}_{file_info['duration']}")

        try:
            # Generate output filename
            safe_duration = file_info["duration"].replace(".", "_").replace(" ", "_")
            embeddings_filename = f"embeddings_{file_info['runner_id']}_{file_info['condition']}_{safe_duration}.npy"
            embeddings_path = os.path.join(output_dir, embeddings_filename)

            # Check if embeddings already exist
            if os.path.exists(embeddings_path):
                logger.info(f"   EXISTS: Embeddings already exist: {embeddings_filename}")
                # Load existing to get shape info
                existing_embeddings = np.load(embeddings_path)
                processing_results.append(
                    {
                        "runner_id": file_info["runner_id"],
                        "condition": file_info["condition"],
                        "duration": file_info["duration"],
                        "file_path": file_info["file_path"],
                        "embeddings_path": embeddings_path,
                        "embeddings_shape": existing_embeddings.shape,
                        "status": "already_exists",
                    }
                )
                continue

            # Load EMT file
            logger.info(f"   Loading EMT file: {file_info['file_path']}")
            df = pd.DataFrame.emt3d.from_emt(str(file_info["file_path"]))
            logger.info(f"      Shape: {df.shape}")

            if len(df) < CONFIG["sequence_length"]:  # Need at least sequence_length frames
                logger.warning(f"File too short ({len(df)} frames, need {CONFIG['sequence_length']}). Skipping.")
                processing_results.append(
                    {
                        "runner_id": file_info["runner_id"],
                        "condition": file_info["condition"],
                        "duration": file_info["duration"],
                        "file_path": file_info["file_path"],
                        "embeddings_path": None,
                        "embeddings_shape": None,
                        "status": "too_short",
                    }
                )
                continue

            # Create dataset with seq_length=5, seq_step=3
            logger.info(f"   Creating GNN dataset (seq_length={CONFIG['sequence_length']}, seq_step={CONFIG['sequence_step']})...")
            dataset = GNNVestDataset(
                dataframe=df,
                mode="sequence",
                sequence_length=CONFIG["sequence_length"],
                sequence_step=CONFIG["sequence_step"],
                test_size=0.0,
                val_size=0.0,
            )

            logger.info("   Preprocessing...")
            dataset.preprocess_data(apply_scaling=True)

            logger.info("   Creating graphs...")
            graphs = dataset.create_graphs()
            logger.info(f"      Generated {len(graphs)} graph sequences")

            if len(graphs) == 0:
                logger.warning("   WARNING: No graphs generated. Skipping.")
                processing_results.append(
                    {
                        "runner_id": file_info["runner_id"],
                        "condition": file_info["condition"],
                        "duration": file_info["duration"],
                        "file_path": file_info["file_path"],
                        "embeddings_path": None,
                        "embeddings_shape": None,
                        "status": "no_graphs",
                    }
                )
                continue

            # Generate embeddings
            logger.info("   Generating embeddings...")
            embeddings = []
            with torch.no_grad():
                for j, graph in enumerate(graphs):
                    if j % 100 == 0 and j > 0:
                        logger.info(f"      Progress: {j}/{len(graphs)} sequences processed")

                    x = graph.x.to(device)
                    edge_index = graph.edge_index.to(device)
                    graph_embedding, _ = model.encode(x, edge_index)
                    embeddings.append(graph_embedding.cpu().numpy())

            # Convert to numpy array and save
            embeddings_array = np.array(embeddings)
            np.save(embeddings_path, embeddings_array)

            logger.info(f"   Saved embeddings: {embeddings_filename}")
            logger.info(f"      Shape: {embeddings_array.shape}")

            processing_results.append(
                {
                    "runner_id": file_info["runner_id"],
                    "condition": file_info["condition"],
                    "duration": file_info["duration"],
                    "file_path": file_info["file_path"],
                    "embeddings_path": embeddings_path,
                    "embeddings_shape": embeddings_array.shape,
                    "status": "success",
                }
            )

        except Exception as e:
            logger.error(f"   ERROR processing {file_info['file_path']}: {str(e)}")
            processing_results.append(
                {
                    "runner_id": file_info["runner_id"],
                    "condition": file_info["condition"],
                    "duration": file_info["duration"],
                    "file_path": file_info["file_path"],
                    "embeddings_path": None,
                    "embeddings_shape": None,
                    "status": f"error: {str(e)}",
                }
            )

    # Save processing summary
    summary_path = os.path.join(output_dir, "processing_summary.json")
    summary_data = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "sequence_length": CONFIG["sequence_length"],
            "sequence_step": CONFIG["sequence_step"],
            "output_dir": CONFIG["output_dir"],
        },
        "total_files": len(all_3d_emt_files),
        "successful": len([r for r in processing_results if r["status"] == "success"]),
        "already_existed": len([r for r in processing_results if r["status"] == "already_exists"]),
        "errors": len([r for r in processing_results if r["status"].startswith("error")]),
        "total_time_seconds": time.time() - start_time,
        "results": processing_results,
    }

    with open(summary_path, "w") as f:
        json.dump(summary_data, f, indent=2, default=str)

        # Print final summary
        logger.info("\nPROCESSING COMPLETE!")
        logger.info("=" * 25)
        logger.info(f"   Total files found: {len(all_3d_emt_files)}")
        logger.info(f"   Successfully processed: {summary_data['successful']}")
        logger.info(f"   Already existed: {summary_data['already_existed']}")
        logger.info(f"   Errors: {summary_data['errors']}")
        logger.info(f"   Total time: {summary_data['total_time_seconds']:.1f} seconds")
        logger.info(f"   Embeddings saved to: {output_dir}")
        logger.info(f"   Summary saved to: {summary_path}")

    # Show breakdown by status
    status_counts = {}
    for result in processing_results:
        status = result["status"]
        if status.startswith("error"):
            status = "error"
        status_counts[status] = status_counts.get(status, 0) + 1

        logger.info("\nStatus Breakdown:")
    for status, count in status_counts.items():
        logger.info(f"   - {status}: {count}")

    return processing_results


def main():
    """Main execution function."""
    logger.info("EMBEDDING GENERATION SCRIPT")
    logger.info(f"Target directory: {CONFIG['output_dir']}")
    logger.info(f"Sequence parameters: length={CONFIG['sequence_length']}, step={CONFIG['sequence_step']}")
    logger.info(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    results = process_all_emt_files()

    if len(results) > 0:
        successful = len([r for r in results if r["status"] in ["success", "already_exists"]])
        total = len(results)
        logger.info("\nFINAL SUMMARY:")
        logger.info(f"   - Processed {successful}/{total} files successfully")
        logger.info("   - Ready for aggregation with process_all_embeddings.py")
        logger.info(f"   - Next step: Run aggregation on {CONFIG['output_dir']}")
    else:
        logger.error("\nERROR: No files were processed.")
        logger.info(f"\nEnd time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
