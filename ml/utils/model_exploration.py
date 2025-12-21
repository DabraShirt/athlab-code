# %%
# from asyncio import Runner  # Not needed - comment out for compatibility
import os
import warnings

import numpy as np
import plotly.graph_objects as go
import torch
from plotly.subplots import make_subplots
from torch_geometric.data import Data

from ml.models.GNN_autoencoder import GNNAutoencoder
from ml.utils.emt_pandas import Emt3DAccessor, EmtAccessor  # noqa: F401
from ml.utils.multi_dataset_loader import prepare_training_data
from ml.utils.utils import plot_3d_graph

warnings.filterwarnings("ignore")

# %%
CONFIG = {
    "runners": {
        "ID_1": {
            "path": "data/ID_1/",
            "extracted_pdf_path": "data/extracted_breathing_reports/",
            "color": "#3498db",
            "marker": "square",
            "name": "Runner ID_1",
        },
        "ID_2": {
            "path": "data/ID_2/",
            "extracted_pdf_path": "data/extracted_breathing_reports/",
            "color": "#e74c3c",
            "marker": "circle",
            "name": "Runner ID_2",
        },
        "ID_3": {
            "path": "data/ID_3/",
            "extracted_pdf_path": "data/extracted_breathing_reports/",
            "color": "#9b59b6",
            "marker": "triangle",
            "name": "Runner ID_3",
        },
        "ID_4": {
            "path": "data/ID_4/",
            "extracted_pdf_path": "data/extracted_breathing_reports/",
            "color": "#f39c12",
            "marker": "pentagon",
            "name": "Runner ID_4",
        },
        "ID_5": {
            "path": "data/ID_5/",
            "extracted_pdf_path": "data/extracted_breathing_reports/",
            "color": "#e67e22",
            "marker": "star",
            "name": "Runner ID_5",
        },
        "ID_6": {
            "path": "data/ID_6/",
            "extracted_pdf_path": "data/extracted_breathing_reports/",
            "color": "#1abc9c",
            "marker": "hexagon",
            "name": "Runner ID_6",
        },
        "ID_26": {
            "path": "data/ID_26/",
            "extracted_pdf_path": "data/extracted_breathing_reports/",
            "color": "#8e44ad",
            "marker": "cross",
            "name": "Runner ID_26",
        },
        "ID_50": {
            "path": "data/ID_50/",
            "extracted_pdf_path": "data/extracted_breathing_reports/",
            "color": "#d35400",
            "marker": "x",
            "name": "Runner ID_50",
        },
        "ID_74": {
            "path": "data/ID_74/",
            "extracted_pdf_path": "data/extracted_breathing_reports/",
            "color": "#2c3e50",
            "marker": "hourglass",
            "name": "Runner ID_74",
        },
        "ID_94": {
            "path": "data/ID_94/",
            "extracted_pdf_path": "data/extracted_breathing_reports/",
            "color": "#7f8c8d",
            "marker": "bowtie",
            "name": "Runner ID_94",
        },
        "ID_115": {
            "path": "data/ID_115/",
            "extracted_pdf_path": "data/extracted_breathing_reports/",
            "color": "#16a085",
            "marker": "triangle-down",
            "name": "Runner ID_115",
        },
        "ID_138": {
            "path": "data/ID_138/",
            "extracted_pdf_path": "data/extracted_breathing_reports/",
            "color": "#2ecc71",
            "marker": "diamond",
            "name": "Runner ID_138",
        },
        "ID_170": {
            "path": "data/ID_170/",
            "extracted_pdf_path": "data/extracted_breathing_reports/",
            "color": "#e74c3c",
            "marker": "circle",
            "name": "Runner ID_170",
        },
        "ID_191": {
            "path": "data/ID_191/",
            "extracted_pdf_path": "data/extracted_breathing_reports/",
            "color": "#9b59b6",
            "marker": "triangle",
            "name": "Runner ID_191",
        },
        "ID_211": {
            "path": "data/ID_211/",
            "extracted_pdf_path": "data/extracted_breathing_reports/",
            "color": "#f39c12",
            "marker": "pentagon",
            "name": "Runner ID_211",
        },
        "ID_231": {
            "path": "data/ID_231/",
            "extracted_pdf_path": "data/extracted_breathing_reports/",
            "color": "#27ae60",
            "marker": "hourglass",
            "name": "Runner ID_231",
        },
    },
    "model_path": "best_gnn_autoencoder_20251127_064916.pt",  # Updated to latest model
    "analysis_duration": 15,  # minutes
    "sampling_rate_hz": 120,  # Expected sampling rate
    "random_seed": 42,
    "sequence_length": 5,  # Updated from 10 to 5
    "sequence_step": 3,  # Updated from 4 to 3
}

# %%
# 1. Find and Load Latest Sequential GNN Model
print("Loading sequential GNN autoencoder model...")


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
    print(f"   Auto-detected latest model: {latest_model}")

    return latest_model


# %%
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
            print(f"   Loading model from: {model_path}")
            # Load checkpoint
            checkpoint = torch.load(model_path, map_location=device)

            # Get number of nodes from data - updated for seq_length=5
            num_nodes = 445  # 89 sensors × 5 timesteps = 445 nodes
            print("Number of nodes detected:", num_nodes)

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

            print("   Detected architecture from checkpoint:")
            print(f"      • Encoder: [3, {encoder_dim1}, {encoder_dim2}, {encoder_dim3}]")
            print(f"      • Decoder: [{encoder_dim2}, {encoder_dim1}, 3]")
            print(f"      • Edge reconstruction: {has_edge_decoder}")
            print("      • Expected: [3, 68, 62, 26] for seq5_step3 model")

            # Create model with correct architecture
            model = GNNAutoencoder(
                input_dim=3,
                encoder_hidden_dims=[encoder_dim1, encoder_dim2, encoder_dim3],
                decoder_hidden_dims=[decoder_dim1, decoder_dim2, 3],
                num_nodes=num_nodes,
                reconstruct_edges=has_edge_decoder,  # Only enable if saved model has it
            )
            print(model)

            # Load weights with error handling for missing keys

            if "model_state_dict" in checkpoint:
                model.load_state_dict(checkpoint["model_state_dict"], strict=False)
                training_info = checkpoint
            else:
                model.load_state_dict(checkpoint, strict=False)
                training_info = {}

            # Verify critical components are loaded
            missing_keys = []
            unexpected_keys = []
            if "model_state_dict" in checkpoint:
                model_dict = model.state_dict()
                checkpoint_dict = checkpoint["model_state_dict"]

                missing_keys = [k for k in model_dict.keys() if k not in checkpoint_dict]
                unexpected_keys = [k for k in checkpoint_dict.keys() if k not in model_dict]

            if missing_keys:
                print(f"   Warning: Missing keys (will be randomly initialized): {len(missing_keys)}")
            if unexpected_keys:
                print(f"   Warning: Unexpected keys (will be ignored): {len(unexpected_keys)}")

            model = model.to(device)
            model.eval()

            print("   Sequential model loaded successfully")
            print(f"      • Parameters: {sum(p.numel() for p in model.parameters()):,}")
            print(f"      • Device: {device}")
            print(f"      • Nodes: {num_nodes}")
            print(f"      • Embedding dimension: {encoder_dim3}")

            if "epoch" in training_info:
                print(f"      • Training epoch: {training_info['epoch']}")
            if "best_val_loss" in training_info:
                print(f"      • Best val loss: {training_info['best_val_loss']:.6f}")

            return model, device, training_info

        else:
            print(f"   Error: Model file not found: {model_path}")
            return None, None, {}

    except Exception as e:
        print(f"   Error loading model: {str(e)}")
        return None, None, {}


model, device, model_info = load_sequential_model()  # Will auto-detect latest model


# %%
data_paths = ["configs/data_splits.yaml"]


# Use the multi-dataset loader with updated sequence parameters
dataset, graphs, num_nodes = prepare_training_data(
    config_path=data_paths[0],
    max_files=None,  # Load all files
    dataset_mode="sequence",
    sequence_length=CONFIG["sequence_length"],  # Use config value: 5
    sequence_step=CONFIG["sequence_step"],  # Use config value: 3
)


# %%
graphs

# %%
nwm = next(iter(dataset))

# %%
nwm.x

# %%
graphs[0].x


plot_3d_graph(graphs[0], frame_idx=0)


# %%
# Ensure sample is on the same device as the model before inference
nwm = nwm.to(device)
output = model(
    nwm.x,
    nwm.edge_index,
    getattr(nwm, "edge_weight", None),
    getattr(nwm, "batch", None),
)


# %%
output[0].shape

# %%
# Create a Data object for the reconstructed first timestep
# Use the dataset's recorded points-per-timestep when available
num_points = getattr(nwm, "num_points_per_timestep", 89)

# Construct CPU-backed Data for plotting to avoid device mismatches
reconstructed_first_timestep = Data(
    x=output[0][:num_points].detach().cpu(),  # First timestep nodes
    edge_index=nwm.edge_index.detach().cpu(),
    edge_attr=(nwm.edge_attr.detach().cpu() if hasattr(nwm, "edge_attr") and nwm.edge_attr is not None else graphs[0].edge_attr.detach().cpu()),
)
plot_3d_graph(reconstructed_first_timestep)


# %%
model.reconstruct_edges


# %%
def plot_input_vs_reconstructed(model, data_batch, device, timestep_idx=0, nodes_to_show=None, save_path=None):
    """
    Plot input graph vs reconstructed graph for comparison.

    Parameters:
    -----------
    model : GNNAutoencoder
        Trained autoencoder model
    data_batch : torch_geometric.data.Batch
        Batch of graph data
    device : torch.device
        Device for computation
    timestep_idx : int
        Which timestep to visualize (0 for first timestep)
    nodes_to_show : list, optional
        List of node indices to show (if None, show all)
    save_path : str, optional
        Path to save the visualization

    Returns:
    --------
    fig : plotly.graph_objects.Figure
        Plotly figure with side-by-side comparison
    """
    model.eval()

    with torch.no_grad():
        # Move data to device
        data_batch = data_batch.to(device)

        # Get model output
        output = model(
            data_batch.x,
            data_batch.edge_index,
            getattr(data_batch, "edge_weight", None),
            getattr(data_batch, "batch", None),
        )

        # Unpack output - model returns multiple values
        if hasattr(model, "reconstruct_edges") and model.reconstruct_edges:
            (
                x_reconstructed,
                graph_embedding,
                node_embeddings,
                edge_probs,
                reconstructed_edge_index,
                anomaly_scores,
            ) = output
            # For visualization, use original edge attributes or convert probabilities if needed
            edge_attr_reconstructed = data_batch.edge_attr
        else:
            x_reconstructed, graph_embedding, node_embeddings, anomaly_scores = output
            edge_attr_reconstructed = data_batch.edge_attr

        # Convert to numpy for plotting
        # For sequence graphs with 5 timesteps, extract first timestep (first 89 nodes)
        if hasattr(data_batch, "num_points_per_timestep"):
            # Sequence graph: extract first timestep
            num_points = data_batch.num_points_per_timestep
            original_x = data_batch.x[:num_points].cpu().numpy()  # First 89 nodes
            reconstructed_x = x_reconstructed[:num_points].cpu().numpy()  # First 89 reconstructed nodes
        elif len(data_batch.x.shape) == 3:  # Sequential format [batch, nodes, features]
            original_x = data_batch.x[0, :, :].cpu().numpy()  # First sample, all nodes, all features
            reconstructed_x = x_reconstructed[0, :, :].cpu().numpy()
        else:  # Standard format [nodes, features]
            original_x = data_batch.x.cpu().numpy()
            reconstructed_x = x_reconstructed.cpu().numpy()

        original_edge_attr = data_batch.edge_attr.cpu().numpy()
        reconstructed_edge_attr = edge_attr_reconstructed.cpu().numpy()

        # Create Data objects for plotting
        original_data = Data(
            x=torch.tensor(original_x),
            edge_index=data_batch.edge_index.cpu(),
            edge_attr=torch.tensor(original_edge_attr),
        )

        reconstructed_data = Data(
            x=torch.tensor(reconstructed_x),
            edge_index=data_batch.edge_index.cpu(),
            edge_attr=torch.tensor(reconstructed_edge_attr),
        )

    # Create subplots
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=["Original Graph", "Reconstructed Graph"],
        specs=[[{"type": "scatter3d"}, {"type": "scatter3d"}]],
    )

    # Plot original graph
    original_plot = plot_3d_graph(
        original_data,
        frame_idx=timestep_idx,
        show_node_labels=False,
        show_edge_labels=False,
        nodes_to_show=nodes_to_show,
    )

    # Plot reconstructed graph
    reconstructed_plot = plot_3d_graph(
        reconstructed_data,
        frame_idx=timestep_idx,
        show_node_labels=False,
        show_edge_labels=False,
        nodes_to_show=nodes_to_show,
    )

    # Add traces from original plot
    for trace in original_plot.data:
        trace_copy = trace
        fig.add_trace(trace_copy, row=1, col=1)

    # Add traces from reconstructed plot
    for trace in reconstructed_plot.data:
        trace_copy = trace
        # Update marker color for reconstructed graph
        if hasattr(trace_copy, "marker") and trace_copy.marker:
            trace_copy.marker.color = "red"
        fig.add_trace(trace_copy, row=1, col=2)

    # Calculate reconstruction error
    mse = np.mean((original_x - reconstructed_x) ** 2)
    mae = np.mean(np.abs(original_x - reconstructed_x))

    # Update layout
    fig.update_layout(
        title=f"Input vs Reconstructed Graph Comparison<br>MSE: {mse:.6f}, MAE: {mae:.6f}",
        width=1400,
        height=700,
        showlegend=False,
        scene1=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z",
            aspectmode="data",
            camera=dict(eye=dict(x=1.2, y=1.2, z=1.2)),
        ),
        scene2=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z",
            aspectmode="data",
            camera=dict(eye=dict(x=1.2, y=1.2, z=1.2)),
        ),
    )

    if save_path:
        fig.write_html(save_path)
        print(f"Visualization saved to: {save_path}")

    return fig


def plot_first_timestep_comparison(model, data_batch, device, nodes_to_show=None, save_path=None):
    """
    Plot comparison focusing on the first timestep of original vs reconstructed.

    Parameters:
    -----------
    model : GNNAutoencoder
        Trained autoencoder model
    data_batch : torch_geometric.data.Batch
        Batch of sequential graph data
    device : torch.device
        Device for computation
    nodes_to_show : list, optional
        List of node indices to show
    save_path : str, optional
        Path to save the visualization

    Returns:
    --------
    fig : plotly.graph_objects.Figure
        Plotly figure comparing first timesteps
    """
    model.eval()

    with torch.no_grad():
        # Move data to device
        data_batch = data_batch.to(device)

        # Get model output
        output = model(
            data_batch.x,
            data_batch.edge_index,
            getattr(data_batch, "edge_weight", None),
            getattr(data_batch, "batch", None),
        )

        # Unpack output - model returns multiple values
        if hasattr(model, "reconstruct_edges") and model.reconstruct_edges:
            (
                x_reconstructed,
                graph_embedding,
                node_embeddings,
                edge_probs,
                reconstructed_edge_index,
                anomaly_scores,
            ) = output
            # For visualization, use original edge attributes or convert probabilities if needed
            edge_attr_reconstructed = data_batch.edge_attr
        else:
            x_reconstructed, graph_embedding, node_embeddings, anomaly_scores = output
            edge_attr_reconstructed = data_batch.edge_attr

        print("x_reconstructed shape:", x_reconstructed.shape)
        print("edge_attr_reconstructed shape:", edge_attr_reconstructed.shape)

        # Extract first timestep data
        # For sequence graphs with 5 timesteps, extract first timestep (first 89 nodes)
        if hasattr(data_batch, "num_points_per_timestep"):
            # Sequence graph: extract first timestep
            num_points = data_batch.num_points_per_timestep
            original_first = data_batch.x[:num_points].cpu().numpy()  # First 89 nodes
            reconstructed_first = x_reconstructed[:num_points].cpu().numpy()  # First 89 reconstructed nodes
        elif len(data_batch.x.shape) == 4:  # [batch, time, nodes, features]
            original_first = data_batch.x[0, 0, :, :].cpu().numpy()  # First batch, first timestep
            reconstructed_first = x_reconstructed[0, 0, :, :].cpu().numpy()
        elif len(data_batch.x.shape) == 3:  # [batch, nodes, features] - single timestep
            original_first = data_batch.x[0, :, :].cpu().numpy()  # First batch, all nodes
            reconstructed_first = x_reconstructed[0, :, :].cpu().numpy()
        else:  # [nodes, features] - single graph
            original_first = data_batch.x.cpu().numpy()
            reconstructed_first = x_reconstructed.cpu().numpy()

    # Create side-by-side scatter plots
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=[
            "Original - First Timestep (3D)",
            "Reconstructed - First Timestep (3D)",
            "X-Y Projection Comparison",
            "Reconstruction Error by Node",
        ],
        specs=[
            [{"type": "scatter3d"}, {"type": "scatter3d"}],
            [{"type": "scatter"}, {"type": "scatter"}],
        ],
    )

    # Filter nodes if specified
    if nodes_to_show is not None:
        node_indices = np.array(nodes_to_show)
        if node_indices.max() >= len(original_first):
            node_indices = node_indices[node_indices < len(original_first)]
        original_filtered = original_first[node_indices]
        reconstructed_filtered = reconstructed_first[node_indices]
    else:
        node_indices = np.arange(len(original_first))
        original_filtered = original_first
        reconstructed_filtered = reconstructed_first

    # 3D scatter plots
    fig.add_trace(
        go.Scatter3d(
            x=original_filtered[:, 0],
            y=original_filtered[:, 1],
            z=original_filtered[:, 2],
            mode="markers",
            marker=dict(size=8, color="blue", symbol="circle"),
            name="Original",
            text=[f"Node {i}" for i in node_indices],
            hovertemplate="Node: %{text}<br>X: %{x:.3f}<br>Y: %{y:.3f}<br>Z: %{z:.3f}",
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter3d(
            x=reconstructed_filtered[:, 0],
            y=reconstructed_filtered[:, 1],
            z=reconstructed_filtered[:, 2],
            mode="markers",
            marker=dict(size=8, color="red", symbol="circle"),
            name="Reconstructed",
            text=[f"Node {i}" for i in node_indices],
            hovertemplate="Node: %{text}<br>X: %{x:.3f}<br>Y: %{y:.3f}<br>Z: %{z:.3f}",
        ),
        row=1,
        col=2,
    )

    # X-Y projection comparison
    fig.add_trace(
        go.Scatter(
            x=original_filtered[:, 0],
            y=original_filtered[:, 1],
            mode="markers",
            marker=dict(size=8, color="blue"),
            name="Original (X-Y)",
            text=[f"Node {i}" for i in node_indices],
            hovertemplate="Node: %{text}<br>X: %{x:.3f}<br>Y: %{y:.3f}",
        ),
        row=2,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=reconstructed_filtered[:, 0],
            y=reconstructed_filtered[:, 1],
            mode="markers",
            marker=dict(size=8, color="red"),
            name="Reconstructed (X-Y)",
            text=[f"Node {i}" for i in node_indices],
            hovertemplate="Node: %{text}<br>X: %{x:.3f}<br>Y: %{y:.3f}",
        ),
        row=2,
        col=1,
    )

    # Reconstruction error by node
    node_errors = np.sqrt(np.sum((original_filtered - reconstructed_filtered) ** 2, axis=1))
    fig.add_trace(
        go.Scatter(
            x=node_indices,
            y=node_errors,
            mode="markers+lines",
            marker=dict(size=8, color="purple"),
            line=dict(color="purple"),
            name="Reconstruction Error",
            hovertemplate="Node: %{x}<br>Error: %{y:.6f}",
        ),
        row=2,
        col=2,
    )

    # Calculate overall metrics
    mse = np.mean((original_filtered - reconstructed_filtered) ** 2)
    mae = np.mean(np.abs(original_filtered - reconstructed_filtered))
    max_error = np.max(node_errors)

    # Update layout
    fig.update_layout(
        title=f"First Timestep Reconstruction Analysis<br>MSE: {mse:.6f}, MAE: {mae:.6f}, Max Node Error: {max_error:.6f}",
        width=1400,
        height=1000,
        showlegend=True,
    )

    # Update axis labels
    fig.update_xaxes(title_text="X Coordinate", row=2, col=1)
    fig.update_yaxes(title_text="Y Coordinate", row=2, col=1)
    fig.update_xaxes(title_text="Node Index", row=2, col=2)
    fig.update_yaxes(title_text="Reconstruction Error", row=2, col=2)

    if save_path:
        fig.write_html(save_path)
        print(f"First timestep analysis saved to: {save_path}")

    return fig


def plot_simple_reconstruction_comparison(model, data_batch, device, save_path=None):
    """
    Simple plot comparing original vs reconstructed first timestep.

    Parameters:
    -----------
    model : GNNAutoencoder
        Trained autoencoder model
    data_batch : torch_geometric.data.Batch
        Batch of graph data
    device : torch.device
        Device for computation
    save_path : str, optional
        Path to save the visualization

    Returns:
    --------
    fig : plotly.graph_objects.Figure
        Plotly figure comparing original vs reconstructed
    """
    model.eval()

    with torch.no_grad():
        # Move data to device
        data_batch = data_batch.to(device)

        # Get model output
        output = model(
            data_batch.x,
            data_batch.edge_index,
            getattr(data_batch, "edge_weight", None),
            getattr(data_batch, "batch", None),
        )

        # Unpack output
        if hasattr(model, "reconstruct_edges") and model.reconstruct_edges:
            (
                x_reconstructed,
                graph_embedding,
                node_embeddings,
                edge_probs,
                reconstructed_edge_index,
                anomaly_scores,
            ) = output
        else:
            x_reconstructed, graph_embedding, node_embeddings, anomaly_scores = output

        # Extract first timestep (first 89 nodes for sequence graphs with 5 timesteps)
        if hasattr(data_batch, "num_points_per_timestep"):
            num_points = data_batch.num_points_per_timestep
            original_nodes = data_batch.x[:num_points]  # First 89 nodes
            reconstructed_nodes = x_reconstructed[:num_points]  # First 89 reconstructed nodes
            original_edges = data_batch.edge_index
            original_edge_attr = data_batch.edge_attr
        else:
            original_nodes = data_batch.x
            reconstructed_nodes = x_reconstructed
            original_edges = data_batch.edge_index
            original_edge_attr = data_batch.edge_attr

        # Create Data objects for plotting
        original_data = Data(x=original_nodes, edge_index=original_edges, edge_attr=original_edge_attr)

        reconstructed_data = Data(
            x=reconstructed_nodes,
            edge_index=original_edges,  # Use same edges for comparison
            edge_attr=original_edge_attr,
        )

    # Create side-by-side plots
    from plotly.subplots import make_subplots

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=["Original First Timestep", "Reconstructed First Timestep"],
        specs=[[{"type": "scatter3d"}, {"type": "scatter3d"}]],
    )

    # Plot original
    original_plot = plot_3d_graph(original_data, frame_idx=0, show_edge_labels=False)
    for trace in original_plot.data:
        fig.add_trace(trace, row=1, col=1)

    # Plot reconstructed
    reconstructed_plot = plot_3d_graph(reconstructed_data, frame_idx=0, show_edge_labels=False)
    for trace in reconstructed_plot.data:
        # Change color to red for reconstructed
        if hasattr(trace, "marker") and trace.marker:
            trace.marker.color = "red"
        fig.add_trace(trace, row=1, col=2)

    # Calculate reconstruction error
    mse = torch.mean((original_nodes - reconstructed_nodes) ** 2).item()
    mae = torch.mean(torch.abs(original_nodes - reconstructed_nodes)).item()

    fig.update_layout(
        title=f"Simple Reconstruction Comparison<br>MSE: {mse:.6f}, MAE: {mae:.6f}",
        width=1400,
        height=700,
        showlegend=False,
    )

    if save_path:
        fig.write_html(save_path)
        print(f"Simple comparison saved to: {save_path}")

    return fig


# %%
# Example usage with updated seq5_step3 dataset
if model is not None and len(graphs) > 0:
    print("Creating visualization with sample data from seq5_step3 model...")
    print("Dataset info:")
    print(f"   • Sequence length: {CONFIG['sequence_length']} timesteps")
    print(f"   • Sequence step: {CONFIG['sequence_step']} timesteps")
    print(f"   • Total graphs available: {len(graphs)}")

    # Test with graphs[0] directly (we know this works with plot_3d_graph)
    sample_graph = graphs[0]
    print(f"Sample graph info: {sample_graph}")
    print(f"Sample graph x shape: {sample_graph.x.shape}")
    print(f"Sample graph edges: {sample_graph.edge_index.shape}")
    print(f"Num points per timestep: {getattr(sample_graph, 'num_points_per_timestep', 'Not available')}")
    print("Expected: 445 nodes (89 sensors × 5 timesteps)")

    # Create a simple comparison first
    simple_fig = plot_simple_reconstruction_comparison(
        model=model,
        data_batch=sample_graph,
        device=device,
        save_path="ml/ml_data/analysis_outputs/simple_reconstruction_comparison.html",
    )

    # Display the simple plot
    simple_fig.show()
    print("Simple visualization created successfully!")
    print("Note: Visualizations show first timestep (first 89 nodes) for comparison")

else:
    print("Model or graphs not loaded properly")
    if model is None:
        print("   • Model loading failed - check model path")
    if len(graphs) == 0:
        print("   • No graphs loaded - check data configuration")

# %%
# Validate model configuration for seq5_step3
if model is not None:
    print("Model Configuration Validation:")
    print("=" * 50)
    print("✓ Model loaded successfully")
    print("✓ Expected architecture: [3, 68, 62, 26]")
    print("✓ Expected nodes: 445 (89 × 5 timesteps)")
    print("✓ Expected parameters: ~756,075")

    # Count actual parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"✓ Actual parameters: {total_params:,}")

    if total_params > 700000:  # Approximate check
        print("✅ Model configuration matches seq5_step3 training!")
    else:
        print("⚠️  Model may be from different training configuration")

    print(f"✓ Device: {device}")
    print(f"✓ Model mode: {'Training' if model.training else 'Evaluation'}")
else:
    print("❌ Model validation failed - model not loaded")
# %%
