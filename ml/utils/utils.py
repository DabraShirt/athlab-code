import numpy as np
import plotly.graph_objects as go
from torch_geometric.data import Data


def plot_3d_graph(
    data: Data,
    frame_idx: int = 0,
    show_node_labels: bool = False,
    show_edge_labels: bool = True,
    show_node_distances: bool = False,
    label_offset: float = 0.05,
    nodes_to_show: list = None,
) -> go.Figure:
    """
    Plot a 3D network graph using Plotly with edge distances displayed.

    Parameters
    ----------
    data : PyG Data object
        A data object containing:
        - x : ndarray
            Node features of shape [num_frames, num_points, 3].
        - edge_index : ndarray
            Edge connections of shape [2, num_edges].
        - edge_attr : ndarray
            Edge features of shape [num_frames, num_edges, 1].
    frame_idx : int, optional
        The frame index to plot, by default 0.
    show_node_labels : bool, optional
        Whether to show node labels (node IDs), by default False.
    show_edge_labels : bool, optional
        Whether to show edge distance labels, by default True.
    show_node_distances : bool, optional
        Whether to show distance from origin to each node, by default False.
    label_offset : float, optional
        The offset distance for label positioning, by default 0.05.
    nodes_to_show : list, optional
        List of node indices (0-based) to visualize. If None, show all nodes, by default None.

    Returns
    -------
    plotly.graph_objects.Figure
        A Plotly figure object representing the 3D network graph.
    """
    # Extract node positions - handle both old and new formats
    if len(data.x.shape) == 3:  # Old format: [num_frames, num_points, 3]
        positions = data.x[frame_idx].detach().cpu().numpy()  # Shape: [num_points, 3]
        distances = data.edge_attr[frame_idx].detach().cpu().numpy()  # Shape: [num_edges, 1]
    else:  # New format: [num_points, 3] - PyTorch Geometric Data format
        positions = data.x.detach().cpu().numpy()  # Shape: [num_points, 3]
        distances = data.edge_attr.detach().cpu().numpy()  # Shape: [num_edges, 1]

    # If nodes_to_show is specified, convert to 0-based indexing if needed
    if nodes_to_show is not None:
        # Convert to set for faster lookup and ensure 0-based indexing
        nodes_to_show_set = set(nodes_to_show)
        # If the nodes are 1-based, convert to 0-based
        if all(node >= 1 for node in nodes_to_show):
            nodes_to_show_set = {node - 1 for node in nodes_to_show}
    else:
        # Show all nodes
        nodes_to_show_set = set(range(len(positions)))

    # Create edges trace
    edge_x = []
    edge_y = []
    edge_z = []
    node_texts = []
    edge_texts = []
    node_distance_texts = []

    # Get edges and their coordinates - only show edges between visible nodes
    for i, edge in enumerate(data.edge_index.t()):
        start_idx, end_idx = edge.detach().cpu().numpy()

        # Only show edge if both nodes are in the nodes_to_show list
        if start_idx in nodes_to_show_set and end_idx in nodes_to_show_set:
            start_pos = positions[start_idx]
            end_pos = positions[end_idx]

            # Add edge coordinates
            edge_x.extend([start_pos[0], end_pos[0], None])
            edge_y.extend([start_pos[1], end_pos[1], None])
            edge_z.extend([start_pos[2], end_pos[2], None])

            # Add distance text at the middle of the edge (only if show_edge_labels is True)
            if show_edge_labels:
                mid_x = (start_pos[0] + end_pos[0]) / 2
                mid_y = (start_pos[1] + end_pos[1]) / 2
                mid_z = (start_pos[2] + end_pos[2]) / 2

                # Add text annotation for distance
                edge_texts.append(
                    go.Scatter3d(
                        x=[mid_x],
                        y=[mid_y],
                        z=[mid_z],
                        mode="text",
                        text=[f"{distances[i][0]:.1f}"],
                        textposition="middle center",
                        hoverinfo="none",
                        showlegend=False,
                        textfont=dict(color="green", size=10),
                    )
                )

    # Add node labels (node IDs) if show_node_labels is True - only for visible nodes
    if show_node_labels:
        for i in nodes_to_show_set:
            pos = positions[i]
            # Place labels directly on the nodes (no offset)

            # Add text annotation for node ID
            node_texts.append(
                go.Scatter3d(
                    x=[pos[0]],
                    y=[pos[1]],
                    z=[pos[2]],
                    mode="text",
                    text=[f"{i + 1}"],  # Node ID (1-indexed)
                    textposition="middle center",
                    hoverinfo="none",
                    showlegend=False,
                    textfont=dict(color="white", size=11, family="Arial Black"),
                )
            )

    # Add distance from origin to each node (only if show_node_distances is True) - only for visible nodes
    if show_node_distances:
        for i in nodes_to_show_set:
            pos = positions[i]
            # Calculate distance from origin
            node_distance = np.sqrt(pos[0] ** 2 + pos[1] ** 2 + pos[2] ** 2)

            # Offset the text slightly from the node position for better visibility
            offset_x = pos[0] + label_offset * 1.5  # Different offset for distances
            offset_y = pos[1] + label_offset * 1.5
            offset_z = pos[2] + label_offset * 1.5

            # Add text annotation for node distance from origin
            node_distance_texts.append(
                go.Scatter3d(
                    x=[offset_x],
                    y=[offset_y],
                    z=[offset_z],
                    mode="text",
                    text=[f"{node_distance:.1f}"],
                    textposition="middle center",
                    hoverinfo="none",
                    showlegend=False,
                    textfont=dict(color="green", size=6),
                )
            )

    # Create edges trace
    edges_trace = go.Scatter3d(
        x=edge_x,
        y=edge_y,
        z=edge_z,
        mode="lines",
        line=dict(color="lightgray", width=3),
        hoverinfo="none",
        showlegend=False,
    )

    # Create nodes trace - only for visible nodes
    visible_positions = positions[list(nodes_to_show_set)]
    visible_indices = list(nodes_to_show_set)

    nodes_trace = go.Scatter3d(
        x=visible_positions[:, 0],
        y=visible_positions[:, 1],
        z=visible_positions[:, 2],
        mode="markers",
        marker=dict(size=12, color="darkblue", symbol="circle", line=dict(width=2, color="navy")),
        hovertext=[f"Point {i + 1}<br>X: {positions[i][0]:.1f}<br>Y: {positions[i][1]:.1f}<br>Z: {positions[i][2]:.1f}<br>Distance: {np.sqrt(positions[i][0]**2 + positions[i][1]**2 + positions[i][2]**2):.1f}" for i in visible_indices],
        hoverinfo="text",
        showlegend=False,
    )

    # Combine all traces
    traces = [edges_trace, nodes_trace] + edge_texts + node_texts + node_distance_texts

    # Create the figure
    fig = go.Figure(data=traces)

    # Update layout for better 3D visualization
    fig.update_layout(
        title="3D Network Graph",
        width=1200,  # Make the graph much wider for notebook display
        height=800,  # Make the graph much taller for notebook display
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z",
            aspectmode="data",
            camera=dict(eye=dict(x=1.2, y=1.2, z=1.2)),
        ),
        showlegend=False,
    )

    return fig
