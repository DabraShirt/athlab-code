import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv, global_max_pool, global_mean_pool
from torch_geometric.utils import negative_sampling, scatter


class GNNEncoder(nn.Module):
    """
    Graph Neural Network Encoder for breathing pattern analysis.

    A PyTorch module that encodes graph-structured data representing breathing
    patterns into low-dimensional embeddings using Graph Convolutional Networks
    or Graph Attention Networks.

    Parameters
    ----------
    input_dim : int, default 3
        Dimensionality of input node features (e.g., 3D coordinates).
    hidden_dims : list of int, default [64, 32, 16]
        List of hidden layer dimensions for the encoder architecture.
    conv_type : {"GCN", "GAT"}, default "GCN"
        Type of graph convolution to use.
    dropout : float, default 0.1
        Dropout probability for regularization.
    use_batch_norm : bool, default True
        Whether to apply batch normalization after each layer.

    Attributes
    ----------
    convs : torch.nn.ModuleList
        List of graph convolutional layers.
    batch_norms : torch.nn.ModuleList or None
        List of batch normalization layers if enabled.
    embedding_dim : int
        Dimension of the final embeddings (last element of hidden_dims).
    pool_type : str
        Type of global pooling used ("mean" or "max").
    """

    def __init__(
        self,
        input_dim: int = 3,
        hidden_dims: list = [64, 32, 16],
        conv_type: str = "GCN",
        dropout: float = 0.1,
        use_batch_norm: bool = True,
    ):
        super(GNNEncoder, self).__init__()

        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.conv_type = conv_type
        self.dropout = dropout
        self.use_batch_norm = use_batch_norm

        # Create convolutional layers
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList() if use_batch_norm else None

        # Input layer
        if conv_type == "GCN":
            self.convs.append(GCNConv(input_dim, hidden_dims[0]))
        elif conv_type == "GAT":
            self.convs.append(GATConv(input_dim, hidden_dims[0], heads=4, concat=False))
        else:
            raise ValueError(f"Unsupported conv_type: {conv_type}")

        if use_batch_norm:
            self.batch_norms.append(nn.BatchNorm1d(hidden_dims[0]))

        # Hidden layers
        for i in range(len(hidden_dims) - 1):
            if conv_type == "GCN":
                self.convs.append(GCNConv(hidden_dims[i], hidden_dims[i + 1]))
            elif conv_type == "GAT":
                self.convs.append(GATConv(hidden_dims[i], hidden_dims[i + 1], heads=4, concat=False))

            if use_batch_norm:
                self.batch_norms.append(nn.BatchNorm1d(hidden_dims[i + 1]))

        # Global pooling for graph-level embeddings
        self.pool_type = "mean"  # Can be "mean", "max", or "attention"

        # Final embedding dimension
        self.embedding_dim = hidden_dims[-1]

    def forward(self, x, edge_index, edge_weight=None, batch=None):
        """
        Encode graph data into embeddings.

        Parameters
        ----------
        x : torch.Tensor
            Node features with shape (num_nodes, input_dim).
        edge_index : torch.Tensor
            Edge indices with shape (2, num_edges).
        edge_weight : torch.Tensor, optional
            Edge weights with shape (num_edges,). Only used with GCN layers.
        batch : torch.Tensor, optional
            Batch assignment vector with shape (num_nodes,) for batched graphs.

        Returns
        -------
        graph_embedding : torch.Tensor
            Graph-level embeddings with shape (batch_size, embedding_dim).
        node_embeddings : torch.Tensor
            Node-level embeddings with shape (num_nodes, embedding_dim).
        """
        # Apply graph convolutions
        for i, conv in enumerate(self.convs):
            if self.conv_type == "GCN":
                x = conv(x, edge_index, edge_weight)
            else:  # GAT doesn't use edge_weight in the same way
                x = conv(x, edge_index)

            if self.use_batch_norm and i < len(self.batch_norms):
                x = self.batch_norms[i](x)

            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        node_embeddings = x

        # Global pooling for graph-level representation
        if batch is None:
            # Single graph
            if self.pool_type == "mean":
                graph_embedding = torch.mean(x, dim=0, keepdim=True)
            elif self.pool_type == "max":
                graph_embedding = torch.max(x, dim=0, keepdim=True)[0]
        else:
            # Batched graphs
            if self.pool_type == "mean":
                graph_embedding = global_mean_pool(x, batch)
            elif self.pool_type == "max":
                graph_embedding = global_max_pool(x, batch)

        return graph_embedding, node_embeddings


class GNNDecoder(nn.Module):
    """
    Graph Neural Network Decoder for reconstructing breathing patterns.

    A PyTorch module that decodes graph-level embeddings back into the original
    node feature space, effectively reconstructing the breathing pattern data.

    Parameters
    ----------
    embedding_dim : int, default 16
        Dimension of input graph embeddings.
    hidden_dims : list of int, default [32, 64]
        List of hidden layer dimensions for the decoder architecture.
    output_dim : int, default 3
        Dimensionality of output node features (e.g., 3D coordinates).
    num_nodes : int, default 89
        Expected number of nodes in each graph.
    conv_type : {"GCN", "GAT"}, default "GCN"
        Type of graph convolution to use.
    dropout : float, default 0.1
        Dropout probability for regularization.
    use_batch_norm : bool, default True
        Whether to apply batch normalization after each layer.

    Attributes
    ----------
    embedding_expander : torch.nn.Linear
        Linear layer to expand graph embeddings to node-level features.
    convs : torch.nn.ModuleList
        List of graph convolutional layers.
    batch_norms : torch.nn.ModuleList or None
        List of batch normalization layers if enabled.
    """

    def __init__(
        self,
        embedding_dim: int = 16,
        hidden_dims: list = [32, 64],
        output_dim: int = 3,
        num_nodes: int = 89,
        conv_type: str = "GCN",
        dropout: float = 0.1,
        use_batch_norm: bool = True,
    ):
        super(GNNDecoder, self).__init__()

        self.embedding_dim = embedding_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.num_nodes = num_nodes
        self.conv_type = conv_type
        self.dropout = dropout
        self.use_batch_norm = use_batch_norm

        # Expand graph embedding to node-level features
        self.embedding_expander = nn.Linear(embedding_dim, num_nodes * hidden_dims[0])

        # Create deconvolutional layers
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList() if use_batch_norm else None

        # Build decoder layers (reverse of encoder)
        decoder_dims = hidden_dims + [output_dim]

        for i in range(len(decoder_dims) - 1):
            if conv_type == "GCN":
                self.convs.append(GCNConv(decoder_dims[i], decoder_dims[i + 1]))
            elif conv_type == "GAT":
                heads = 4 if i < len(decoder_dims) - 2 else 1  # Single head for output layer
                concat = False
                self.convs.append(GATConv(decoder_dims[i], decoder_dims[i + 1], heads=heads, concat=concat))

            if use_batch_norm and i < len(decoder_dims) - 2:  # No batch norm on output layer
                self.batch_norms.append(nn.BatchNorm1d(decoder_dims[i + 1]))

    def forward(self, graph_embedding, edge_index, edge_weight=None, batch=None):
        """
        Decode graph embeddings into reconstructed node features.

        Parameters
        ----------
        graph_embedding : torch.Tensor
            Graph-level embeddings with shape (batch_size, embedding_dim).
        edge_index : torch.Tensor
            Edge indices with shape (2, num_edges).
        edge_weight : torch.Tensor, optional
            Edge weights with shape (num_edges,). Only used with GCN layers.
        batch : torch.Tensor, optional
            Batch assignment vector with shape (num_nodes,) for batched graphs.

        Returns
        -------
        reconstructed_x : torch.Tensor
            Reconstructed node features with shape (num_nodes, output_dim).
        """
        batch_size = graph_embedding.size(0)

        # Expand graph embedding to node-level features
        x = self.embedding_expander(graph_embedding)  # [batch_size, num_nodes * hidden_dim]
        x = x.view(batch_size * self.num_nodes, self.hidden_dims[0])  # [batch_size * num_nodes, hidden_dim]

        # Apply graph deconvolutions
        for i, conv in enumerate(self.convs):
            if self.conv_type == "GCN":
                x = conv(x, edge_index, edge_weight)
            else:  # GAT doesn't use edge_weight in the same way
                x = conv(x, edge_index)

            # Apply batch norm and activation (except for last layer)
            if i < len(self.convs) - 1:
                if self.use_batch_norm and i < len(self.batch_norms):
                    x = self.batch_norms[i](x)
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)

        return x


class EdgeDecoder(nn.Module):
    """
    Edge Decoder for reconstructing graph connectivity.

    A PyTorch module that predicts edge existence probabilities based on
    node embeddings using a multi-layer perceptron.

    Parameters
    ----------
    node_embedding_dim : int, default 16
        Dimension of node embeddings.
    hidden_dim : int, default 64
        Hidden layer dimension for the edge prediction MLP.
    dropout : float, default 0.1
        Dropout probability for regularization.

    Attributes
    ----------
    edge_mlp : torch.nn.Sequential
        Multi-layer perceptron for edge prediction.
    """

    def __init__(self, node_embedding_dim: int = 16, hidden_dim: int = 64, dropout: float = 0.1):
        super(EdgeDecoder, self).__init__()

        self.node_embedding_dim = node_embedding_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout

        # Edge prediction layers
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * node_embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, node_embeddings, edge_index=None, num_nodes=None):
        """
        Predict edge existence probabilities from node embeddings.

        Parameters
        ----------
        node_embeddings : torch.Tensor
            Node embeddings with shape (num_nodes, embedding_dim).
        edge_index : torch.Tensor, optional
            Original edge indices for training mode. When provided, includes
            negative sampling.
        num_nodes : int, optional
            Number of nodes for generating all possible edges in inference mode.

        Returns
        -------
        edge_probs : torch.Tensor
            Edge existence probabilities with shape (num_edges, 1).
        predicted_edge_index : torch.Tensor
            Edge indices with shape (2, num_predicted_edges).
        """
        if num_nodes is None:
            num_nodes = node_embeddings.size(0)

        # Generate all possible edges (or use provided edge_index for training)
        if edge_index is not None:
            # Training mode: use provided edges + some negative samples
            pos_edges = edge_index
            neg_edges = negative_sampling(edge_index, num_nodes=num_nodes, num_neg_samples=edge_index.size(1))
            all_edges = torch.cat([pos_edges, neg_edges], dim=1)
        else:
            # Inference mode: generate all possible edges
            row, col = torch.meshgrid(
                torch.arange(num_nodes, device=node_embeddings.device),
                torch.arange(num_nodes, device=node_embeddings.device),
                indexing="ij",
            )
            all_edges = torch.stack([row.flatten(), col.flatten()], dim=0)
            # Remove self-loops
            mask = all_edges[0] != all_edges[1]
            all_edges = all_edges[:, mask]

        # Create edge features by concatenating node embeddings
        edge_features = torch.cat([node_embeddings[all_edges[0]], node_embeddings[all_edges[1]]], dim=1)

        # Predict edge probabilities
        edge_probs = self.edge_mlp(edge_features)

        return edge_probs, all_edges


class GNNAutoencoder(nn.Module):
    """
    Complete GNN Autoencoder for breathing pattern analysis and anomaly detection.

    A complete autoencoder architecture that combines graph neural network
    encoding and decoding for breathing pattern analysis. The model learns
    to compress breathing patterns into low-dimensional embeddings and
    reconstruct the original data, enabling anomaly detection through
    reconstruction error analysis.

    Parameters
    ----------
    input_dim : int, default 3
        Dimensionality of input node features (e.g., 3D coordinates).
    encoder_hidden_dims : list of int, default [64, 32, 16]
        Hidden layer dimensions for the encoder.
    decoder_hidden_dims : list of int, default [32, 64]
        Hidden layer dimensions for the decoder.
    num_nodes : int, default 89
        Expected number of nodes in each graph.
    conv_type : {"GCN", "GAT"}, default "GCN"
        Type of graph convolution to use.
    dropout : float, default 0.1
        Dropout probability for regularization.
    use_batch_norm : bool, default True
        Whether to apply batch normalization.
    reconstruction_weight : float, default 1.0
        Weight for the reconstruction loss component.
    edge_reconstruction_weight : float, default 0.5
        Weight for the edge reconstruction loss component.
    embedding_regularization_weight : float, default 0.01
        Weight for the embedding regularization loss component.
    reconstruct_edges : bool, default True
        Whether to include edge reconstruction in the model.

    Attributes
    ----------
    encoder : GNNEncoder
        Graph neural network encoder.
    decoder : GNNDecoder
        Graph neural network decoder.
    edge_decoder : EdgeDecoder, optional
        Edge reconstruction decoder (if reconstruct_edges=True).
    embedding_dim : int
        Dimension of the learned embeddings.
    """

    def __init__(
        self,
        input_dim: int = 3,
        encoder_hidden_dims: list = [64, 32, 16],
        decoder_hidden_dims: list = [32, 64],
        num_nodes: int = 89,
        conv_type: str = "GCN",
        dropout: float = 0.1,
        use_batch_norm: bool = True,
        reconstruction_weight: float = 1.0,
        edge_reconstruction_weight: float = 0.5,
        embedding_regularization_weight: float = 0.01,
        reconstruct_edges: bool = True,
    ):
        super(GNNAutoencoder, self).__init__()

        self.input_dim = input_dim
        self.num_nodes = num_nodes
        self.embedding_dim = encoder_hidden_dims[-1]
        self.reconstruction_weight = reconstruction_weight
        self.edge_reconstruction_weight = edge_reconstruction_weight
        self.embedding_regularization_weight = embedding_regularization_weight
        self.reconstruct_edges = reconstruct_edges

        # Encoder
        self.encoder = GNNEncoder(
            input_dim=input_dim,
            hidden_dims=encoder_hidden_dims,
            conv_type=conv_type,
            dropout=dropout,
            use_batch_norm=use_batch_norm,
        )

        # Decoder
        self.decoder = GNNDecoder(
            embedding_dim=encoder_hidden_dims[-1],
            hidden_dims=decoder_hidden_dims,
            output_dim=input_dim,
            num_nodes=num_nodes,
            conv_type=conv_type,
            dropout=dropout,
            use_batch_norm=use_batch_norm,
        )

        # Edge Decoder (optional)
        if self.reconstruct_edges:
            self.edge_decoder = EdgeDecoder(
                node_embedding_dim=encoder_hidden_dims[-1],
                hidden_dim=max(64, encoder_hidden_dims[-1] * 2),
                dropout=dropout,
            )

    def forward(self, x, edge_index, edge_weight=None, batch=None):
        """
        Forward pass through the complete autoencoder.

        Parameters
        ----------
        x : torch.Tensor
            Node features with shape (num_nodes, input_dim).
        edge_index : torch.Tensor
            Edge indices with shape (2, num_edges).
        edge_weight : torch.Tensor, optional
            Edge weights with shape (num_edges,).
        batch : torch.Tensor, optional
            Batch assignment vector with shape (num_nodes,) for batched graphs.

        Returns
        -------
        reconstructed_x : torch.Tensor
            Reconstructed node features with shape (num_nodes, input_dim).
        graph_embedding : torch.Tensor
            Graph-level embeddings with shape (batch_size, embedding_dim).
        node_embeddings : torch.Tensor
            Node-level embeddings with shape (num_nodes, embedding_dim).
        edge_probs : torch.Tensor, optional
            Edge reconstruction probabilities (returned if reconstruct_edges=True).
        reconstructed_edge_index : torch.Tensor, optional
            Reconstructed edge indices (returned if reconstruct_edges=True).
        anomaly_scores : torch.Tensor
            Anomaly scores based on reconstruction error with shape (batch_size,).
        """
        # Encode
        graph_embedding, node_embeddings = self.encoder(x, edge_index, edge_weight, batch)

        # Decode nodes
        reconstructed_x = self.decoder(graph_embedding, edge_index, edge_weight, batch)

        # Decode edges (if enabled)
        edge_probs = None
        reconstructed_edge_index = None
        if self.reconstruct_edges:
            edge_probs, reconstructed_edge_index = self.edge_decoder(
                node_embeddings,
                edge_index=edge_index if self.training else None,
                num_nodes=self.num_nodes,
            )

        # Compute anomaly scores based on reconstruction error
        reconstruction_error = torch.mean((reconstructed_x - x) ** 2, dim=-1)

        # Convert to anomaly scores using reconstruction error
        # Higher reconstruction error -> higher anomaly score
        if batch is not None:
            anomaly_scores = scatter(reconstruction_error, batch, dim=0, reduce="mean")
        else:
            # For single graph, compute overall anomaly score
            anomaly_scores = torch.mean(reconstruction_error).unsqueeze(0)

        if self.reconstruct_edges:
            return (
                reconstructed_x,
                graph_embedding,
                node_embeddings,
                edge_probs,
                reconstructed_edge_index,
                anomaly_scores,
            )
        else:
            return reconstructed_x, graph_embedding, node_embeddings, anomaly_scores

    def encode(self, x, edge_index, edge_weight=None, batch=None):
        """
        Extract embeddings without reconstruction.

        Parameters
        ----------
        x : torch.Tensor
            Node features with shape (num_nodes, input_dim).
        edge_index : torch.Tensor
            Edge indices with shape (2, num_edges).
        edge_weight : torch.Tensor, optional
            Edge weights with shape (num_edges,).
        batch : torch.Tensor, optional
            Batch assignment vector with shape (num_nodes,).

        Returns
        -------
        graph_embedding : torch.Tensor
            Graph-level embeddings with shape (batch_size, embedding_dim).
        node_embeddings : torch.Tensor
            Node-level embeddings with shape (num_nodes, embedding_dim).
        """
        with torch.no_grad():
            graph_embedding, node_embeddings = self.encoder(x, edge_index, edge_weight, batch)
        return graph_embedding, node_embeddings

    def compute_loss(
        self,
        x_original,
        x_reconstructed,
        graph_embedding,
        edge_index_original=None,
        edge_probs=None,
        reconstructed_edge_index=None,
        node_embeddings=None,
        reduction: str = "mean",
    ):
        """
        Compute multi-component autoencoder loss function.

        Parameters
        ----------
        x_original : torch.Tensor
            Original node features with shape (num_nodes, input_dim).
        x_reconstructed : torch.Tensor
            Reconstructed node features with shape (num_nodes, input_dim).
        graph_embedding : torch.Tensor
            Graph-level embeddings with shape (batch_size, embedding_dim).
        edge_index_original : torch.Tensor, optional
            Original edge indices for edge reconstruction loss.
        edge_probs : torch.Tensor, optional
            Edge reconstruction probabilities.
        reconstructed_edge_index : torch.Tensor, optional
            Reconstructed edge indices.
        node_embeddings : torch.Tensor, optional
            Node-level embeddings (currently unused).
        reduction : str, default "mean"
            Loss reduction method ("mean", "sum", or "none").

        Returns
        -------
        loss_dict : dict
            Dictionary containing loss components:
            - "total_loss": Combined weighted loss
            - "reconstruction_loss": MSE reconstruction loss
            - "edge_reconstruction_loss": Binary cross-entropy edge loss
            - "embedding_regularization_loss": L2 embedding regularization
        """
        # Reconstruction loss (MSE)
        reconstruction_loss = F.mse_loss(x_reconstructed, x_original, reduction=reduction)

        # Embedding regularization (L2 penalty to prevent overfitting)
        embedding_reg_loss = torch.norm(graph_embedding, p=2, dim=1).mean()

        # Edge reconstruction loss (if enabled)
        edge_reconstruction_loss = torch.tensor(0.0, device=reconstruction_loss.device)
        if self.reconstruct_edges and edge_probs is not None and edge_index_original is not None:
            # Create target labels for edge prediction
            num_pos_edges = edge_index_original.size(1)
            pos_labels = torch.ones(num_pos_edges, device=edge_probs.device)
            neg_labels = torch.zeros(edge_probs.size(0) - num_pos_edges, device=edge_probs.device)
            edge_labels = torch.cat([pos_labels, neg_labels], dim=0)

            # Binary cross-entropy loss for edge prediction
            edge_reconstruction_loss = F.binary_cross_entropy(edge_probs.squeeze(), edge_labels, reduction=reduction)

        # Total loss
        total_loss = self.reconstruction_weight * reconstruction_loss + self.edge_reconstruction_weight * edge_reconstruction_loss + self.embedding_regularization_weight * embedding_reg_loss

        loss_dict = {
            "total_loss": total_loss,
            "reconstruction_loss": reconstruction_loss,
            "edge_reconstruction_loss": edge_reconstruction_loss,
            "embedding_regularization_loss": embedding_reg_loss,
        }

        return loss_dict

    def compute_anomaly_score(self, x, edge_index, edge_weight=None, batch=None):
        """
        Compute anomaly scores based on reconstruction error.

        Parameters
        ----------
        x : torch.Tensor
            Node features with shape (num_nodes, input_dim).
        edge_index : torch.Tensor
            Edge indices with shape (2, num_edges).
        edge_weight : torch.Tensor, optional
            Edge weights with shape (num_edges,).
        batch : torch.Tensor, optional
            Batch assignment vector with shape (num_nodes,) for batched graphs.

        Returns
        -------
        anomaly_scores : float or numpy.ndarray
            Reconstruction-based anomaly scores. Returns a single float for
            single graphs or a numpy array for batched graphs.
        """
        self.eval()
        with torch.no_grad():
            forward_result = self.forward(x, edge_index, edge_weight, batch)
            if self.reconstruct_edges:
                reconstructed_x, _, _, _, _, _ = forward_result
            else:
                reconstructed_x, _, _, _ = forward_result

            # Compute reconstruction error per graph
            if batch is None:
                # Single graph
                mse_per_node = F.mse_loss(reconstructed_x, x, reduction="none").mean(dim=1)
                anomaly_score = mse_per_node.mean().item()
                return anomaly_score
            else:
                # Batched graphs
                mse_per_node = F.mse_loss(reconstructed_x, x, reduction="none").mean(dim=1)

                # Aggregate per graph
                unique_batch = torch.unique(batch)
                anomaly_scores = []

                for graph_id in unique_batch:
                    graph_mask = batch == graph_id
                    graph_mse = mse_per_node[graph_mask].mean()
                    anomaly_scores.append(graph_mse.item())

                return np.array(anomaly_scores)

    def get_breathing_pattern_features(self, x, edge_index, edge_weight=None, batch=None):
        """
        Extract comprehensive breathing pattern features from embeddings.

        Parameters
        ----------
        x : torch.Tensor
            Node features with shape (num_nodes, input_dim).
        edge_index : torch.Tensor
            Edge indices with shape (2, num_edges).
        edge_weight : torch.Tensor, optional
            Edge weights with shape (num_edges,).
        batch : torch.Tensor, optional
            Batch assignment vector with shape (num_nodes,) for batched graphs.

        Returns
        -------
        features_dict : dict
            Dictionary containing various breathing pattern features:
            - "graph_embedding": Graph-level embeddings
            - "embedding_norm": L2 norm of embeddings
            - "embedding_variance": Variance across embedding dimensions
            - "embedding_mean": Mean across embedding dimensions
            - Additional node-level statistics if available
        """
        graph_embedding, node_embeddings = self.encode(x, edge_index, edge_weight, batch)

        # Compute various features from embeddings
        features_dict = {
            "graph_embedding": graph_embedding.cpu().numpy(),
            "embedding_norm": torch.norm(graph_embedding, p=2, dim=1).cpu().numpy(),
            "embedding_variance": torch.var(graph_embedding, dim=1).cpu().numpy(),
            "embedding_mean": torch.mean(graph_embedding, dim=1).cpu().numpy(),
        }

        if node_embeddings is not None:
            features_dict.update(
                {
                    "node_embeddings": node_embeddings.cpu().numpy(),
                    "node_embedding_std": torch.std(node_embeddings, dim=0).cpu().numpy(),
                    "node_embedding_range": (torch.max(node_embeddings, dim=0)[0] - torch.min(node_embeddings, dim=0)[0]).cpu().numpy(),
                }
            )

        return features_dict

    def reconstruct_graph(self, x, edge_index, edge_weight=None, batch=None, edge_threshold=0.5):
        """
        Reconstruct the complete graph structure with features and edges.

        Parameters
        ----------
        x : torch.Tensor
            Node features with shape (num_nodes, input_dim).
        edge_index : torch.Tensor
            Original edge indices with shape (2, num_edges).
        edge_weight : torch.Tensor, optional
            Edge weights with shape (num_edges,).
        batch : torch.Tensor, optional
            Batch assignment vector with shape (num_nodes,) for batched graphs.
        edge_threshold : float, default 0.5
            Threshold for filtering edge reconstruction probabilities.

        Returns
        -------
        reconstructed_data : torch_geometric.data.Data
            Data object with reconstructed node features and edges.
        graph_embedding : torch.Tensor
            Graph-level embeddings.
        node_embeddings : torch.Tensor
            Node-level embeddings.
        anomaly_scores : torch.Tensor
            Anomaly scores based on reconstruction error.
        """
        self.eval()
        with torch.no_grad():
            forward_result = self.forward(x, edge_index, edge_weight, batch)

            if self.reconstruct_edges:
                (
                    reconstructed_x,
                    graph_embedding,
                    node_embeddings,
                    edge_probs,
                    reconstructed_edge_index,
                    anomaly_scores,
                ) = forward_result

                # Filter edges by threshold
                edge_mask = edge_probs.squeeze() > edge_threshold
                filtered_edges = reconstructed_edge_index[:, edge_mask]

                # Create new Data object
                from torch_geometric.data import Data

                reconstructed_data = Data(
                    x=reconstructed_x,
                    edge_index=filtered_edges,
                    edge_attr=edge_probs[edge_mask],
                    batch=batch,
                )

                return (
                    reconstructed_data,
                    graph_embedding,
                    node_embeddings,
                    anomaly_scores,
                )
            else:
                (
                    reconstructed_x,
                    graph_embedding,
                    node_embeddings,
                    anomaly_scores,
                ) = forward_result

                # Use original edges if not reconstructing
                from torch_geometric.data import Data

                reconstructed_data = Data(x=reconstructed_x, edge_index=edge_index, batch=batch)

                return (
                    reconstructed_data,
                    graph_embedding,
                    node_embeddings,
                    anomaly_scores,
                )
