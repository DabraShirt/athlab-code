from __future__ import annotations

import argparse
import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

# Optuna imports
import optuna
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from optuna import create_study
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

from ml.datasets.gnn_vest_dataset import GNNVestDataset
from ml.models.GNN_autoencoder import GNNAutoencoder
from ml.utils.data_leakage_validator import run_comprehensive_validation

# Local imports
from ml.utils.emt_pandas import Emt3DAccessor, EmtAccessor  # noqa: F401
from ml.utils.logging_utils import get_logger
from ml.utils.mlflow_utils import MLflowTracker

LOGGER_BASE_NAME = "athlab"


def load_yaml_config(path):
    """Load YAML configuration file."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_configs(
    model_config_path: str,
    training_config_path: str,
    model_config_name: str = "base_gnn_autoencoder",
    training_config_name: str = "default_training",
):
    """
    Load model and training configurations.

    Parameters
    ----------
    model_config_path : str
        Path to model configuration file
    training_config_path : str
        Path to training configuration file
    model_config_name : str, default="base_gnn_autoencoder"
        Name of model configuration to use
    training_config_name : str, default="default_training"
        Name of training configuration to use

    Returns
    -------
    dict
        Merged configuration dictionary
    """
    # Load model config
    model_configs = load_yaml_config(model_config_path)

    # Load training config
    training_configs = load_yaml_config(training_config_path)

    # Select specific configurations
    if model_config_name not in model_configs:
        print(f"WARNING: Model config '{model_config_name}' not found, using 'base_gnn_autoencoder'")
        model_config_name = "base_gnn_autoencoder"

    model_config = model_configs[model_config_name]

    # Handle nested training config structure
    if training_config_name in training_configs:
        training_config = training_configs[training_config_name]
    else:
        print(f"WARNING: Training config '{training_config_name}' not found, using 'default_training'")
        if "default_training" in training_configs:
            training_config = training_configs["default_training"]
        else:
            # Use the entire config if no nested structure
            training_config = training_configs

    # Merge configurations
    config = {"model": model_config, "training": training_config}

    return config


class GNNAutoencoderTrainer:
    """
    GNN Autoencoder trainer with MLflow integration.

    Parameters
    ----------
    model : nn.Module
        The GNN autoencoder model to train
    device : torch.device
        Device to run training on
    config : dict
        Configuration dictionary
    mlflow_logger : MLflowTracker, optional
        MLflow logger for experiment tracking
    """

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        config: Dict,
        mlflow_logger: MLflowTracker = None,
    ):
        self.model = model
        self.device = device
        self.config = config
        self.mlflow_logger = mlflow_logger
        self.logger = get_logger(f"{LOGGER_BASE_NAME}.trainer")

        # Setup optimizer and scheduler
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config["training"].get("learning_rate", 0.001),
            weight_decay=config["training"].get("weight_decay", 1e-5),
        )

        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode="min", factor=0.7, patience=5)

        # Training tracking
        self.train_losses = []
        self.val_losses = []
        self.val_aucs = []
        self.epochs = []

        # Loss component tracking
        self.train_reconstruction_losses = []
        self.train_regularization_losses = []
        self.val_reconstruction_losses = []
        self.val_regularization_losses = []

    def save_checkpoint(
        self,
        epoch: int,
        val_loss: float,
        val_score: float,
        is_best: bool = False,
        checkpoint_dir: str = "checkpoints",
    ):
        """
        Save model checkpoint during training.

        Parameters
        ----------
        epoch : int
            Current epoch number
        val_loss : float
            Validation loss
        val_score : float
            Validation discrimination score
        is_best : bool, default=False
            Whether this is the best checkpoint so far
        checkpoint_dir : str, default="checkpoints"
            Directory to save checkpoints
        """
        os.makedirs(checkpoint_dir, exist_ok=True)

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "val_loss": val_loss,
            "val_score": val_score,
            "config": self.config,
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "val_aucs": self.val_aucs,
            "train_reconstruction_losses": self.train_reconstruction_losses,
            "train_regularization_losses": self.train_regularization_losses,
            "val_reconstruction_losses": self.val_reconstruction_losses,
            "val_regularization_losses": self.val_regularization_losses,
        }

        # Save regular checkpoint
        checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch}.pt")
        torch.save(checkpoint, checkpoint_path)

        # Save best checkpoint if this is the best
        if is_best:
            best_path = os.path.join(checkpoint_dir, "best_checkpoint.pt")
            torch.save(checkpoint, best_path)
            self.logger.info(f" New best checkpoint saved at epoch {epoch}: score = {val_score:.6f}")

        return checkpoint_path

    def train(self, train_loader, val_loader, num_epochs: int = 20) -> Dict:
        """
        Train the model.

        Parameters
        ----------
        train_loader : DataLoader
            Training data loader
        val_loader : DataLoader
            Validation data loader
        num_epochs : int, default=100
            Number of training epochs

        Returns
        -------
        dict
            Training history
        """
        self.model.train()
        best_val_score = -float("inf")  # Track best discrimination score

        for epoch in range(num_epochs):
            # Training phase
            train_loss = self._train_epoch(train_loader)

            # Validation phase
            val_loss, val_discrimination_score = self._validate_epoch(val_loader)

            # Update scheduler
            self.scheduler.step(val_loss)

            # Store history
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.val_aucs.append(val_discrimination_score)  # Keep same variable name for compatibility
            self.epochs.append(epoch)

            # Check if this is the best model so far
            is_best = val_discrimination_score > best_val_score
            if is_best:
                best_val_score = val_discrimination_score

            # Save checkpoint every 10 epochs or if it's the best
            if epoch % 10 == 0 or is_best or epoch == num_epochs - 1:
                self.save_checkpoint(epoch, val_loss, val_discrimination_score, is_best)

            # Log metrics
            if self.mlflow_logger:
                metrics_dict = {
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "val_discrimination_score": val_discrimination_score,
                    "learning_rate": self.optimizer.param_groups[0]["lr"],
                    "best_val_score": best_val_score,
                }

                # Add loss components if available
                if self.train_reconstruction_losses and len(self.train_reconstruction_losses) > 0:
                    metrics_dict.update(
                        {
                            "train_reconstruction_loss": self.train_reconstruction_losses[-1],
                            "train_regularization_loss": self.train_regularization_losses[-1],
                            "val_reconstruction_loss": self.val_reconstruction_losses[-1],
                            "val_regularization_loss": self.val_regularization_losses[-1],
                        }
                    )

                self.mlflow_logger.log_metrics(metrics_dict, step=epoch)

            # Print detailed metrics every 10 epochs or if there's improvement
            if epoch % 10 == 0 or is_best:
                self.logger.info(f"Epoch {epoch}: Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}, " f"Val Discrimination: {val_discrimination_score:.6f} " f"{' NEW BEST!' if is_best else ''}")

        return {
            "train_loss": self.train_losses,
            "val_loss": self.val_losses,
            "val_auc": self.val_aucs,
            "epoch": self.epochs,
            "best_val_score": best_val_score,
            "train_reconstruction_loss": self.train_reconstruction_losses,
            "train_regularization_loss": self.train_regularization_losses,
            "val_reconstruction_loss": self.val_reconstruction_losses,
            "val_regularization_loss": self.val_regularization_losses,
        }

    def _train_epoch(self, train_loader) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        total_recon_loss = 0
        total_reg_loss = 0
        num_batches = 0

        for batch in train_loader:
            batch = batch.to(self.device)

            self.optimizer.zero_grad()

            # Forward pass
            try:
                # Call model with correct parameter order: (x, edge_index, edge_weight=None, batch)
                (
                    reconstructed,
                    graph_embedding,
                    node_embeddings,
                    anomaly_scores,
                ) = self.model(batch.x, batch.edge_index, None, batch.batch)

            except Exception as e:
                print(f" Model forward error: {e}")
                print(f" batch.x is None: {batch.x is None}")
                print(f" batch.edge_index is None: {batch.edge_index is None}")
                print(f" batch.batch is None: {batch.batch is None}")
                raise

            # Use model's compute_loss method for MSE + L2 regularization
            loss_dict = self.model.compute_loss(
                x_original=batch.x,
                x_reconstructed=reconstructed,
                graph_embedding=graph_embedding,
                edge_index_original=batch.edge_index,
                node_embeddings=node_embeddings,
            )
            loss = loss_dict["total_loss"]

            # Log detailed loss components periodically
            if num_batches % 50 == 0:  # Log every 50 batches
                self.logger.debug(f"   Loss components - Total: {loss.item():.6f}, " f"Recon: {loss_dict['reconstruction_loss'].item():.6f}, " f"L2 Reg: {loss_dict['embedding_regularization_loss'].item():.6f}")

            # Backward pass
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            total_recon_loss += loss_dict["reconstruction_loss"].item()
            total_reg_loss += loss_dict["embedding_regularization_loss"].item()
            num_batches += 1

        # Store average loss components for this epoch
        if num_batches > 0:
            self.train_reconstruction_losses.append(total_recon_loss / num_batches)
            self.train_regularization_losses.append(total_reg_loss / num_batches)

        return total_loss / num_batches if num_batches > 0 else 0

    def _validate_epoch(self, val_loader) -> Tuple[float, float]:
        """Validate for one epoch."""
        self.model.eval()
        total_loss = 0
        total_recon_loss = 0
        total_reg_loss = 0
        num_batches = 0

        all_errors = []
        all_reconstruction_errors = []

        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(self.device)

                # Forward pass with correct parameter order
                (
                    reconstructed,
                    graph_embedding,
                    node_embeddings,
                    anomaly_scores,
                ) = self.model(batch.x, batch.edge_index, None, batch.batch)

                # Use model's compute_loss method for MSE + L2 regularization
                loss_dict = self.model.compute_loss(
                    x_original=batch.x,
                    x_reconstructed=reconstructed,
                    graph_embedding=graph_embedding,
                    edge_index_original=batch.edge_index,
                    node_embeddings=node_embeddings,
                )
                loss = loss_dict["total_loss"].item()

                # Get both types of errors for analysis
                anomaly_scores_np = anomaly_scores.cpu().numpy()
                reconstruction_mse = nn.MSELoss(reduction="none")(reconstructed, batch.x).mean(dim=[1]).cpu().numpy()

                all_errors.extend(anomaly_scores_np)
                all_reconstruction_errors.extend(reconstruction_mse)

                total_loss += loss
                total_recon_loss += loss_dict["reconstruction_loss"].item()
                total_reg_loss += loss_dict["embedding_regularization_loss"].item()
                num_batches += 1

        # Store average loss components for this epoch
        if num_batches > 0:
            self.val_reconstruction_losses.append(total_recon_loss / num_batches)
            self.val_regularization_losses.append(total_reg_loss / num_batches)

        avg_loss = total_loss / num_batches if num_batches > 0 else 0

        # For autoencoders without ground truth labels, use reconstruction quality metrics
        # Higher variance in anomaly scores indicates better discrimination capability
        error_variance = np.var(all_errors) if len(all_errors) > 0 else 0
        error_mean = np.mean(all_errors) if len(all_errors) > 0 else 0
        error_std = np.std(all_errors) if len(all_errors) > 0 else 0

        # Debug information every few epochs
        if len(self.val_losses) % 10 == 0:
            self.logger.info(f"    Validation - Total Loss: {avg_loss:.6f}")
            self.logger.info(f"    Anomaly scores - Mean: {error_mean:.6f}, Std: {error_std:.6f}, Var: {error_variance:.6f}")
            self.logger.info(f"    Sample scores: {all_errors[:5] if len(all_errors) >= 5 else all_errors}")
            # Log loss components for the last batch
            if "loss_dict" in locals():
                self.logger.info(f"    Loss components - Recon: {loss_dict['reconstruction_loss'].item():.6f}, " f"L2 Reg: {loss_dict['embedding_regularization_loss'].item():.6f}")

        # Use error variance as a proxy for model's discrimination ability
        # Higher variance = better at distinguishing different patterns
        discrimination_score = min(error_variance / (error_mean + 1e-8), 1.0) if error_mean > 0 else 0

        return avg_loss, discrimination_score

    def evaluate_model(self, test_loader, save_results: bool = True) -> Dict:
        """
        Evaluate the trained model.

        Parameters
        ----------
        test_loader : DataLoader
            Test data loader
        save_results : bool, default=True
            Whether to save results

        Returns
        -------
        dict
            Evaluation results
        """
        self.model.eval()
        all_errors = []
        all_reconstruction_errors = []
        all_embeddings = []

        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(self.device)

                # Forward pass with correct parameter order
                (
                    reconstructed,
                    graph_embedding,
                    node_embeddings,
                    anomaly_scores,
                ) = self.model(batch.x, batch.edge_index, None, batch.batch)

                # Collect various metrics
                anomaly_scores_np = anomaly_scores.cpu().numpy()
                reconstruction_mse = nn.MSELoss(reduction="none")(reconstructed, batch.x).mean(dim=[1]).cpu().numpy()
                embeddings = graph_embedding.cpu().numpy()

                all_errors.extend(anomaly_scores_np)
                all_reconstruction_errors.extend(reconstruction_mse)
                all_embeddings.extend(embeddings)

        # Calculate comprehensive metrics for unsupervised anomaly detection
        all_errors = np.array(all_errors)
        all_reconstruction_errors = np.array(all_reconstruction_errors)
        all_embeddings = np.array(all_embeddings)

        # Anomaly detection quality metrics
        error_variance = np.var(all_errors)
        error_mean = np.mean(all_errors)
        error_std = np.std(all_errors)

        # Reconstruction quality
        recon_variance = np.var(all_reconstruction_errors)
        recon_mean = np.mean(all_reconstruction_errors)

        # Embedding quality (how well separated are the embeddings)
        embedding_variance = np.var(all_embeddings, axis=0).mean()

        # Combined discrimination score
        discrimination_score = min(error_variance / (error_mean + 1e-8), 1.0) if error_mean > 0 else 0

        results = {
            "discrimination_score": discrimination_score,
            "anomaly_score_mean": error_mean,
            "anomaly_score_std": error_std,
            "anomaly_score_variance": error_variance,
            "reconstruction_error_mean": recon_mean,
            "reconstruction_error_variance": recon_variance,
            "embedding_variance": embedding_variance,
            "num_samples": len(all_errors),
        }

        if save_results and self.mlflow_logger:
            self.mlflow_logger.log_metrics(
                {
                    "test_discrimination_score": discrimination_score,
                    "test_anomaly_mean": error_mean,
                    "test_anomaly_std": error_std,
                    "test_recon_mean": recon_mean,
                    "test_embedding_var": embedding_variance,
                }
            )

        return results


class OptunaTuner:
    """
    Optuna hyperparameter tuner that uses pre-prepared dataset for efficiency.

    Parameters
    ----------
    config : dict
        Base configuration
    dataset : GNNVestDataset
        Pre-prepared dataset
    graphs : list
        Pre-created graphs
    num_nodes : int
        Number of nodes per graph
    device : torch.device
        Training device
    n_trials : int, default=100
        Number of optimization trials
    mlflow_tracker : MLflowTracker, optional
        MLflow tracker for logging
    """

    def __init__(
        self,
        config: Dict,
        dataset: GNNVestDataset,
        graphs: List,
        num_nodes: int,
        device: torch.device,
        n_trials: int = 5,
        mlflow_tracker: MLflowTracker = None,
        dataset_mode: str = "timestep",
    ):
        self.config = config
        self.dataset = dataset
        self.graphs = graphs
        self.num_nodes = num_nodes
        self.device = device
        self.n_trials = n_trials
        self.mlflow_tracker = mlflow_tracker
        self.dataset_mode = dataset_mode
        self.logger = get_logger(f"{LOGGER_BASE_NAME}.optuna")
        self.dataloaders_cache = {}  # Cache for different batch sizes

    def get_dataloaders(self, batch_size: int, split_ratio: List[float]):
        """
        Get or create dataloaders for a given batch size.
        Cache them to avoid recreation across trials.

        Parameters
        ----------
        batch_size : int
            Batch size for dataloaders
        split_ratio : list of float
            Train/val/test split ratios

        Returns
        -------
        tuple
            (train_loader, val_loader, test_loader)
        """
        cache_key = (batch_size, tuple(split_ratio))

        if cache_key not in self.dataloaders_cache:
            self.logger.info(f" Creating new dataloaders for batch_size={batch_size}")

            try:
                # Use the dataset's configured splitting strategy to prevent data leakage
                subject_aware = getattr(self.dataset, "use_subject_aware_splits", False)
                recording_aware = getattr(self.dataset, "use_recording_aware_splits", True)

                self.logger.info(f" Using {'subject-aware' if subject_aware else 'recording-aware' if recording_aware else 'random'} splits to prevent data leakage")

                train_loader, val_loader, test_loader = self.dataset.create_dataloaders(
                    batch_size=batch_size,
                    split_ratio=split_ratio,
                    subject_aware=subject_aware,
                    recording_aware=recording_aware,
                )

                self.dataloaders_cache[cache_key] = (
                    train_loader,
                    val_loader,
                    test_loader,
                )

                self.logger.info(f" Dataloaders cached - Train: {len(train_loader.dataset)} graphs ({len(train_loader)} batches), " f"Val: {len(val_loader.dataset)} graphs ({len(val_loader)} batches), " f"Test: {len(test_loader.dataset)} graphs ({len(test_loader)} batches)")

            except Exception as e:
                self.logger.error(f"Error creating data loaders: {e}")
                raise
        else:
            self.logger.info(f"  Using cached dataloaders for batch_size={batch_size}")

        return self.dataloaders_cache[cache_key]

    def objective(self, trial: optuna.trial.Trial) -> float:
        """
        Optuna objective function using pre-prepared dataset.

        Parameters
        ----------
        trial : optuna.trial.Trial
            Optuna trial object

        Returns
        -------
        float
            Objective value (validation AUC)
        """
        # Suggest hyperparameters
        # Adjust batch size suggestions based on dataset mode
        if self.dataset_mode == "timestep":
            # For timestep mode, prefer batch_size=1 to avoid temporal leakage
            self.logger.warning("  Timestep mode detected: Using batch_size=1 to prevent temporal leakage")
            batch_size_options = [1]
        elif self.dataset_mode == "sequence":
            # For sequence mode, use batch_size=1 to avoid edge batching issues
            self.logger.warning("  Sequence mode detected: Using batch_size=1 to avoid edge batching complexity")
            batch_size_options = [1]
        else:
            # For whole_file mode, normal batch sizes should be fine (but usually only 1 graph anyway)
            batch_size_options = [1, 4, 8]

        params = {
            "model": {
                "input_dim": self.config["model"].get("input_dim", 3),
                "hidden_dims": [
                    trial.suggest_int("hidden_dim_1", 32, 128),
                    trial.suggest_int("hidden_dim_2", 16, 64),
                    trial.suggest_int("hidden_dim_3", 8, 32),
                ],
                "dropout": trial.suggest_float("dropout", 0.1, 0.5),
                "gnn_type": trial.suggest_categorical("gnn_type", ["GCN"]),  # Debug: GCN only
                "batch_norm": trial.suggest_categorical("batch_norm", [True, False]),
            },
            "training": {
                "learning_rate": trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True),
                "batch_size": trial.suggest_categorical("batch_size", batch_size_options),
                "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True),
                "num_epochs": 50,  # Reduced for faster optimization
            },
        }

        # Update config with suggested parameters
        trial_config = self.config.copy()
        trial_config.update(params)

        # Get data split configuration
        split_ratio = [
            1.0 - trial_config["training"].get("test_split", 0.15) - trial_config["training"].get("validation_split", 0.15),
            trial_config["training"].get("validation_split", 0.15),
            trial_config["training"].get("test_split", 0.15),
        ]

        # Get or create cached dataloaders for this batch size
        batch_size = trial_config["training"]["batch_size"]
        train_loader, val_loader, test_loader = self.get_dataloaders(batch_size, split_ratio)

        try:
            # Start MLflow run for this trial (nested under main experiment)
            if self.mlflow_tracker:
                run_id = self.mlflow_tracker.start_run(run_name=f"trial_{trial.number}", nested=True)
                self.logger.info(f"Started MLflow run {run_id} for trial {trial.number}")

            # Train model with suggested parameters using cached dataloaders
            result = train_with_cached_dataloaders(
                config=trial_config,
                train_loader=train_loader,
                val_loader=val_loader,
                test_loader=test_loader,
                num_nodes=self.num_nodes,
                device=self.device,
                mlflow_tracker=self.mlflow_tracker,
            )

            # Extract validation discrimination score (stored in val_auc for compatibility)
            if result and "history" in result:
                val_scores = result["history"].get("val_auc", [])  # This now contains discrimination scores
                if val_scores:
                    best_discrimination_score = max(val_scores)
                    # Store the model state in trial for later retrieval
                    trial.set_user_attr("best_model_state_dict", result["model"].state_dict())
                    trial.set_user_attr("model_config", trial_config["model"])
                    trial.set_user_attr("training_config", trial_config["training"])
                    trial.set_user_attr("best_score", best_discrimination_score)
                else:
                    best_discrimination_score = 0.0
            else:
                best_discrimination_score = 0.0

            # Log trial results to MLflow
            if self.mlflow_tracker:
                self.mlflow_tracker.log_params({f"trial_{trial.number}_{k}": v for k, v in params.items() if isinstance(v, (int, float, str))})
                self.mlflow_tracker.log_metrics({f"trial_{trial.number}_discrimination_score": best_discrimination_score})
                # End MLflow run for this trial
                self.mlflow_tracker.end_run()

            self.logger.info(f"Trial {trial.number}: Discrimination Score = {best_discrimination_score:.6f}")
            return best_discrimination_score

        except Exception as e:
            self.logger.error(f"Trial {trial.number} failed: {e}")
            # End MLflow run even on failure
            if self.mlflow_tracker:
                self.mlflow_tracker.end_run()
            return 0.0

    def optimize(self) -> Tuple[Dict, float, optuna.study.Study]:
        """
        Run Optuna optimization.

        Returns
        -------
        tuple
            Best parameters, best value, and study object
        """
        # Create study
        study = create_study(
            direction="maximize",
            sampler=TPESampler(seed=42),
            pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=10),
        )

        # Optimize
        self.logger.info(f"Starting optimization with {self.n_trials} trials...")
        study.optimize(self.objective, n_trials=self.n_trials)

        self.logger.info("Optimization completed!")
        self.logger.info(f"Best trial: {study.best_trial.number}")
        self.logger.info(f"Best value: {study.best_value}")
        self.logger.info(f"Best params: {study.best_params}")

        return study.best_params, study.best_value, study


def prepare_dataset(
    data_paths: List[str],
    config: Dict,
    dataset_mode: str = "timestep",
    sequence_length: int = 10,
    sequence_step: int = 1,
) -> Tuple[GNNVestDataset, List, int]:
    """
    Prepare dataset once for multiple training runs.

    Parameters
    ----------
    data_paths : list of str
        List of data paths (can be config file path or individual EMT files)
    config : dict
        Configuration dictionary
    dataset_mode : str, default='timestep'
        Dataset mode ('timestep', 'sequence', 'whole_file')
    sequence_length : int, default=10
        Sequence length for sequence mode

    Returns
    -------
    tuple
        (dataset, graphs, num_nodes)
    """
    logger = get_logger(f"{LOGGER_BASE_NAME}.data")

    # Check if the first path is a config file (YAML)
    if len(data_paths) == 1 and data_paths[0].endswith(".yaml"):
        logger.info(f" Using multi-dataset configuration: {data_paths[0]}")

        from ml.utils.multi_dataset_loader import prepare_training_data

        # Use the multi-dataset loader
        dataset, graphs, num_nodes = prepare_training_data(
            config_path=data_paths[0],
            max_files=None,  # Load all files
            dataset_mode=dataset_mode,
            sequence_length=sequence_length if dataset_mode == "sequence" else None,
            sequence_step=sequence_step,
        )

        logger.info(f" Multi-dataset prepared: {len(graphs)} graphs with {num_nodes} nodes each")
        logger.info(f"Dataset info: {dataset.file_info}")

        return dataset, graphs, num_nodes

    else:
        # Original single-dataset approach for backwards compatibility
        logger.info(" Using single-dataset approach (legacy)")

        # Process data paths to find EMT files
        processed_data_paths = []
        for path_str in data_paths:
            path = Path(path_str)
            if path.is_dir():
                # Find all 3D Point Tracks.emt files recursively
                emt_files = list(path.glob("**/3D Point Tracks.emt"))
                processed_data_paths.extend([str(f) for f in emt_files])
            elif path.suffix == ".emt":
                processed_data_paths.append(path_str)
            else:
                logger.warning(f"Skipping non-EMT file: {path_str}")

        if not processed_data_paths:
            logger.error(f"No EMT files found in directories: {data_paths}")
            for path_str in data_paths:
                path = Path(path_str)
                if path.is_dir():
                    logger.info(f"Contents of {path_str}: {list(path.iterdir())}")
            raise ValueError("No EMT files found in the provided data paths")

        logger.info(f"Found {len(processed_data_paths)} EMT files")

        # Load data from EMT files
        try:
            # Load the first EMT file as example (in real scenario, you'd combine multiple files)
            # TODO: Combine multiple EMT files if needed
            if processed_data_paths:

                # Load 3D point tracks data
                first_file = processed_data_paths[0]
                logger.info(f"Loading data from: {first_file}")

                # Load the dataframe using EMT accessor
                dataframe = pd.DataFrame.emt3d.from_emt(first_file)
                logger.info(f"Loaded dataframe with shape: {dataframe.shape}")

            else:
                raise ValueError("No data files found")

        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise

        # Prepare data using the dataset's built-in methods
        try:
            # Initialize dataset with specified mode for GNN autoencoder training
            dataset = GNNVestDataset(
                dataframe=dataframe,
                mode=dataset_mode,  # Use the specified dataset mode
                sequence_length=sequence_length if dataset_mode == "sequence" else None,
                sequence_step=sequence_step if dataset_mode == "sequence" else 1,
                test_size=config["training"].get("test_split", 0.15),
                val_size=config["training"].get("validation_split", 0.15),
                scaler_type=config["training"].get("scaler_type", "standard"),
                validate_data=True,
            )

            # Preprocess the data to extract node features
            dataset.preprocess_data(apply_scaling=True)

            # Create graphs using the specified mode
            graphs = dataset.create_graphs()

            # Log mode-specific information
            logger.info(f"Dataset mode: {dataset_mode}")
            if dataset_mode == "sequence":
                logger.info(f"Sequence length: {sequence_length}")
            elif dataset_mode == "whole_file":
                logger.info("Using entire file as single graph")

            # Add dummy labels for anomaly detection (in practice, these would be real labels)
            # Set random seed for reproducible pseudo-anomalies across trials
            # This is currently not used since we are training AE
            np.random.seed(42)
            for i, graph in enumerate(graphs):
                # Generate pseudo-anomalies: mark 10% of timesteps as anomalous
                is_anomaly = np.random.random() < 0.1
                graph.y = torch.tensor([1 if is_anomaly else 0], dtype=torch.long)

            # Get the actual number of nodes from the dataset
            sample_graph = graphs[0] if graphs else None
            num_nodes = sample_graph.x.shape[0] if sample_graph is not None else 89

            logger.info(f"Dataset prepared with {len(graphs)} graphs and {num_nodes} nodes per graph")
            logger.info(f"Mode: {dataset_mode}, Graphs: {len(graphs)}, Nodes per graph: {num_nodes}")

            return dataset, graphs, num_nodes

        except Exception as e:
            logger.error(f"Error creating dataset: {e}")
            raise


def train_with_prepared_data(
    config: Dict,
    dataset: GNNVestDataset,
    graphs: List,
    num_nodes: int,
    device: torch.device,
    mlflow_tracker: MLflowTracker = None,
) -> Dict:
    """
    Train a model with pre-prepared dataset and given configuration.

    Parameters
    ----------
    config : dict
        Complete configuration dictionary
    dataset : GNNVestDataset
        Pre-prepared dataset
    graphs : list
        Pre-created graphs
    num_nodes : int
        Number of nodes per graph
    device : torch.device
        Training device
    mlflow_tracker : MLflowTracker, optional
        MLflow tracker for logging

    Returns
    -------
    dict
        Training results dictionary containing model, trainer, results, and config
    """
    logger = get_logger(f"{LOGGER_BASE_NAME}.train_run")

    # Create data loaders with current config
    try:
        # Create data loaders using built-in dataset method
        split_ratio = [
            1.0 - config["training"].get("test_split", 0.15) - config["training"].get("validation_split", 0.15),
            config["training"].get("validation_split", 0.15),
            config["training"].get("test_split", 0.15),
        ]

        batch_size = config["training"].get("batch_size", 16)

        # Use the dataset's configured splitting strategy to prevent data leakage
        subject_aware = getattr(dataset, "use_subject_aware_splits", False)
        recording_aware = getattr(dataset, "use_recording_aware_splits", True)

        logger.info(f" Using {'subject-aware' if subject_aware else 'recording-aware' if recording_aware else 'random'} splits to prevent data leakage")

        train_loader, val_loader, test_loader = dataset.create_dataloaders(
            batch_size=batch_size,
            split_ratio=split_ratio,
            subject_aware=subject_aware,
            recording_aware=recording_aware,
        )

    except Exception as e:
        logger.error(f"Error creating data loaders: {e}")
        return {}

    logger.info(f"Created data loaders - Train: {len(train_loader)}, Val: {len(val_loader)}, Test: {len(test_loader)} batches")

    # CRITICAL: Validate data splits to ensure no leakage
    logger.info("🔍 Validating data splits for leakage...")
    try:
        validation_results = run_comprehensive_validation(
            train_loader,
            val_loader,
            test_loader,
            check_temporal=(dataset.mode == "timestep"),
            verbose=True,
        )

        if not validation_results["overall_valid"]:
            logger.error("❌ Data validation failed! Training may produce misleading results.")
            logger.error("Please fix data leakage issues before proceeding.")
        else:
            logger.info("✅ Data validation passed - no leakage detected!")

    except Exception as e:
        logger.warning(f"⚠️  Could not validate data splits: {e}")
        logger.warning("Proceeding with training, but results may not be reliable.")

    # Initialize model
    try:
        model = GNNAutoencoder(
            input_dim=config["model"]["input_dim"],
            encoder_hidden_dims=config["model"]["hidden_dims"],
            decoder_hidden_dims=config["model"]["hidden_dims"][::-1][1:] + [config["model"]["input_dim"]],
            num_nodes=num_nodes,  # Use actual number of nodes from dataset
            conv_type=config["model"].get("gnn_type", "GCN"),
            dropout=config["model"]["dropout"],
            use_batch_norm=config["model"].get("batch_norm", True),
            reconstruct_edges=False,  # For training, focus on node reconstruction only
        )
        model = model.to(device)

        total_params = sum(p.numel() for p in model.parameters())
        logger.info(f"Model created with {total_params:,} total parameters")
        logger.info(f"Using {num_nodes} nodes per graph")

    except Exception as e:
        logger.error(f"Error creating model: {e}")
        raise

    # Setup trainer with MLflow logging
    trainer = GNNAutoencoderTrainer(model=model, device=device, config=config, mlflow_logger=mlflow_tracker)

    # Train model
    logger.info("Starting training...")
    history = trainer.train(train_loader, val_loader, config["training"].get("num_epochs", 100))

    # Evaluate
    logger.info("Evaluating model...")
    results = trainer.evaluate_model(test_loader, save_results=True)

    # Calculate and log custom metrics
    custom_metrics = calculate_training_metrics(history, results)
    if mlflow_tracker:
        mlflow_tracker.log_metrics(custom_metrics)

    return {
        "model": model,
        "trainer": trainer,
        "results": results,
        "config": config,
        "history": history,
        "custom_metrics": custom_metrics,
    }


def train_with_config(
    config: Dict,
    data_paths: List[str],
    device: torch.device,
    dataset_mode: str = "timestep",
    sequence_length: int = 10,
    sequence_step: int = 1,
) -> Dict:
    """
    Train a model with given configuration (legacy function for backwards compatibility).

    Parameters
    ----------
    config : dict
        Complete configuration dictionary
    data_paths : list of str
        List of data paths
    device : torch.device
        Training device
    dataset_mode : str, default='timestep'
        Dataset mode ('timestep', 'sequence', 'whole_file')
    sequence_length : int, default=10
        Sequence length for sequence mode
    sequence_step : int, default=1
        Step size for sequence sliding window

    Returns
    -------
    dict
        Training results dictionary containing model, trainer, results, and config
    """
    # Initialize MLflow tracker
    mlflow_tracker = MLflowTracker("GNN_Autoencoder_Training")

    # Prepare dataset once
    dataset, graphs, num_nodes = prepare_dataset(
        data_paths,
        config,
        dataset_mode=dataset_mode,
        sequence_length=sequence_length,
        sequence_step=sequence_step,
    )

    # Train with prepared data
    return train_with_prepared_data(config, dataset, graphs, num_nodes, device, mlflow_tracker)


def calculate_training_metrics(history: Dict, results: Dict) -> Dict[str, float]:
    """
    Calculate custom training metrics for MLflow logging.

    Parameters
    ----------
    history : dict
        Training history dictionary
    results : dict
        Evaluation results dictionary

    Returns
    -------
    dict
        Custom metrics dictionary
    """
    if not history.get("val_loss"):
        return {}

    metrics = {}

    # Basic performance metrics
    metrics["final_train_loss"] = history["train_loss"][-1] if history["train_loss"] else 0
    metrics["final_val_loss"] = history["val_loss"][-1] if history["val_loss"] else 0
    metrics["min_val_loss"] = min(history["val_loss"]) if history["val_loss"] else 0
    metrics["max_discrimination_score"] = max(history["val_auc"]) if history["val_auc"] else 0  # val_auc now stores discrimination scores

    # Training stability metrics
    val_losses = np.array(history["val_loss"])
    if len(val_losses) > 1:
        metrics["val_loss_std"] = np.std(val_losses)
        metrics["val_loss_trend"] = np.polyfit(range(len(val_losses)), val_losses, 1)[0]  # slope

    # Convergence metrics
    if history["val_loss"]:
        best_epoch = np.argmin(history["val_loss"])
        metrics["epochs_to_best"] = best_epoch + 1
        metrics["improvement_ratio"] = (history["val_loss"][0] - metrics["min_val_loss"]) / history["val_loss"][0]
        metrics["training_efficiency"] = metrics["improvement_ratio"] / len(history["epoch"]) if history["epoch"] else 0

    # Overfitting detection
    if history["train_loss"] and history["val_loss"]:
        metrics["overfitting_score"] = max(0, metrics["final_val_loss"] - metrics["final_train_loss"])
        metrics["generalization_gap"] = metrics["final_val_loss"] - metrics["final_train_loss"]

    # Model complexity metrics
    metrics["total_epochs"] = len(history["epoch"]) if history["epoch"] else 0

    # Add evaluation metrics if available
    if "discrimination_score" in results:
        metrics["test_discrimination_score"] = results["discrimination_score"]
    if "anomaly_score_mean" in results:
        metrics["test_anomaly_mean"] = results["anomaly_score_mean"]
    if "reconstruction_error_mean" in results:
        metrics["test_reconstruction_error"] = results["reconstruction_error_mean"]

    return metrics


def show_available_configs():
    """Show available model and training configurations."""
    print(" Available Model Configurations (configs/models/gnn_autoencoder.yaml):")
    try:
        model_configs = load_yaml_config("configs/models/gnn_autoencoder.yaml")
        for name, config in model_configs.items():
            hidden_dims = config.get("hidden_dims", "N/A")
            gnn_type = config.get("gnn_type", "N/A")
            print(f"  • {name}: {hidden_dims} dims, {gnn_type} type")
    except Exception as e:
        print(f"   Error loading model configs: {e}")

    print("\n Available Training Configurations (configs/training/gnn_autoencoder_training.yaml):")
    try:
        training_configs = load_yaml_config("configs/training/gnn_autoencoder_training.yaml")
        for name, config in training_configs.items():
            if isinstance(config, dict):
                epochs = config.get("num_epochs", "N/A")
                batch_size = config.get("batch_size", "N/A")
                print(f"  • {name}: {epochs} epochs, batch size {batch_size}")
    except Exception as e:
        print(f"   Error loading training configs: {e}")


def main():
    """
    Main training entry point with Optuna optimization.
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Train GNN Autoencoder with Optuna optimization")
    parser.add_argument(
        "--model-config",
        default="base_gnn_autoencoder",
        help="Model configuration name (from gnn_autoencoder.yaml)",
    )
    parser.add_argument(
        "--training-config",
        default="default_training",
        help="Training configuration name (from gnn_autoencoder_training.yaml)",
    )
    parser.add_argument(
        "--model-config-path",
        default="configs/models/gnn_autoencoder.yaml",
        help="Path to model configuration file",
    )
    parser.add_argument(
        "--training-config-path",
        default="configs/training/gnn_autoencoder_training.yaml",
        help="Path to training configuration file",
    )
    parser.add_argument("--data-paths", nargs="+", default=None, help="List of data paths")
    parser.add_argument("--n-trials", type=int, default=10, help="Number of Optuna trials")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick training mode (10 trials, 20 epochs)",
    )
    parser.add_argument("--no-optimize", action="store_true", help="Skip hyperparameter optimization")
    parser.add_argument(
        "--list-configs",
        action="store_true",
        help="Show available configurations and exit",
    )
    parser.add_argument(
        "--dataset-mode",
        default="timestep",
        choices=["timestep", "sequence", "whole_file"],
        help="Dataset mode: timestep (each timestep as graph), sequence (sliding window), whole_file (entire file as one graph)",
    )
    parser.add_argument(
        "--sequence-length",
        type=int,
        default=10,
        help="Sequence length for sequence mode (default: 10)",
    )
    parser.add_argument(
        "--sequence-step",
        type=int,
        default=1,
        help="Step size for sequence sliding window (default: 1, use >1 to reduce overlap)",
    )
    parser.add_argument(
        "--force-batch-size-1",
        action="store_true",
        help="Force batch size to 1 for timestep/sequence modes to avoid batching issues",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Device to use: auto (detect), cuda (force GPU), cpu (force CPU)",
    )

    args = parser.parse_args()

    if args.list_configs:
        show_available_configs()
        return

    logger = get_logger(f"{LOGGER_BASE_NAME}.cli")

    # Load configurations
    logger.info(f"Loading model config: {args.model_config} from {args.model_config_path}")
    logger.info(f"Loading training config: {args.training_config} from {args.training_config_path}")

    try:
        config = load_configs(
            model_config_path=args.model_config_path,
            training_config_path=args.training_config_path,
            model_config_name=args.model_config,
            training_config_name=args.training_config,
        )
        logger.info(" Configurations loaded successfully")
        logger.info(f"Model: {args.model_config}, Training: {args.training_config}")
    except Exception as e:
        logger.error(f" Error loading configurations: {e}")
        raise

    # Device selection with diagnostics
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    logger.info(f"Using device: {device}")

    # GPU diagnostics
    if device.type == "cuda":
        logger.info("CUDA available: True")
        logger.info(f"CUDA device: {torch.cuda.get_device_name(0)}")
        logger.info(f"CUDA version: {torch.version.cuda}")
        logger.info(f"CUDNN version: {torch.backends.cudnn.version()}")
        logger.info(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        if torch.cuda.is_available():
            logger.warning("CUDA available but not using it. Use --device cuda to force GPU usage")
        logger.info("Running on CPU - consider using GPU for faster training")

    # Data paths - using multi-dataset configuration by default
    if args.data_paths is None:
        data_paths = ["configs/data_splits.yaml"]  # Use multi-dataset configuration
    else:
        data_paths = args.data_paths

    logger.info(f"Using data configuration: {data_paths}")

    # Quick mode adjustments
    if args.quick:
        # Only set n_trials to 10 if user didn't specify a custom value
        if args.n_trials == 1:  # 100 is the default value
            args.n_trials = 1
        config["training"]["num_epochs"] = 15
        logger.info(f" Quick mode: {args.n_trials} trials, 20 epochs")

    if args.no_optimize:
        # Direct training without optimization
        logger.info(" Direct training mode (no hyperparameter optimization)")
        results = train_with_config(
            config,
            data_paths,
            device,
            dataset_mode=args.dataset_mode,
            sequence_length=args.sequence_length,
            sequence_step=args.sequence_step,
        )
        logger.info("Training completed successfully!")
        return results

    # Initialize MLflow
    experiment_name = f"GNN_Autoencoder_{args.model_config}_{args.training_config}_{args.dataset_mode}"
    if args.dataset_mode == "sequence":
        experiment_name += f"_seq{args.sequence_length}"
        if args.sequence_step > 1:
            experiment_name += f"_step{args.sequence_step}"
    mlflow_tracker = MLflowTracker(experiment_name)

    # Prepare dataset once for all trials (this is the key optimization!)
    logger.info(" Preparing dataset once for all optimization trials...")
    logger.info(f"Dataset mode: {args.dataset_mode}")
    if args.dataset_mode == "sequence":
        logger.info(f"Sequence length: {args.sequence_length}")
        logger.info(f"Sequence step: {args.sequence_step}")

    dataset, graphs, num_nodes = prepare_dataset(
        data_paths,
        config,
        dataset_mode=args.dataset_mode,
        sequence_length=args.sequence_length,
        sequence_step=args.sequence_step,
    )
    logger.info(f" Dataset prepared: {len(graphs)} graphs with {num_nodes} nodes each")

    # Log mode-specific information and recommendations
    if args.dataset_mode == "timestep":
        logger.info(" Timestep mode: Each timestep becomes a separate graph")
        if args.force_batch_size_1:
            logger.info(" Using batch_size=1 to prevent temporal leakage")
            config["training"]["batch_size"] = 1
        else:
            logger.warning("     Consider using --force-batch-size-1 or switch to sequence mode")
    elif args.dataset_mode == "sequence":
        if args.force_batch_size_1:
            logger.info(" Force using batch_size=1 to prevent edge batching issues")
            config["training"]["batch_size"] = 1
    elif args.dataset_mode == "sequence":
        logger.info(f" Sequence mode: Sliding windows of {args.sequence_length} timesteps")
        logger.info(f"   • Graph structure: {num_nodes} nodes = {89} vest points × {args.sequence_length} timesteps")
        logger.info(" Using batch_size=1 to avoid edge batching complexity")
    elif args.dataset_mode == "whole_file":
        logger.info(" Whole file mode: Entire recording as one large graph")

    # Validate model configuration for the dataset mode
    sample_graph = graphs[0] if graphs else None
    if sample_graph is not None:
        num_edges = sample_graph.edge_index.shape[1] if hasattr(sample_graph, "edge_index") else 0
        logger.info(f" Sample graph: {sample_graph.x.shape[0]} nodes, {num_edges} edges")

        # Warn about potential memory issues
        if args.dataset_mode == "whole_file" and num_nodes > 5000:
            logger.warning("  Large graph detected! Consider reducing sequence length or using sequence mode")
        elif args.dataset_mode == "sequence" and num_nodes > 2000:
            logger.warning(f"  Large sequence graphs! Consider reducing sequence length from {args.sequence_length}")

    logger.info(f" Ready for optimization with {len(graphs)} graphs")

    # Start a parent MLflow run for the entire optimization
    if mlflow_tracker:
        # Add timestamp to run name to avoid confusion with previous runs
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        parent_run_id = mlflow_tracker.start_run(
            run_name=f"optuna_optimization_{args.n_trials}_trials_{timestamp}",
            nested=False,
        )
        logger.info(f"Started parent MLflow run: {parent_run_id}")
        logger.info(f"Run name: optuna_optimization_{args.n_trials}_trials_{timestamp}")

        # Log experiment configuration
        mlflow_tracker.log_params(
            {
                "n_trials": args.n_trials,
                "dataset_mode": args.dataset_mode,
                "sequence_length": (args.sequence_length if args.dataset_mode == "sequence" else None),
                "sequence_step": (args.sequence_step if args.dataset_mode == "sequence" else None),
                "num_graphs": len(graphs),
                "num_nodes": num_nodes,
                "timestamp": timestamp,
            }
        )

    try:
        # Create and run Optuna tuner with pre-prepared data
        tuner = OptunaTuner(
            config=config,
            dataset=dataset,
            graphs=graphs,
            num_nodes=num_nodes,
            device=device,
            n_trials=args.n_trials,
            mlflow_tracker=mlflow_tracker,
            dataset_mode=args.dataset_mode,
        )

        logger.info(f" Starting Optuna hyperparameter optimization with {args.n_trials} trials...")
        logger.info(" Using pre-prepared dataset for faster optimization!")
        best_params, best_value, study = tuner.optimize()

        logger.info(" Optimization completed!")
        logger.info(f"Best parameters: {best_params}")
        logger.info(f"Best validation AUC: {best_value}")

        # Get the best model from the study instead of retraining
        logger.info(" Loading best model from optimization trials...")

        # Create model with best parameters
        final_config = config.copy()
        final_config["model"].update(
            {
                "input_dim": config["model"].get("input_dim", 3),
                "hidden_dims": [
                    study.best_params["hidden_dim_1"],
                    study.best_params["hidden_dim_2"],
                    study.best_params["hidden_dim_3"],
                ],
                "dropout": study.best_params["dropout"],
                "gnn_type": study.best_params["gnn_type"],
                "batch_norm": study.best_params["batch_norm"],
            }
        )

        final_config["training"].update(
            {
                "learning_rate": study.best_params["learning_rate"],
                "batch_size": study.best_params["batch_size"],
                "weight_decay": study.best_params["weight_decay"],
            }
        )

        # Create model architecture
        best_model = GNNAutoencoder(
            input_dim=final_config["model"]["input_dim"],
            encoder_hidden_dims=final_config["model"]["hidden_dims"],
            decoder_hidden_dims=final_config["model"]["hidden_dims"][::-1][1:] + [final_config["model"]["input_dim"]],
            num_nodes=num_nodes,
            conv_type=final_config["model"].get("gnn_type", "GCN"),
            dropout=final_config["model"]["dropout"],
            use_batch_norm=final_config["model"].get("batch_norm", True),
            reconstruct_edges=False,  # For training, focus on node reconstruction only
        )

        # Load the best model state from the study
        if study.best_trial.user_attrs.get("best_model_state_dict"):
            best_model.load_state_dict(study.best_trial.user_attrs["best_model_state_dict"])
            logger.info(f" Loaded best model from trial {study.best_trial.number}")
        else:
            logger.warning(" No model state found in best trial, using untrained model")

        # Create a simple results dictionary for compatibility
        final_results = {
            "model": best_model,
            "results": {"auc": best_value, "discrimination_score": best_value},
            "history": {"val_auc": [best_value]},
            "config": final_config,
            "custom_metrics": {"best_discrimination_score": best_value},
        }

        # Log final results and save best model
        if mlflow_tracker:
            mlflow_tracker.log_params(best_params)
            mlflow_tracker.log_metrics({"final_test_auc": final_results["results"].get("auc", 0)})

            # Save complete model checkpoint like in train_1D_from_3D_gnn.py
            if "model" in final_results:
                # Create comprehensive model checkpoint
                model_name = f"best_gnn_autoencoder_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                checkpoint = {
                    "model_state_dict": final_results["model"].state_dict(),
                    "model_config": final_config["model"],
                    "training_config": final_config["training"],
                    "best_params": best_params,
                    "best_value": best_value,
                    "dataset_info": {
                        "num_nodes": num_nodes,
                        "num_graphs": len(graphs) if graphs else 0,
                        "dataset_mode": args.dataset_mode,
                        "sequence_length": (args.sequence_length if args.dataset_mode == "sequence" else None),
                    },
                    "results": final_results["results"],
                    "custom_metrics": final_results.get("custom_metrics", {}),
                    "training_history": final_results.get("history", {}),
                    "study_info": {
                        "n_trials": len(study.trials),
                        "best_trial_number": study.best_trial.number,
                        "optimization_direction": "maximize",
                    },
                }

                # Save to local file
                checkpoint_path = f"{model_name}.pt"
                torch.save(checkpoint, checkpoint_path)
                logger.info(f" Saved complete model checkpoint to: {checkpoint_path}")

                # Log the checkpoint as MLflow artifact
                with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
                    torch.save(checkpoint, f.name)
                    mlflow_tracker.log_artifact(f.name)

                # Also log just the state_dict for compatibility
                with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
                    torch.save(final_results["model"].state_dict(), f.name)
                    mlflow_tracker.log_artifact(f.name)

                logger.info(" Model artifacts logged to MLflow")
                logger.info(f" Best model performance: discrimination score = {best_value:.6f}")

                # Print model loading instructions for exploration notebook
                logger.info("=" * 60)
                logger.info(" To load this model in the exploration notebook:")
                logger.info(f"   model_path = '{checkpoint_path}'")
                logger.info("   checkpoint = torch.load(model_path, map_location='cpu')")
                logger.info("   model.load_state_dict(checkpoint['model_state_dict'])")
                logger.info("=" * 60)

    finally:
        # Always end the parent MLflow run, even if there was an error
        if mlflow_tracker:
            mlflow_tracker.end_run()
            logger.info("Ended parent MLflow run")

    # Final summary
    logger.info(" Training Summary:")
    logger.info(f"   • Dataset mode: {args.dataset_mode}")
    if args.dataset_mode == "sequence":
        logger.info(f"   • Sequence length: {args.sequence_length}")
    logger.info(f"   • Total graphs: {len(graphs) if graphs else 0}")
    logger.info(f"   • Nodes per graph: {num_nodes}")
    logger.info(f"   • Best discrimination score: {best_value:.6f}")
    logger.info(f"   • Optimization trials: {len(study.trials) if study else 0}")

    logger.info(" Training completed successfully!")
    return final_results


def train_with_cached_dataloaders(
    config: Dict,
    train_loader,
    val_loader,
    test_loader,
    num_nodes: int,
    device: torch.device,
    mlflow_tracker: MLflowTracker = None,
) -> Dict:
    """
    Train a model with pre-created dataloaders (optimized for hyperparameter trials).

    Parameters
    ----------
    config : dict
        Complete configuration dictionary
    train_loader : DataLoader
        Pre-created training dataloader
    val_loader : DataLoader
        Pre-created validation dataloader
    test_loader : DataLoader
        Pre-created test dataloader
    num_nodes : int
        Number of nodes per graph
    device : torch.device
        Training device
    mlflow_tracker : MLflowTracker, optional
        MLflow tracker for logging

    Returns
    -------
    dict
        Training results dictionary containing model, trainer, results, and config
    """
    logger = get_logger(f"{LOGGER_BASE_NAME}.cached_train")

    logger.info(f"Using cached dataloaders - Train: {len(train_loader)} batches, " f"Val: {len(val_loader)} batches, Test: {len(test_loader)} batches")

    # Initialize model
    try:
        model = GNNAutoencoder(
            input_dim=config["model"]["input_dim"],
            encoder_hidden_dims=config["model"]["hidden_dims"],
            decoder_hidden_dims=config["model"]["hidden_dims"][::-1][1:] + [config["model"]["input_dim"]],
            num_nodes=num_nodes,  # Use actual number of nodes from dataset
            conv_type=config["model"].get("gnn_type", "GCN"),
            dropout=config["model"]["dropout"],
            use_batch_norm=config["model"].get("batch_norm", True),
            reconstruct_edges=False,  # For training, focus on node reconstruction only
        )
        model = model.to(device)

        total_params = sum(p.numel() for p in model.parameters())
        logger.info(f"Model created with {total_params:,} total parameters")
        logger.info(f"Using {num_nodes} nodes per graph")

    except Exception as e:
        logger.error(f"Error creating model: {e}")
        raise

    # Setup trainer with MLflow logging
    trainer = GNNAutoencoderTrainer(model=model, device=device, config=config, mlflow_logger=mlflow_tracker)

    # Train model
    logger.info("Starting training...")
    history = trainer.train(train_loader, val_loader, config["training"].get("num_epochs", 100))

    # Evaluate
    logger.info("Evaluating model...")
    results = trainer.evaluate_model(test_loader, save_results=True)

    # Calculate and log custom metrics
    custom_metrics = calculate_training_metrics(history, results)
    if mlflow_tracker:
        mlflow_tracker.log_metrics(custom_metrics)

    return {
        "model": model,
        "trainer": trainer,
        "results": results,
        "config": config,
        "history": history,
        "custom_metrics": custom_metrics,
    }


if __name__ == "__main__":
    main()
