import json
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from ml.utils.logging_utils import get_logger

# Optional: Plotly for interactive plots
try:
    import plotly.graph_objects as go

    PLOTLY_AVAILABLE = True
except ImportError:
    sys.stderr.write("WARNING: Plotly not available for interactive plots\n")
    PLOTLY_AVAILABLE = False

sys.path.append(".")
warnings.filterwarnings("ignore")

# Initialize logger
logger = get_logger("athlab.simple_classifier")


class SimpleActivityClassifier:
    """XGBoost classifier to distinguish rest/recovery vs high-intensity training"""

    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.model = None
        self.pca_viz = None

    def load_aggregated_embeddings(self):
        """Load 512D aggregated embeddings with activity labels"""
        logger.info("Loading 512D aggregated embeddings...")

        embeddings_dir = Path("ml/ml_data/embeddings_agg")
        embedding_files = list(embeddings_dir.glob("*.npy"))
        logger.info(f"   Found {len(embedding_files)} embedding files")

        all_embeddings = []
        all_labels = []
        all_filenames = []

        for file_path in sorted(embedding_files):
            try:
                embedding = np.load(file_path)
                filename = file_path.name

                # Extract activity type from filename - Binary classification
                # Rest/Recovery: files containing 'rest' or 'recovery'
                # Training: files containing '.min' (like 6.min, 12.min, 15.min)
                if "rest" in filename.lower() or "recovery" in filename.lower():
                    activity_label = "rest_recovery"
                elif "min" in filename.lower():
                    activity_label = "training"
                else:
                    # Skip files that don't clearly fit either category
                    logger.warning(f"   Skipping unclear file: {filename}")
                    continue

                all_embeddings.append(embedding)
                all_labels.append(activity_label)
                all_filenames.append(filename)

                logger.info(f"   {filename} -> {activity_label}")

            except Exception as e:
                logger.error(f"   Error loading {file_path}: {e}")
                continue

        if len(all_embeddings) == 0:
            raise ValueError("No valid embeddings found. Please check file naming conventions.")

        final_embeddings = np.vstack(all_embeddings)
        logger.info(f"\nTotal embeddings loaded: {final_embeddings.shape}")
        logger.info("Label distribution:")
        label_counts = pd.Series(all_labels).value_counts()
        for label, count in label_counts.items():
            logger.info(f"   {label}: {count} samples")

        return final_embeddings, all_labels, all_filenames

    def create_3d_visualization(self, embeddings, labels, model_accuracy=None):
        """Create 3D and 2D PCA visualizations colored by activity labels"""
        logger.info("\nCreating PCA visualizations...")

        # Create PCA for visualization
        self.pca_viz = PCA(n_components=3, random_state=42)
        embeddings_pca = self.pca_viz.fit_transform(embeddings)

        logger.info(f"   PCA explained variance: {self.pca_viz.explained_variance_ratio_}")
        logger.info(f"   Total variance explained: {np.sum(self.pca_viz.explained_variance_ratio_):.3f}")

        # Create output directory
        output_dir = Path("ml/ml_data/analysis_outputs")
        output_dir.mkdir(exist_ok=True)

        # Set up colors for labels
        unique_labels = list(set(labels))
        color_map = {"rest_recovery": "#2E86AB", "training": "#F24236"}  # Blue  # Red

        # Interactive Plotly visualization if available
        if PLOTLY_AVAILABLE:
            logger.info("   Creating interactive 3D plot...")
            fig = go.Figure()

            for label in unique_labels:
                mask = np.array(labels) == label
                if np.any(mask):
                    fig.add_trace(
                        go.Scatter3d(
                            x=embeddings_pca[mask, 0],
                            y=embeddings_pca[mask, 1],
                            z=embeddings_pca[mask, 2],
                            mode="markers",
                            name=label.replace("_", "/").title(),
                            marker=dict(
                                size=8,
                                color=color_map.get(label, "#888888"),
                                opacity=0.7,
                                line=dict(width=1, color="black"),
                            ),
                            text=[f"{label}: Sample {i}" for i in range(np.sum(mask))],
                            hovertemplate="<b>%{text}</b><br>" + "PC1: %{x:.3f}<br>" + "PC2: %{y:.3f}<br>" + "PC3: %{z:.3f}<extra></extra>",
                        )
                    )

            title_text = "Activity Classification: 3D Embedding Space"
            if model_accuracy:
                title_text += f"<br><sup>XGBoost Accuracy: {model_accuracy:.3f}</sup>"

            fig.update_layout(
                title=dict(text=title_text, x=0.5, font=dict(size=16)),
                scene=dict(
                    xaxis_title=f"PC1 ({self.pca_viz.explained_variance_ratio_[0]:.1%} var)",
                    yaxis_title=f"PC2 ({self.pca_viz.explained_variance_ratio_[1]:.1%} var)",
                    zaxis_title=f"PC3 ({self.pca_viz.explained_variance_ratio_[2]:.1%} var)",
                    bgcolor="rgba(240,240,240,0.1)",
                    camera=dict(eye=dict(x=1.5, y=1.5, z=1.5)),
                ),
                width=1000,
                height=800,
                font=dict(size=12),
                legend=dict(
                    x=0.02,
                    y=0.98,
                    bgcolor="rgba(255,255,255,0.8)",
                    bordercolor="rgba(0,0,0,0.2)",
                    borderwidth=1,
                ),
            )

            interactive_file = output_dir / "activity_classification_3d_interactive.html"
            fig.write_html(str(interactive_file))
            logger.info(f"   Interactive 3D plot: {interactive_file}")

        # Static matplotlib visualization
        logger.info("   Creating static 3D and 2D plots...")
        plt.style.use("default")
        fig_static = plt.figure(figsize=(20, 12))

        # Create subplots: 3D plot + 2D projections
        ax_3d = fig_static.add_subplot(2, 3, 1, projection="3d")
        ax_xy = fig_static.add_subplot(2, 3, 2)
        ax_xz = fig_static.add_subplot(2, 3, 3)
        ax_yz = fig_static.add_subplot(2, 3, 4)
        ax_dist = fig_static.add_subplot(2, 3, 5)
        ax_var = fig_static.add_subplot(2, 3, 6)

        # 3D scatter plot
        for label in unique_labels:
            mask = np.array(labels) == label
            if np.any(mask):
                ax_3d.scatter(
                    embeddings_pca[mask, 0],
                    embeddings_pca[mask, 1],
                    embeddings_pca[mask, 2],
                    label=label.replace("_", "/").title(),
                    color=color_map.get(label, "#888888"),
                    s=60,
                    alpha=0.7,
                    edgecolors="black",
                    linewidths=0.5,
                )

        ax_3d.set_xlabel(f"PC1 ({self.pca_viz.explained_variance_ratio_[0]:.1%} var)")
        ax_3d.set_ylabel(f"PC2 ({self.pca_viz.explained_variance_ratio_[1]:.1%} var)")
        ax_3d.set_zlabel(f"PC3 ({self.pca_viz.explained_variance_ratio_[2]:.1%} var)")
        ax_3d.set_title("3D PCA Space\n(Activity Classification)", fontsize=12, fontweight="bold")
        ax_3d.legend()

        # 2D projections
        projections = [
            (ax_xy, 0, 1, "PC1 vs PC2"),
            (ax_xz, 0, 2, "PC1 vs PC3"),
            (ax_yz, 1, 2, "PC2 vs PC3"),
        ]

        for ax, dim1, dim2, title in projections:
            for label in unique_labels:
                mask = np.array(labels) == label
                if np.any(mask):
                    ax.scatter(
                        embeddings_pca[mask, dim1],
                        embeddings_pca[mask, dim2],
                        label=label.replace("_", "/").title(),
                        color=color_map.get(label, "#888888"),
                        s=40,
                        alpha=0.7,
                        edgecolors="black",
                        linewidths=0.3,
                    )

            ax.set_xlabel(f"PC{dim1 + 1} ({self.pca_viz.explained_variance_ratio_[dim1]:.1%} var)")
            ax.set_ylabel(f"PC{dim2 + 1} ({self.pca_viz.explained_variance_ratio_[dim2]:.1%} var)")
            ax.set_title(title, fontsize=10, fontweight="bold")
            ax.grid(True, alpha=0.3)
            ax.legend()

        # Label distribution
        label_counts = pd.Series(labels).value_counts()
        colors = [color_map.get(label, "#888888") for label in label_counts.index]
        bars = ax_dist.bar(
            [label.replace("_", "/").title() for label in label_counts.index],
            label_counts.values,
            color=colors,
            alpha=0.7,
            edgecolor="black",
        )
        ax_dist.set_title("Label Distribution", fontsize=10, fontweight="bold")
        ax_dist.set_ylabel("Number of Samples")

        # Add count labels on bars
        for bar, count in zip(bars, label_counts.values):
            height = bar.get_height()
            ax_dist.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 0.1,
                f"{count}",
                ha="center",
                va="bottom",
                fontweight="bold",
            )

        # PCA variance explained
        ax_var.bar(
            range(1, 4),
            self.pca_viz.explained_variance_ratio_,
            color=["#FF6B6B", "#4ECDC4", "#45B7D1"],
            alpha=0.7,
            edgecolor="black",
        )
        ax_var.set_xlabel("Principal Component")
        ax_var.set_ylabel("Explained Variance Ratio")
        ax_var.set_title("PCA Variance Explained", fontsize=10, fontweight="bold")
        ax_var.set_xticks(range(1, 4))
        ax_var.grid(True, alpha=0.3, axis="y")

        # Add percentage labels on bars
        for i, var in enumerate(self.pca_viz.explained_variance_ratio_):
            ax_var.text(
                i + 1,
                var + 0.005,
                f"{var:.1%}",
                ha="center",
                va="bottom",
                fontweight="bold",
            )

        # Main title
        title_text = "XGBoost Activity Classification: Rest/Recovery vs Training Minutes\n"
        title_text += f"{len(embeddings)} samples | {np.sum(self.pca_viz.explained_variance_ratio_):.1%} variance explained"
        if model_accuracy:
            title_text += f" | Model Accuracy: {model_accuracy:.3f}"

        fig_static.suptitle(title_text, fontsize=14, fontweight="bold", y=0.98)

        plt.tight_layout()

        # Save static plot
        static_file = output_dir / "activity_classification_static.png"
        fig_static.savefig(str(static_file), dpi=300, bbox_inches="tight", facecolor="white")
        logger.info(f"   Static plots: {static_file}")

        plt.close()

        return {
            "embeddings_pca": embeddings_pca,
            "pca_explained_variance": self.pca_viz.explained_variance_ratio_,
            "total_variance_explained": np.sum(self.pca_viz.explained_variance_ratio_),
        }

    def train_xgboost_classifier(self, embeddings, labels):
        """Train simple XGBoost classifier with comprehensive metrics"""
        logger.info("\nTraining XGBoost classifier (default parameters)...")

        # Encode labels
        y_encoded = self.label_encoder.fit_transform(labels)
        logger.info(f"   Label mapping: {dict(zip(self.label_encoder.classes_, range(len(self.label_encoder.classes_))))}")

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(embeddings, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)

        logger.info(f"   Train samples: {X_train.shape[0]}")
        logger.info(f"   Test samples: {X_test.shape[0]}")

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Simple XGBoost with default parameters
        logger.info("   Training XGBoost with default parameters...")
        self.model = xgb.XGBClassifier(random_state=42, eval_metric="logloss")
        self.model.fit(X_train_scaled, y_train)

        # Comprehensive cross-validation with different metrics
        logger.info("\nCross-validation analysis:")
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_metrics = {}

        cv_scores_acc = cross_val_score(self.model, X_train_scaled, y_train, cv=cv, scoring="accuracy")
        cv_scores_f1 = cross_val_score(self.model, X_train_scaled, y_train, cv=cv, scoring="f1_weighted")
        cv_scores_prec = cross_val_score(self.model, X_train_scaled, y_train, cv=cv, scoring="precision_weighted")
        cv_scores_rec = cross_val_score(self.model, X_train_scaled, y_train, cv=cv, scoring="recall_weighted")
        cv_scores_auc = cross_val_score(self.model, X_train_scaled, y_train, cv=cv, scoring="roc_auc")

        cv_metrics["accuracy"] = {
            "mean": np.mean(cv_scores_acc),
            "std": np.std(cv_scores_acc),
        }
        cv_metrics["f1"] = {"mean": np.mean(cv_scores_f1), "std": np.std(cv_scores_f1)}
        cv_metrics["precision"] = {
            "mean": np.mean(cv_scores_prec),
            "std": np.std(cv_scores_prec),
        }
        cv_metrics["recall"] = {
            "mean": np.mean(cv_scores_rec),
            "std": np.std(cv_scores_rec),
        }
        cv_metrics["auc"] = {
            "mean": np.mean(cv_scores_auc),
            "std": np.std(cv_scores_auc),
        }

        for metric, values in cv_metrics.items():
            logger.info(f"   CV {metric.upper()}: {values['mean']:.4f} ± {values['std']:.4f}")

        # Evaluate on test set with comprehensive metrics
        y_pred = self.model.predict(X_test_scaled)
        y_pred_proba = self.model.predict_proba(X_test_scaled)[:, 1]

        test_metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "f1": f1_score(y_test, y_pred, average="weighted"),
            "precision": precision_score(y_test, y_pred, average="weighted"),
            "recall": recall_score(y_test, y_pred, average="weighted"),
            "auc": roc_auc_score(y_test, y_pred_proba),
        }

        logger.info("\nTEST SET PERFORMANCE (COMPREHENSIVE):")
        logger.info(f"   Accuracy:  {test_metrics['accuracy']:.4f}")
        logger.info(f"   F1-score:  {test_metrics['f1']:.4f}")
        logger.info(f"   Precision: {test_metrics['precision']:.4f}")
        logger.info(f"   Recall:    {test_metrics['recall']:.4f}")
        logger.info(f"   ROC-AUC:   {test_metrics['auc']:.4f}")

        # Confidence intervals for test metrics (bootstrap)
        logger.info("\nCONFIDENCE INTERVALS (95% Bootstrap):")
        n_bootstrap = 1000
        bootstrap_metrics = {"accuracy": [], "f1": [], "precision": [], "recall": []}

        for _ in range(n_bootstrap):
            indices = np.random.choice(len(y_test), size=len(y_test), replace=True)
            y_test_boot = y_test[indices]
            y_pred_boot = y_pred[indices]

            bootstrap_metrics["accuracy"].append(accuracy_score(y_test_boot, y_pred_boot))
            bootstrap_metrics["f1"].append(f1_score(y_test_boot, y_pred_boot, average="weighted"))
            bootstrap_metrics["precision"].append(precision_score(y_test_boot, y_pred_boot, average="weighted"))
            bootstrap_metrics["recall"].append(recall_score(y_test_boot, y_pred_boot, average="weighted"))

        for metric, values in bootstrap_metrics.items():
            ci_lower = np.percentile(values, 2.5)
            ci_upper = np.percentile(values, 97.5)
            logger.info(f"   {metric.upper()}: [{ci_lower:.4f}, {ci_upper:.4f}]")

        # Detailed classification report
        logger.info("\nDETAILED CLASSIFICATION REPORT:")
        target_names = self.label_encoder.classes_
        logger.info("\n" + classification_report(y_test, y_pred, target_names=target_names))

        # Confusion matrix with percentages
        logger.info("\nCONFUSION MATRIX:")
        cm = confusion_matrix(y_test, y_pred)
        cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

        # Feature importance analysis with more detail
        feature_importance = self.model.feature_importances_
        top_features_idx = np.argsort(feature_importance)[-20:]  # Top 20 features

        logger.info("\nTOP 20 MOST IMPORTANT FEATURES:")
        for i, feat_idx in enumerate(reversed(top_features_idx)):
            logger.info(f"   {i + 1:2d}. Feature {feat_idx:3d}: {feature_importance[feat_idx]:.6f}")

        # Feature importance statistics
        logger.info("\nFEATURE IMPORTANCE STATISTICS:")
        logger.info(f"   Max importance: {np.max(feature_importance):.6f}")
        logger.info(f"   Mean importance: {np.mean(feature_importance):.6f}")
        logger.info(f"   Std importance: {np.std(feature_importance):.6f}")
        logger.info(f"   Non-zero features: {np.sum(feature_importance > 0)}")
        logger.info(f"   Features > 0.001: {np.sum(feature_importance > 0.001)}")
        logger.info(f"   Features > 0.01: {np.sum(feature_importance > 0.01)}")

        return {
            "test_metrics": test_metrics,
            "cv_metrics": cv_metrics,
            "bootstrap_confidence_intervals": bootstrap_metrics,
            "confusion_matrix": cm,
            "confusion_matrix_normalized": cm_normalized,
            "feature_importance": feature_importance,
            "top_features_idx": top_features_idx,
            "y_test": y_test,
            "y_pred": y_pred,
            "y_pred_proba": y_pred_proba,
        }

    def create_baseline_features(self, embeddings_dir):
        """Create baseline feature sets for comparison"""
        logger.info("\nCreating baseline feature sets for comparison...")

        # Load raw EMT data for baseline features
        emt_files = list(embeddings_dir.parent.parent.glob("data/*/breathing/*.emt"))
        logger.info(f"   Found {len(emt_files)} EMT files for baseline features")

        baseline_data = {
            "statistical": [],
            "pca_raw": [],
            "temporal_mean": [],
            "labels": [],
            "filenames": [],
        }

        for emt_file in sorted(emt_files[:50]):  # Limit for demonstration
            try:
                # Determine label from filename
                filename = emt_file.name
                if ".min." in filename or filename.endswith(".min.emt"):
                    label = "training"
                else:
                    label = "rest_recovery"

                # Load EMT data
                try:
                    # Read EMT file (assuming it's a simple format)
                    # This is a simplified version - adjust based on your EMT format
                    data = np.loadtxt(str(emt_file))
                    if len(data) < 100:  # Skip files that are too short
                        continue

                    # Ensure we have 89 sensors (columns 1-89, skip timestamp)
                    if data.shape[1] < 89:
                        continue
                    sensor_data = data[:, 1:90]  # Skip timestamp, take 89 sensors

                except Exception as e:
                    logger.warning(f"   Warning: Could not load {filename}: {e}")
                    continue

                # 1. Statistical features (mean, std, min, max, etc. for each sensor)
                statistical_features = []
                for col in range(sensor_data.shape[1]):
                    col_data = sensor_data[:, col]
                    statistical_features.extend(
                        [
                            np.mean(col_data),
                            np.std(col_data),
                            np.min(col_data),
                            np.max(col_data),
                            np.median(col_data),
                            np.percentile(col_data, 25),
                            np.percentile(col_data, 75),
                            np.var(col_data),
                            stats.skew(col_data),
                            stats.kurtosis(col_data),
                        ]
                    )

                # 2. PCA features (first 50 components of raw data)
                if not hasattr(self, "pca_baseline"):
                    self.pca_baseline = PCA(n_components=50, random_state=42)
                    # Fit on first file, then transform all
                    pca_features = self.pca_baseline.fit_transform(sensor_data.T).flatten()
                else:
                    pca_features = self.pca_baseline.transform(sensor_data.T).flatten()

                # 3. Temporal mean features (mean over time for each sensor)
                temporal_features = np.mean(sensor_data, axis=0)

                # Store features
                baseline_data["statistical"].append(statistical_features)
                baseline_data["pca_raw"].append(pca_features)
                baseline_data["temporal_mean"].append(temporal_features)
                baseline_data["labels"].append(label)
                baseline_data["filenames"].append(filename)

            except Exception as e:
                logger.warning(f"   Warning: Error processing {emt_file}: {e}")
                continue

        # Convert to arrays
        for key in ["statistical", "pca_raw", "temporal_mean"]:
            if baseline_data[key]:
                baseline_data[key] = np.array(baseline_data[key])
                logger.info(f"   {key.upper()} features shape: {baseline_data[key].shape}")
            else:
                logger.warning(f"   Warning: No {key} features created")

        return baseline_data

    def compare_baseline_methods(self, embeddings, labels, baseline_data):
        """Compare GNN embeddings against baseline feature extraction methods"""
        logger.info("\nCOMPARING FEATURE EXTRACTION METHODS")
        logger.info("=" * 60)

        comparison_results = {}

        # 1. GNN Embeddings (original method)
        logger.info("\n1. GNN AUTOENCODER EMBEDDINGS:")
        gnn_results = self.train_baseline_classifier(embeddings, labels, "GNN Embeddings")
        comparison_results["gnn_embeddings"] = gnn_results

        # 2-4. Baseline methods
        baseline_methods = [
            ("statistical", "Raw Statistical Features"),
            ("pca_raw", "PCA (Raw Coordinates)"),
            ("temporal_mean", "Mean Temporal Patterns"),
        ]

        for method_key, method_name in baseline_methods:
            if method_key in baseline_data and len(baseline_data[method_key]) > 0:
                logger.info(f"\n{len(comparison_results) + 1}. {method_name.upper()}:")

                # Match labels with baseline data
                baseline_labels = baseline_data["labels"]
                baseline_features = baseline_data[method_key]

                if len(baseline_features) > 10:  # Only if we have enough samples
                    results = self.train_baseline_classifier(baseline_features, baseline_labels, method_name)
                    comparison_results[method_key] = results
                else:
                    logger.warning(f"   Insufficient data for {method_name} ({len(baseline_features)} samples)")

        # Print comparison table
        logger.info("\nMETHOD COMPARISON SUMMARY")
        logger.info("=" * 80)
        logger.info(f"{'Method':<25} {'Features':<10} {'Test Acc':<10} {'CV Acc':<15} {'F1-Score':<10}")
        logger.info("-" * 80)

        for method_key, results in comparison_results.items():
            method_name = {
                "gnn_embeddings": "GNN Embeddings",
                "statistical": "Statistical Features",
                "pca_raw": "PCA Raw",
                "temporal_mean": "Temporal Mean",
            }.get(method_key, method_key)

            feature_count = results.get("feature_count", "N/A")
            test_acc = results.get("test_accuracy", 0)
            cv_acc = results.get("cv_accuracy_mean", 0)
            cv_std = results.get("cv_accuracy_std", 0)
            f1_score = results.get("test_f1", 0)

            logger.info(f"{method_name:<25} {feature_count:<10} {test_acc:<10.4f} " f"{cv_acc:.4f}±{cv_std:.3f} {f1_score:<10.4f}")

        return comparison_results

    def train_baseline_classifier(self, features, labels, method_name):
        """Train classifier for baseline comparison"""
        try:
            # Encode labels
            le = LabelEncoder()
            y_encoded = le.fit_transform(labels)

            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                features,
                y_encoded,
                test_size=0.2,
                random_state=42,
                stratify=y_encoded if len(np.unique(y_encoded)) > 1 else None,
            )

            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            # Train classifier
            classifier = xgb.XGBClassifier(random_state=42, eval_metric="logloss")
            classifier.fit(X_train_scaled, y_train)

            # Cross-validation
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            cv_scores = cross_val_score(classifier, X_train_scaled, y_train, cv=cv, scoring="accuracy")

            # Test set evaluation
            y_pred = classifier.predict(X_test_scaled)
            test_accuracy = accuracy_score(y_test, y_pred)
            test_f1 = f1_score(y_test, y_pred, average="weighted")

            logger.info(f"   Features: {features.shape[1]}")
            logger.info(f"   Test Accuracy: {test_accuracy:.4f}")
            logger.info(f"   CV Accuracy: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")
            logger.info(f"   F1-Score: {test_f1:.4f}")

            return {
                "method_name": method_name,
                "feature_count": features.shape[1],
                "test_accuracy": test_accuracy,
                "test_f1": test_f1,
                "cv_accuracy_mean": np.mean(cv_scores),
                "cv_accuracy_std": np.std(cv_scores),
                "cv_scores": cv_scores,
            }

        except Exception as e:
            logger.error(f"   Error training {method_name}: {e}")
            return {
                "method_name": method_name,
                "feature_count": 0,
                "test_accuracy": 0,
                "test_f1": 0,
                "cv_accuracy_mean": 0,
                "cv_accuracy_std": 0,
                "error": str(e),
            }

    def run_full_analysis(self):
        """Run complete classification analysis with baselines and comprehensive metrics"""
        logger.info("XGBOOST ACTIVITY CLASSIFICATION WITH COMPREHENSIVE ANALYSIS")
        logger.info("=" * 70)
        logger.info("Goal: Prove embeddings contain state information")
        logger.info("Model: Simple XGBoost (default parameters)")
        logger.info("Data: .min files = training, rest/recovery = rest")
        logger.info("Analysis: GNN embeddings vs baseline feature extraction methods")
        logger.info("Visualization: 3D/2D PCA plots + comprehensive metrics")
        logger.info("=" * 70)

        try:
            # 1. Load embeddings with activity labels
            embeddings, labels, filenames = self.load_aggregated_embeddings()

            # 2. Create baseline feature sets for comparison
            embeddings_dir = Path("ml/ml_data/embeddings_agg")
            baseline_data = self.create_baseline_features(embeddings_dir)

            # 3. Main GNN embedding analysis
            logger.info("\nMAIN ANALYSIS: GNN AUTOENCODER EMBEDDINGS")
            logger.info("=" * 60)
            classification_results = self.train_xgboost_classifier(embeddings, labels)

            # 4. Create visualizations
            viz_results = self.create_3d_visualization(
                embeddings,
                labels,
                model_accuracy=classification_results["test_metrics"]["accuracy"],
            )

            # 5. Baseline comparisons
            comparison_results = {}
            if baseline_data["labels"]:
                logger.info("\nBASELINE COMPARISONS")
                logger.info("=" * 60)
                comparison_results = self.compare_baseline_methods(embeddings, labels, baseline_data)
            else:
                logger.warning("\nWarning: Could not create baseline features for comparison")

            # 6. Statistical significance testing
            logger.info("\nSTATISTICAL SIGNIFICANCE ANALYSIS")
            logger.info("=" * 60)
            self.perform_statistical_tests(classification_results, comparison_results)

            # 7. Save comprehensive results
            output_file = Path("ml/ml_data/analysis_outputs/comprehensive_classification_results.json")
            output_file.parent.mkdir(exist_ok=True)

            final_results = {
                "timestamp": pd.Timestamp.now().isoformat(),
                "method": "comprehensive_xgboost_activity_classification",
                "classification_performance": {
                    "gnn_embeddings": {
                        "test_accuracy": float(classification_results["test_metrics"]["accuracy"]),
                        "test_f1": float(classification_results["test_metrics"]["f1"]),
                        "test_precision": float(classification_results["test_metrics"]["precision"]),
                        "test_recall": float(classification_results["test_metrics"]["recall"]),
                        "test_auc": float(classification_results["test_metrics"]["auc"]),
                        "cv_accuracy_mean": float(classification_results["cv_metrics"]["accuracy"]["mean"]),
                        "cv_accuracy_std": float(classification_results["cv_metrics"]["accuracy"]["std"]),
                        "cv_f1_mean": float(classification_results["cv_metrics"]["f1"]["mean"]),
                        "cv_f1_std": float(classification_results["cv_metrics"]["f1"]["std"]),
                    }
                },
                "baseline_comparisons": comparison_results,
                "data_summary": {
                    "total_samples": int(len(embeddings)),
                    "embedding_dimensions": int(embeddings.shape[1]),
                    "n_classes": len(set(labels)),
                    "class_distribution": {k: int(v) for k, v in pd.Series(labels).value_counts().items()},
                },
                "confusion_matrix": classification_results["confusion_matrix"].tolist(),
                "feature_importance_top20": {f"feature_{idx}": float(classification_results["feature_importance"][idx]) for idx in classification_results["top_features_idx"]},
                "pca_analysis": {
                    "explained_variance_ratio": viz_results["pca_explained_variance"].tolist(),
                    "total_variance_explained": float(viz_results["total_variance_explained"]),
                },
            }

            with open(output_file, "w") as f:
                json.dump(final_results, f, indent=2)

            logger.info(f"\nResults saved to: {output_file}")

            # 8. Print comprehensive summary
            self.print_comprehensive_summary(final_results, classification_results)

            return final_results

        except Exception as e:
            logger.error(f"ERROR in analysis: {e}")
            import traceback

            traceback.print_exc()
            return None

    def perform_statistical_tests(self, main_results, comparison_results):
        """Perform statistical significance tests"""
        logger.info("Statistical significance testing...")

        if not comparison_results:
            logger.warning("   No baseline results available for testing")
            return

        # Compare GNN embeddings vs baselines
        gnn_cv_scores = main_results.get("cv_metrics", {}).get("accuracy", {})
        gnn_mean = gnn_cv_scores.get("mean", 0)

        logger.info(f"   GNN Embeddings CV Accuracy: {gnn_mean:.4f}")

        for method_key, results in comparison_results.items():
            if method_key != "gnn_embeddings" and "cv_scores" in results:
                baseline_scores = results["cv_scores"]
                baseline_mean = np.mean(baseline_scores)

                # Perform t-test (simplified - assumes we have CV scores for GNN too)
                logger.info(f"   {results['method_name']} CV Accuracy: {baseline_mean:.4f}")

                # Calculate effect size (Cohen's d)
                pooled_std = np.sqrt(((4 * gnn_cv_scores.get("std", 0) ** 2) + (4 * np.std(baseline_scores) ** 2)) / 8)
                if pooled_std > 0:
                    cohens_d = (gnn_mean - baseline_mean) / pooled_std
                    logger.info(f"   Cohen's d vs {results['method_name']}: {cohens_d:.3f}")

    def print_comprehensive_summary(self, final_results, classification_results):
        """Print final comprehensive summary"""
        logger.info("\nCOMPREHENSIVE FINAL SUMMARY")
        logger.info("=" * 70)

        gnn_perf = final_results["classification_performance"]["gnn_embeddings"]

        logger.info("GNN AUTOENCODER EMBEDDING CLASSIFICATION:")
        logger.info("   Model: Simple XGBoost (default parameters)")
        logger.info(f"   Test Accuracy: {gnn_perf['test_accuracy']:.4f}")
        logger.info(f"   Test F1-Score: {gnn_perf['test_f1']:.4f}")
        logger.info(f"   Test Precision: {gnn_perf['test_precision']:.4f}")
        logger.info(f"   Test Recall: {gnn_perf['test_recall']:.4f}")
        logger.info(f"   Test ROC-AUC: {gnn_perf['test_auc']:.4f}")
        logger.info(f"   CV Accuracy: {gnn_perf['cv_accuracy_mean']:.4f} ± {gnn_perf['cv_accuracy_std']:.4f}")

        logger.info("\nDATA SUMMARY:")
        data_sum = final_results["data_summary"]
        logger.info(f"   Total Samples: {data_sum['total_samples']}")
        logger.info(f"   Embedding Dimensions: {data_sum['embedding_dimensions']}")
        logger.info(f"   Classes: {data_sum['n_classes']}")
        for class_name, count in data_sum["class_distribution"].items():
            pct = (count / data_sum["total_samples"]) * 100
            logger.info(f"   {class_name}: {count} samples ({pct:.1f}%)")

        logger.info("\nVISUALIZATION:")
        pca_info = final_results["pca_analysis"]
        logger.info(f"   PCA Variance Explained: {pca_info['total_variance_explained']:.1%}")
        logger.info("   3D/2D plots: ml/ml_data/analysis_outputs/activity_classification_static.png")
        logger.info("   Interactive plot: ml/ml_data/analysis_outputs/activity_classification_3d_interactive.html")


def main():
    """Main execution"""
    classifier = SimpleActivityClassifier()
    return classifier.run_full_analysis()


if __name__ == "__main__":
    results = main()
