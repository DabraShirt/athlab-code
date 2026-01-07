# Graph Neural Network Autoencoder for Breathing Pattern Analysis

This repository contains the implementation of a Graph Neural Network (GNN) autoencoder for analyzing respiratory patterns from 3D motion capture data. Our approach demonstrates that unsupervised learning can extract meaningful physiological state information from complex spatiotemporal breathing patterns, achieving 98.08% accuracy in distinguishing between rest/recovery and training states using a single discriminative feature.

## Key Contributions

- **First GNN Application**: Novel application of graph neural networks to 3D breathing pattern analysis
- **Subject-Aware Data Splitting**: Proper methodology preventing data leakage across subjects
- **Single-Feature Discovery**: Identification of one critical temporal feature (Feature 54) with perfect discriminative power
- **Benchmark Establishment**: First systematic benchmark for GNN-based respiratory monitoring (98.08% test accuracy, 98.54% CV accuracy)

## Methodology Overview

Our approach processes 3D point cloud data from high-frequency cameras (60 Hz) capturing 89 body markers during breathing. The GNN autoencoder:

1. **Constructs spatiotemporal graphs** from 5-timestep sequences with 445 nodes (89 markers × 5 timesteps)
2. **Learns unsupervised representations** through reconstruction objectives
3. **Extracts temporal breathing patterns** that distinguish physiological states
4. **Enables subject-level generalization** through proper data splitting methodology

The key finding: temporal evolution of breathing patterns contains more discriminative information than instantaneous measurements.

## Paper & Results

**Status**: Research paper in preparation

**Key Results**:
- **Classification Performance**: 98.08% test accuracy, 98.54% ± 1.19% cross-validation accuracy
- **Feature Discovery**: Single feature (Feature 54) achieved perfect discriminative power (importance = 1.0)
- **Dataset Scale**: 301,705 sequence graphs from 280 EMT files across 17 subjects
- **Architecture**: 3-layer GNN encoder [68, 62, 26] with 756,075 parameters
- **Methodology**: Subject-aware data splitting preventing subject-level data leakage

This work establishes the first benchmark for GNN-based breathing pattern analysis, demonstrating that temporal breathing evolution is the key signature of physiological states.

## Pipeline Overview

The following commands execute the full analysis pipeline in order:

### 1. Start MLflow Tracking Server
```bash
mlflow server --backend-store-uri sqlite:///mlruns.db --default-artifact-root ./mlruns --host 0.0.0.0 --port 5001
```
Launches the MLflow server for experiment tracking and model management.

### 2. Train GNN Autoencoder
```bash
python -m ml.train.train_gnn_autoencoder --dataset-mode sequence --sequence-length 5 --sequence-step 3 --n-trials 2
```
Trains the GNN autoencoder using subject-aware data splitting and Optuna hyperparameter optimization. The model processes 301,705 sequence graphs with 445 nodes each, learning to encode breathing patterns into a 26-dimensional latent space. Training uses the discrimination score metric for unsupervised model selection.

### 3. Generate Embeddings
```bash
python ml/utils/generate_embeddings.py
```
Applies the trained autoencoder to generate 512-dimensional embeddings for all data samples. The encoder processes each sequence graph through the optimized architecture to extract meaningful physiological representations.

### 4. Process and Aggregate Embeddings
```bash
python ml/utils/process_all_embeddings.py
```
Aggregates embeddings across sequences using multiple strategies (mean, max, std, etc.) and prepares them for downstream classification tasks.

### 5. Train Classifier
```bash
python ml/train/train_simple_classifier.py
```
Trains an XGBoost classifier on the generated embeddings to distinguish between physiological states. Achieves 98.08% test accuracy using only Feature 54, demonstrating the single-feature discriminative power discovered through our GNN autoencoder approach.

## Data Format & Requirements

Due to the clinical nature of the breathing pattern data, the original dataset is not included in this repository. However, we provide example data structure and format specifications to demonstrate the expected input format.

### Input Data Specifications

The pipeline processes 3D motion capture data from high-frequency cameras capturing respiratory movements:

- **Data Type**: 3D point tracks from motion capture systems
- **Format**: BTS ASCII format (`.emt` files) 
- **Frequency**: 60 Hz sampling rate
- **Markers**: 89 3D body markers (X, Y, Z coordinates in millimeters)
- **Subjects**: 17 subjects across multiple breathing conditions
- **Total Scale**: 280 EMT files containing 905,118 frames

See `data_example/example_3D_Point_Tracks.emt` for the format specification:
- Tab-separated values with Frame, Time, and marker coordinates
- Consistent marker labeling across subjects and conditions
- High-precision coordinate data for accurate breathing pattern capture

### Directory Structure

The data should be organized to support subject-aware data splitting:
```
data/
├── Subject_ID_1/
│   ├── Normal/          # Normal breathing condition
│   │   ├── 6.min/       # 6-minute exercise
│   │   ├── 12.min/      # 12-minute exercise  
│   │   ├── rest/        # Rest periods
│   │   └── recovery/    # Recovery periods
│   ├── Tight/           # Tight bra condition
│   └── Loose/           # Loose bra condition
├── Subject_ID_2/
│   └── ...
└── 1201/                # Subject with different naming convention
    ├── 1201_normal/
    ├── 1201_tightbra/
    └── ...
```

## Technical Requirements

### Dependencies
See `requirements.txt` for complete Python dependencies. Key packages include:
- PyTorch & PyTorch Geometric for GNN implementation
- Optuna for hyperparameter optimization  
- MLflow for experiment tracking
- XGBoost for classification
- Standard scientific computing stack (numpy, pandas, scikit-learn)

### Hardware Specifications
- **Model Training**: Trained on CPU with 756,075 parameters
- **Memory**: Sufficient RAM for processing 301,705 graphs simultaneously
- **Storage**: Space for MLflow experiments and embedding generation

## Research Impact & Applications

This work establishes the foundation for a new research domain combining graph neural networks with respiratory analysis. Key impacts include:

- **Clinical Monitoring**: Potential for simplified respiratory monitoring systems
- **Sports Science**: Real-time physiological state detection for athletes  
- **Research Benchmark**: First systematic evaluation providing baseline performance metrics
- **Methodological Innovation**: Subject-aware splitting methodology for valid biomedical ML research

## Limitations & Future Work

As an exploratory study in this novel domain:
- **Scope**: Limited to healthy runners under controlled conditions
- **Architecture**: Basic GNN structure - more sophisticated spatial architectures likely needed
- **Validation**: Requires broader population studies for clinical deployment
- **States**: Currently binary classification (more complex state detection possible)

This work serves as a proof-of-concept and benchmark, demonstrating that GNN autoencoders can extract meaningful physiological patterns from complex breathing data.

## Citation

If you use this code in your research, please cite our paper:

```
@article{[AuthorYear]breathing_gnn,
  title={Graph Neural Network Autoencoders for Breathing Pattern Analysis: A Novel Approach to Physiological State Detection},
  author={[Authors]},
  journal={[Journal]},
  year={[Year]},
  note={[Publication details to be added upon acceptance]}
}
```

## License

MIT

---

**Note**: This repository contains the complete implementation used in our research paper. The code represents the first systematic application of graph neural networks to 3D breathing pattern analysis, establishing benchmarks and methodological foundations for this emerging research domain.
