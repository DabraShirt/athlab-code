# Graph Neural Network for Breathing Pattern Analysis

TBD

## Paper Release

TBD

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
Trains the GNN autoencoder model using sequence-based data. The model learns to encode breathing patterns into a low-dimensional representation.

### 3. Generate Embeddings
```bash
python ml/utils/generate_embeddings.py
```
Applies the trained autoencoder to generate embeddings for all data samples.

### 4. Process and Aggregate Embeddings
```bash
python ml/utils/process_all_embeddings.py
```
Aggregates embeddings across sequences and prepares them for classification.

### 5. Train Classifier
```bash
python ml/train/train_simple_classifier.py
```
Trains an XGBoost classifier on the generated embeddings to distinguish between physiological states (rest/recovery vs training).

## Data Format

Due to the clinical nature of the data, the original dataset is not included in this repository. However, we provide an example data file in `data_example/` to demonstrate the expected format.

### Input Data Structure

The pipeline expects 3D point track data in BTS ASCII format (`.emt` files). See `data_example/example_3D_Point_Tracks.emt` for the format specification:

- **Type**: Point 3D tracks
- **Measurement unit**: millimeters (mm)
- **Frequency**: 60 Hz
- **Tracks**: 89 3D markers (X, Y, Z coordinates)
- **Format**: Tab-separated values with Frame, Time, and marker coordinates

The data should be organized in the following directory structure:
```
data/
├── ID_1/
│   ├── Loose/
│   │   ├── 6.min/
│   │   ├── 12.min/
│   │   ├── rest/
│   │   └── recovery/
│   ├── Normal/
│   └── Tight/
├── ID_2/
...
```

## Requirements

See `requirements.txt` for Python dependencies.

## Citation

If you use this code in your research, please cite our paper:

```
[Citation details to be added upon publication]
```

## License

[License details to be added]
