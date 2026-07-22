# Chest Disease Classification using Deep Learning

A reproducible deep learning project for binary chest image classification using TensorFlow/Keras, VGG16 transfer learning, DVC pipeline orchestration, and a FastAPI-powered web interface for image prediction.

The repository is organized as a small production-style ML package: configuration-driven components, staged training pipelines, tracked experiment artifacts, and an API/UI layer for serving predictions.

> Important: This project is intended for machine learning experimentation and education. It is not a medical device and should not be used as a substitute for professional clinical diagnosis.

## Table of Contents

- [Project Highlights](#project-highlights)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Setup](#setup)
- [Run the Training Pipeline](#run-the-training-pipeline)
- [Run the Prediction App](#run-the-prediction-app)
- [API Endpoints](#api-endpoints)
- [Configuration](#configuration)
- [Model and Artifacts](#model-and-artifacts)
- [Experiment Tracking](#experiment-tracking)
- [Results](#results)
- [Troubleshooting](#troubleshooting)

## Project Highlights

- Binary image classification workflow built with TensorFlow/Keras.
- VGG16 transfer learning with ImageNet weights.
- Configurable model, training, and data settings through YAML files.
- DVC pipeline for reproducible data ingestion, base model preparation, training, and evaluation.
- FastAPI backend with a browser-based image upload interface.
- MLflow/DagsHub integration support for experiment tracking.
- Modular Python package layout under `src/cnnClassifier`.

## Architecture

```text
Raw image data
     |
     v
Data ingestion
     |
     v
Prepare VGG16 base model
     |
     v
Train classifier head
     |
     v
Evaluate model and write metrics
     |
     v
Serve model through FastAPI UI/API
```

The project is split into four ML pipeline stages:

1. `data_ingestion`: downloads the dataset from Google Drive and extracts it.
2. `prepare_base_model`: loads VGG16, freezes the base layers, and attaches a classification head.
3. `training`: trains the updated model using Keras image generators.
4. `evaluation`: evaluates the trained model and writes metrics to `scores.json`.

## Project Structure

```text
.
|-- app.py                         # FastAPI application and prediction routes
|-- main.py                        # Runs all ML pipeline stages sequentially
|-- dvc.yaml                       # DVC stage definitions
|-- dvc.lock                       # DVC pipeline lock file
|-- params.yaml                    # Training/model hyperparameters
|-- config/
|   `-- config.yaml                # Data, artifact, and model paths
|-- src/
|   `-- cnnClassifier/
|       |-- components/            # Data ingestion, model prep, training, evaluation logic
|       |-- config/                # Configuration manager
|       |-- constants/             # Constant paths
|       |-- entity/                # Dataclass config entities
|       |-- pipeline/              # Executable pipeline stages and prediction pipeline
|       `-- utils/                 # Common utility functions
|-- templates/
|   `-- index.html                 # Web UI for image upload, prediction, and training trigger
|-- artifacts/                     # Generated data/model artifacts
|-- model/
|   `-- model.h5                   # Runtime model used by the prediction API
|-- scores.json                    # Latest evaluation metrics
|-- requirements.txt               # Python dependencies
`-- setup.py                       # Package metadata
```

## Setup

### 1. Clone the repository

```bash
git clone <repository-url>
cd "Cheast Disease Classification"
```

### 2. Create and activate a virtual environment

Windows PowerShell:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

macOS/Linux:

```bash
python -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

The main dependencies include:

- `tensorflow`
- `keras`
- `fastapi`
- `uvicorn`
- `dvc`
- `gdown`
- `mlflow`
- `dagshub`
- `python-box`
- `ensure`

## Run the Training Pipeline

You can run the complete training workflow either through Python or DVC.

### Option 1: Run all stages with Python

```bash
python main.py
```

### Option 2: Run the reproducible DVC pipeline

```bash
dvc repro
```

DVC executes the stages defined in `dvc.yaml`:

```text
data_ingestion -> prepare_base_model -> training -> evaluation
```

Generated artifacts are written under `artifacts/`, and evaluation metrics are written to `scores.json`.

## Run the Prediction App

The FastAPI application serves both the API and the browser interface.

```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

Then open:

```text
http://localhost:8000
```

The web UI allows you to:

- Upload a JPG or PNG image.
- Run model prediction.
- Trigger the DVC training pipeline from the browser.

For inference, the application loads the model from:

```text
model/model.h5
```

The DVC training stage writes the trained model to:

```text
artifacts/training/model.h5
```

Make sure the model used for serving is available at `model/model.h5`.

## API Endpoints

### `GET /`

Serves the web interface from `templates/index.html`.

### `POST /predict`

Runs prediction on a base64-encoded image.

Request body:

```json
{
  "image": "<base64-encoded-image>"
}
```

Example response:

```json
[
  {
    "image": "Healthy"
  }
]
```

Current prediction labels in `src/cnnClassifier/pipeline/predict.py` are:

- `Healthy`
- `Coccidiosis`

Review and align these labels with your dataset classes before using the model in a domain-specific workflow.

### `GET /train` or `POST /train`

Runs:

```bash
dvc repro
```

Successful response:

```text
Training done successfully!!
```

## Configuration

### `config/config.yaml`

Controls artifact paths, dataset location, and model output paths.

Key settings:

```yaml
artifacts_root: artifacts

data_ingestion:
  source_URL: <google-drive-dataset-url>
  local_data_file: artifacts/data_ingestion/data.zip
  unzip_dir: artifacts/data_ingestion

training:
  trained_model_path: artifacts/training/model.h5
```

### `params.yaml`

Controls model and training hyperparameters.

Current values:

```yaml
AUGMENTATION: True
IMAGE_SIZE: [224, 224, 3]
BATCH_SIZE: 16
INCLUDE_TOP: False
EPOCHS: 10
CLASSES: 2
WEIGHTS: imagenet
LEARNING_RATE: 0.01
```

## Model and Artifacts

The project uses VGG16 as a frozen feature extractor and adds a softmax classification head for two classes.

Important paths:

- `artifacts/data_ingestion/`: downloaded and extracted dataset.
- `artifacts/prepare_base_model/base_model.h5`: original VGG16 base model.
- `artifacts/prepare_base_model/updated_base_model.h5`: VGG16 with the project classification head.
- `artifacts/training/model.h5`: trained model generated by the training stage.
- `model/model.h5`: model loaded by the FastAPI prediction service.
- `scores.json`: latest evaluation metrics.

## Experiment Tracking

The evaluation component includes MLflow/DagsHub integration:

```python
dagshub.init(
    repo_owner="sahil6098",
    repo_name="Chest-Disease-Classification-using-Deep-Learning",
    mlflow=True
)
```

The call to `evaluation.log_into_mlflow()` is currently commented out in `stage_04_evaluation.py`. Enable it if you want to log parameters, metrics, and model artifacts to the configured MLflow tracking server.

## Results

The latest checked-in `scores.json` contains:

```json
{
  "loss": 31.488407135009766,
  "accuracy": 1.0
}
```

Treat these metrics as a snapshot from the current local run. Always validate performance on a clean holdout set before drawing conclusions about real-world usefulness.

## Troubleshooting

### `ModuleNotFoundError: No module named 'cnnClassifier'`

Install the project package or ensure `src` is available on `PYTHONPATH`:

```bash
pip install -e .
```

### Dataset download fails

Check that the Google Drive URL in `config/config.yaml` is accessible and that the file ID can be downloaded by `gdown`.

### Prediction fails because the model file is missing

Confirm that a trained Keras model exists at:

```text
model/model.h5
```

### Training from the web UI fails

The `/train` endpoint runs `dvc repro`, so DVC must be installed and available in the same environment that runs FastAPI.

### TensorFlow installation issues

TensorFlow compatibility depends on Python version, operating system, and hardware. If installation fails, use a Python version supported by the TensorFlow version pinned in `requirements.txt`.

## License

No license file is currently included in this repository. Add a license before publishing or distributing the project.
