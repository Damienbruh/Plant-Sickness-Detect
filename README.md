# Plant Disease Detection - AI Implementation

## Project Overview
This project implements an AI-based image classification system to detect plant diseases using deep learning. The model classifies plant leaf images into three categories:
- **Healthy**: No disease detected
- **Powdery**: Powdery mildew disease
- **Rust**: Rust disease

## Features
- Transfer learning using MobileNetV2 pre-trained on ImageNet
- Achieves 98.33% validation accuracy
- Interactive Streamlit web application
- REST API for programmatic access (FastAPI)
- Modular Python codebase for easy deployment and maintenance

## Academic Context
This is a complete AI implementation project (kunskapskontroll) demonstrating:
- Deep learning for image classification
- Dataset handling and preprocessing
- Model training with transfer learning
- Model evaluation and deployment
- AI implementation best practices

**Dataset Source**: Kaggle (https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset)

## Project Structure
```
Plant-Sickness-Detect/
├── Dataset/
│   ├── Train/Train/          # Training images (1322 samples)
│   ├── Validation/Validation/ # Validation images (60 samples)
│   └── Test/Test/             # Test images (150 samples)
├── Notebooks/
│   ├── data_exploration.ipynb # Dataset analysis and visualization
│   ├── model_training.ipynb   # Model building and training
│   └── evaluation.ipynb       # Model evaluation and metrics
├── models/
│   └── best_model.keras       # Trained model (98.33% val accuracy)
├── src/
│   ├── data_utils.py          # Data preprocessing and augmentation
│   ├── model.py               # Model architecture and training utilities
│   └── app.py                 # FastAPI REST API
├── app.py                     # Streamlit web application
├── requirements.txt           # Python dependencies
├── API_README.md              # API documentation
└── README.md                  # This file
```

## Technology Stack
- **Framework**: TensorFlow 2.19/Keras
- **Language**: Python 3.8+
- **Web Frameworks**: Streamlit, FastAPI
- **Model Architecture**: MobileNetV2 (Transfer Learning)
- **Key Libraries**:
  - TensorFlow/Keras: Deep learning framework
  - NumPy: Numerical computing
  - Pandas: Data manipulation
  - Matplotlib/Seaborn: Visualization
  - Pillow: Image processing
  - scikit-learn: Metrics and evaluation
  - FastAPI: REST API framework
  - Uvicorn: ASGI server
  - Streamlit: Web UI framework

## Model Details
- **Architecture**: MobileNetV2 + Custom Layers
  - Base: MobileNetV2 pre-trained on ImageNet (frozen)
  - Custom layers: GlobalAveragePooling2D, Dropout(0.5), Dense(128), Dropout(0.5), Dense(3)
- **Training**: 15 epochs (~34 minutes on CPU)
- **Performance**:
  - Validation Accuracy: 98.33%
  - Final Training Accuracy: 97.13%
- **Input Size**: 224x224 RGB images
- **Output**: 3 classes with confidence scores
- **Data Augmentation**: Rotation, shift, shear, zoom, horizontal flip
- **Callbacks**: ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

## Methodology

### 1. Data Exploration (data_exploration.ipynb)
- Analyzed dataset distribution: 1322 training, 60 validation, 150 test images
- Visualized sample images from each disease category
- Examined image properties (dimensions, formats, color modes)
- Identified class distributions and preprocessing requirements

### 2. Model Training (model_training.ipynb)
**Approach**: Transfer learning with MobileNetV2

Implemented pipeline:
- Image preprocessing and normalization (rescaling to 0-1)
- Data augmentation: rotation (20°), shifts (20%), shear (20%), zoom (20%), horizontal flip
- Transfer learning: MobileNetV2 pre-trained on ImageNet (base frozen)
- Custom classification head: GlobalAveragePooling + Dense layers with Dropout
- Training: 15 epochs with Adam optimizer (lr=0.001)
- Callbacks: ModelCheckpoint (val_accuracy), EarlyStopping (patience=5), ReduceLROnPlateau
- Results: 98.33% validation accuracy, 97.13% training accuracy

### 3. Model Evaluation (evaluation.ipynb)
- Test set performance metrics
- Confusion matrix analysis
- Precision, recall, F1-score per class
- Visualization of predictions
- Error analysis

### 4. Deployment
**Modular codebase** (`src/`):
- `data_utils.py`: Data generators, preprocessing, augmentation
- `model.py`: Model building, training, evaluation utilities
- `app.py`: FastAPI REST API for programmatic access

**Web applications**:
- Streamlit app (`app.py`): Interactive UI for image upload and diagnosis
- FastAPI (`src/app.py`): REST API with endpoints for single/batch predictions

## Getting Started

### Installation
1. Clone or navigate to the project directory
2. Install dependencies:
```bash
pip install -r requirements.txt
```

### Quick Start - Using the Applications

**Option 1: Streamlit Web App (Recommended for end users)**
```bash
streamlit run app.py
```
Open your browser to `http://localhost:8501` and upload plant images for instant diagnosis.

**Option 2: FastAPI REST API (For developers/integration)**
```bash
cd src
python app.py
```
API available at `http://localhost:8000` with interactive docs at `/docs`

See [API_README.md](API_README.md) for detailed API documentation.

### Development Workflow

**Working with Notebooks:**
1. Launch Jupyter:
```bash
jupyter notebook
```
2. Open notebooks in order:
   - `data_exploration.ipynb` - Understand the dataset
   - `model_training.ipynb` - Train the model (~34 min on CPU)
   - `evaluation.ipynb` - Evaluate model performance

**Using Python Modules:**
```python
from src.data_utils import get_data_generators, DataConfig
from src.model import build_model, train_model, ModelConfig

# Load data
train_gen, val_gen, test_gen = get_data_generators(
    'Dataset/Train/Train',
    'Dataset/Validation/Validation',
    'Dataset/Test/Test'
)

# Build and train model
model = build_model()
history = train_model(model, train_gen, val_gen)
```

## Dataset Details
The dataset is organized into three splits:
- **Training**: 1322 images (used to train the model)
- **Validation**: 60 images (monitoring during training, early stopping)
- **Test**: 150 images (final evaluation on unseen data)

Each split contains three subdirectories:
- `Healthy/`: Healthy plant leaves
- `Powdery/`: Leaves with powdery mildew disease
- `Rust/`: Leaves with rust disease

## Usage Examples

### Making Predictions

**Using Streamlit App:**
1. Run `streamlit run app.py`
2. Upload a plant leaf image
3. View diagnosis and treatment recommendations

**Using FastAPI:**
```python
import requests

url = "http://localhost:8000/predict"
files = {"file": open("plant_image.jpg", "rb")}
response = requests.post(url, files=files)
print(response.json())
```

**Using Python Modules:**
```python
from src.model import load_trained_model, predict_image
from src.data_utils import preprocess_image, get_class_names

model = load_trained_model('models/best_model.keras')
class_names = get_class_names('Dataset/Train/Train')

img_array = preprocess_image('test_image.jpg')
result = predict_image(model, img_array, class_names)
print(f"{result['class']}: {result['confidence']:.2%}")
```

## Results
- **Validation Accuracy**: 98.33%
- **Training Accuracy**: 97.13%
- **Training Time**: ~34 minutes on CPU (15 epochs)
- **Model Size**: 164,355 trainable parameters

## Future Enhancements
- Cloud deployment (Azure, AWS, GCP)
- Mobile application integration
- Expand to more disease types and plant species
- Real-time inference from camera feed
- Multi-language support for agricultural use
- Integration with plant care recommendation system
- Fine-tuning for specific crop varieties

## Author
Damien

## License
Academic project - educational purposes
