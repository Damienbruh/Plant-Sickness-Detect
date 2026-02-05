# Plant Disease Detection - AI Implementation

## Project Overview
This project implements an AI-based image classification system to detect plant diseases. The model classifies plant leaf images into three categories:
- **Healthy**: No disease detected
- **Powdery**: Powdery mildew disease
- **Rust**: Rust disease

## Academic Context
This is a complete AI implementation project (kunskapskontroll) with flexibility in approach. The project demonstrates:
- Deep learning for image classification
- Dataset handling and preprocessing
- Model training and evaluation
- AI implementation best practices

**Dataset Source**: Kaggle (https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset)

## Project Structure
```
Plant-Sickness-Detect/
├── Dataset/
│   ├── Train/Train/          # Training images (Healthy, Powdery, Rust)
│   ├── Validation/Validation/ # Validation images
│   └── Test/Test/             # Test images
├── Notebooks/
│   ├── data_exploration.ipynb # Dataset analysis and visualization
│   ├── model_training.ipynb   # Model building and training
│   └── evaluation.ipynb       # Model evaluation and metrics
├── src/
│   ├── data_utils.py          # Data loading utilities (future)
│   └── model.py               # Model architecture (future)
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

## Technology Stack
- **Framework**: TensorFlow 2.16+/Keras
- **Language**: Python 3.8+
- **Development Environment**: Jupyter Notebooks
- **Key Libraries**:
  - TensorFlow/Keras: Deep learning framework
  - NumPy: Numerical computing
  - Pandas: Data manipulation
  - Matplotlib/Seaborn: Visualization
  - Pillow: Image processing
  - scikit-learn: Metrics and evaluation

## Classification Task
- **Type**: Multi-class image classification
- **Classes**: 3 (Healthy, Powdery, Rust)
- **Input**: RGB plant leaf images (various sizes)
- **Output**: Class prediction with confidence scores

## Methodology

### 1. Data Exploration (data_exploration.ipynb)
- Analyze dataset distribution across train/validation/test splits
- Visualize sample images from each disease category
- Examine image properties (dimensions, formats, color modes)
- Identify class imbalances and potential data augmentation needs

### 2. Model Training (model_training.ipynb)
Potential approaches:
- **Custom CNN**: Build convolutional neural network from scratch
- **Transfer Learning**: Use pre-trained models (ResNet, EfficientNet, MobileNet)
- **Data Augmentation**: Apply transformations to increase training data diversity

Training pipeline:
- Image preprocessing and normalization
- Data augmentation (rotation, flip, zoom, etc.)
- Model architecture definition
- Training with validation monitoring
- Hyperparameter tuning
- Model checkpointing (save best model)

### 3. Model Evaluation (evaluation.ipynb)
- Test set performance metrics
- Confusion matrix analysis
- Precision, recall, F1-score per class
- Visualization of predictions
- Error analysis

## Getting Started

### Installation
1. Clone or navigate to the project directory
2. Install dependencies:
```bash
pip install -r requirements.txt
```

### Running the Notebooks
1. Launch Jupyter:
```bash
jupyter notebook
```
2. Open notebooks in order:
   - Start with `data_exploration.ipynb`
   - Then `model_training.ipynb`
   - Finally `evaluation.ipynb`

### Workflow
1. **Explore**: Run data_exploration.ipynb to understand the dataset
2. **Train**: Build and train model in model_training.ipynb
3. **Evaluate**: Assess performance in evaluation.ipynb

## Dataset Details
The dataset is organized into three splits:
- **Training**: Used to train the model
- **Validation**: Used for hyperparameter tuning and monitoring during training
- **Test**: Final evaluation on unseen data

Each split contains three subdirectories (Healthy, Powdery, Rust) with corresponding images.

## Model Requirements
For a passing grade, the implementation should include:
- Complete AI implementation with clear methodology
- Proper dataset handling (train/validation/test splits)
- Model training with appropriate architecture
- Evaluation metrics and analysis
- Documentation of approach and results
- Reproducible code in Jupyter notebooks

## Potential Extensions
- YOLO for object detection (if expanding scope)
- API deployment using cloud services (Azure, AWS, GCP)
- Web interface for plant disease prediction
- Real-time inference from camera feed
- Multi-plant species support

## Notes
- This is an image classification task, not object detection
- Focus on achieving good classification accuracy across all three classes
- Document any class imbalance strategies
- Explain model architecture choices
- Include visualization of results

## Future Work
- Deployment as REST API
- Mobile application integration
- Expand to more disease types
- Multi-language support for agricultural use
- Integration with plant care recommendations

## Author
Damien

## License
Academic project - educational purposes
