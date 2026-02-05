"""
Model utilities for plant disease detection
Handles model creation, training, and evaluation
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from pathlib import Path


class ModelConfig:
    """Configuration for model building and training"""
    def __init__(
        self,
        img_size=(224, 224),
        num_classes=3,
        learning_rate=0.001,
        epochs=20,
        dropout_rate=0.5
    ):
        self.img_size = img_size
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.dropout_rate = dropout_rate


def build_model(config=None):
    """
    Build MobileNetV2-based transfer learning model
    
    Args:
        config: ModelConfig object (optional)
        
    Returns:
        Compiled Keras model
    """
    if config is None:
        config = ModelConfig()
    
    # Load pre-trained MobileNetV2 (without top layers)
    base_model = MobileNetV2(
        input_shape=(*config.img_size, 3),
        include_top=False,
        weights='imagenet'
    )
    
    # Freeze base model layers
    base_model.trainable = False
    
    # Build model
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(config.dropout_rate),
        layers.Dense(128, activation='relu'),
        layers.Dropout(config.dropout_rate),
        layers.Dense(config.num_classes, activation='softmax')
    ])
    
    # Compile model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=config.learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def get_callbacks(model_dir='../models'):
    """
    Get training callbacks for model checkpoint, early stopping, and learning rate reduction
    
    Args:
        model_dir: Directory to save model checkpoints
        
    Returns:
        List of Keras callbacks
    """
    model_path = Path(model_dir)
    model_path.mkdir(exist_ok=True, parents=True)
    
    callbacks = [
        # Save best model
        ModelCheckpoint(
            str(model_path / 'best_model.keras'),
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        # Early stopping
        EarlyStopping(
            monitor='val_loss',
            patience=7,
            restore_best_weights=True,
            verbose=1
        ),
        # Reduce learning rate on plateau
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.3,
            patience=3,
            min_lr=1e-7,
            verbose=1
        )
    ]
    
    return callbacks


def train_model(
    model,
    train_generator,
    val_generator,
    config=None,
    callbacks=None
):
    """
    Train the model
    
    Args:
        model: Compiled Keras model
        train_generator: Training data generator
        val_generator: Validation data generator
        config: ModelConfig object (optional)
        callbacks: List of Keras callbacks (optional)
        
    Returns:
        Training history
    """
    if config is None:
        config = ModelConfig()
    
    if callbacks is None:
        callbacks = get_callbacks()
    
    history = model.fit(
        train_generator,
        epochs=config.epochs,
        validation_data=val_generator,
        callbacks=callbacks
    )
    
    return history


def evaluate_model(model, test_generator):
    """
    Evaluate model on test data
    
    Args:
        model: Trained Keras model
        test_generator: Test data generator
        
    Returns:
        Dictionary with evaluation metrics
    """
    test_loss, test_accuracy = model.evaluate(test_generator)
    
    return {
        'test_loss': test_loss,
        'test_accuracy': test_accuracy
    }


def predict_image(model, image_array, class_names):
    """
    Make prediction on a single preprocessed image
    
    Args:
        model: Trained Keras model
        image_array: Preprocessed image array
        class_names: List of class names
        
    Returns:
        Dictionary with prediction results
    """
    predictions = model.predict(image_array)
    predicted_class_idx = predictions[0].argmax()
    confidence = predictions[0][predicted_class_idx]
    
    return {
        'class': class_names[predicted_class_idx],
        'confidence': float(confidence),
        'all_probabilities': {
            class_names[i]: float(predictions[0][i]) 
            for i in range(len(class_names))
        }
    }


def fine_tune_model(model, train_generator, val_generator, 
                     base_model_layers=100, learning_rate=0.0001, epochs=10):
    """
    Fine-tune the model by unfreezing some base model layers
    
    Args:
        model: Pre-trained model
        train_generator: Training data generator
        val_generator: Validation data generator
        base_model_layers: Number of layers to unfreeze from the end
        learning_rate: Learning rate for fine-tuning (should be lower)
        epochs: Number of epochs for fine-tuning
        
    Returns:
        Training history
    """
    # Unfreeze the base model
    base_model = model.layers[0]
    base_model.trainable = True
    
    # Freeze all layers except the last N
    for layer in base_model.layers[:-base_model_layers]:
        layer.trainable = False
    
    # Recompile with lower learning rate
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Continue training
    history = model.fit(
        train_generator,
        epochs=epochs,
        validation_data=val_generator,
        callbacks=get_callbacks()
    )
    
    return history


def load_trained_model(model_path='../models/best_model.keras'):
    """
    Load a saved trained model
    
    Args:
        model_path: Path to saved model file
        
    Returns:
        Loaded Keras model
    """
    return keras.models.load_model(model_path)


def get_model_summary(model):
    """
    Get a summary of the model architecture
    
    Args:
        model: Keras model
        
    Returns:
        Model summary as string
    """
    import io
    stream = io.StringIO()
    model.summary(print_fn=lambda x: stream.write(x + '\n'))
    return stream.getvalue()
