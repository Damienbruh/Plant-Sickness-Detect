"""
Data utilities for plant disease detection
Handles data loading, preprocessing, and augmentation
"""

from pathlib import Path
from tensorflow.keras.preprocessing.image import ImageDataGenerator


class DataConfig:
    """Configuration for data loading"""
    def __init__(
        self,
        img_size=(224, 224),
        batch_size=32,
        class_names=None
    ):
        self.img_size = img_size
        self.batch_size = batch_size
        self.class_names = class_names


def get_data_generators(
    train_dir,
    val_dir,
    test_dir,
    config=None
):
    """
    Create data generators for training, validation, and testing
    
    Args:
        train_dir: Path to training data directory
        val_dir: Path to validation data directory
        test_dir: Path to test data directory
        config: DataConfig object (optional)
        
    Returns:
        Tuple of (train_generator, val_generator, test_generator)
    """
    if config is None:
        config = DataConfig()
    
    # Data augmentation for training
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    
    # Only rescaling for validation and test
    val_test_datagen = ImageDataGenerator(rescale=1./255)
    
    # Create data generators
    train_generator = train_datagen.flow_from_directory(
        str(train_dir),
        target_size=config.img_size,
        batch_size=config.batch_size,
        class_mode='categorical',
        shuffle=True
    )
    
    val_generator = val_test_datagen.flow_from_directory(
        str(val_dir),
        target_size=config.img_size,
        batch_size=config.batch_size,
        class_mode='categorical',
        shuffle=False
    )
    
    test_generator = val_test_datagen.flow_from_directory(
        str(test_dir),
        target_size=config.img_size,
        batch_size=config.batch_size,
        class_mode='categorical',
        shuffle=False
    )
    
    return train_generator, val_generator, test_generator


def get_class_names(data_dir):
    """
    Get sorted list of class names from data directory
    
    Args:
        data_dir: Path to data directory containing class subdirectories
        
    Returns:
        List of class names
    """
    data_path = Path(data_dir)
    return sorted([d.name for d in data_path.iterdir() if d.is_dir()])


def preprocess_image(image_input, img_size=(224, 224)):
    """
    Load and preprocess a single image for prediction
    
    Args:
        image_input: Path to image file (str/Path) or PIL Image object
        img_size: Target size for resizing
        
    Returns:
        Preprocessed image array ready for model prediction
    """
    from tensorflow.keras.preprocessing import image
    from PIL import Image
    import numpy as np
    
    # Handle both file paths and PIL Image objects
    if isinstance(image_input, (str, Path)):
        img = image.load_img(image_input, target_size=img_size)
    elif isinstance(image_input, Image.Image):
        img = image_input.resize(img_size)
    else:
        raise TypeError("image_input must be a file path or PIL Image object")
    
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    
    return img_array


def get_data_statistics(generator):
    """
    Get statistics about a data generator
    
    Args:
        generator: Keras ImageDataGenerator
        
    Returns:
        Dictionary with statistics
    """
    return {
        'num_samples': generator.samples,
        'num_classes': generator.num_classes,
        'class_indices': generator.class_indices,
        'batch_size': generator.batch_size,
        'image_shape': generator.image_shape
    }
