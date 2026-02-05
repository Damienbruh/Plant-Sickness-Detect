# Plant Disease Detection - Streamlit App

A web-based application for detecting plant diseases using deep learning.

## Features

- Upload plant leaf images through a user-friendly web interface
- Real-time disease detection with confidence scores
- Identifies three classes:
  - Healthy plants
  - Powdery Mildew disease
  - Rust disease
- Provides detailed diagnosis and treatment recommendations
- Shows prediction probabilities for all classes

## Installation

1. Make sure you have Python 3.8+ installed

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

## Running the App

To start the Streamlit application:

```bash
streamlit run app.py
```

The app will automatically open in your default web browser at `http://localhost:8501`

## How to Use

1. **Upload an Image**: Click the "Browse files" button and select a plant leaf image (JPG, JPEG, or PNG)

2. **Wait for Analysis**: The model will automatically analyze the image

3. **Review Results**: 
   - View the diagnosis (Healthy, Powdery Mildew, or Rust)
   - Check the confidence score
   - Read the detailed description
   - Follow the recommended actions

4. **Understand Predictions**: Review the detailed probability breakdown for all three classes

## Tips for Best Results

- Use well-lit, clear photos
- Focus on the leaf surface
- Avoid blurry or dark images
- Ensure the leaf fills most of the frame
- Take photos of symptomatic areas if disease is suspected

## Model Information

- **Architecture**: Convolutional Neural Network (CNN)
- **Input Size**: 224x224 pixels
- **Number of Classes**: 3
- **Training Dataset**: New Plant Diseases Dataset from Kaggle
- **Training Samples**: 1,322 images
- **Validation Samples**: 60 images
- **Test Samples**: 150 images

## Disease Information

### Healthy
- Status: Plant is healthy
- Action: Continue regular maintenance

### Powdery Mildew
- Symptoms: White powdery coating on leaves
- Cause: Fungal infection
- Treatment: Remove affected leaves, apply fungicide, improve air circulation

### Rust Disease
- Symptoms: Orange-brown pustules on leaves
- Cause: Fungal infection
- Treatment: Remove infected leaves, apply fungicide, improve drainage

## Technical Details

The application uses:
- **Streamlit**: Web application framework
- **TensorFlow/Keras**: Deep learning model
- **PIL (Pillow)**: Image processing
- **NumPy**: Array operations

Image preprocessing:
1. Resize to 224x224 pixels
2. Normalize pixel values (divide by 255)
3. Add batch dimension

## Troubleshooting

### Model Loading Error
If you see "Failed to load model", ensure:
- The `models/final_model.keras` file exists
- You're running the app from the project root directory

### Import Errors
If you get import errors, reinstall dependencies:
```bash
pip install -r requirements.txt --upgrade
```

### Performance Issues
- First run may be slower due to model loading
- Subsequent predictions are faster due to caching
- Large images may take longer to process

## Project Structure

```
Plant-Sickness-Detect/
├── app.py                    # Main Streamlit application
├── models/
│   └── final_model.keras    # Trained model
├── Dataset/                  # Training data
├── Notebooks/               # Jupyter notebooks for training
├── requirements.txt         # Python dependencies
└── STREAMLIT_README.md     # This file
```

## Future Enhancements

Possible improvements:
- Add more plant species
- Include more disease types
- Batch image processing
- Export diagnosis reports
- Multi-language support
- Mobile app version

## License

Academic project - educational purposes

## Author

Damien
