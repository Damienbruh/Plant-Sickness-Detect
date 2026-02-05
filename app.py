import streamlit as st
import tensorflow as tf
from tensorflow import keras
from PIL import Image
import numpy as np

st.set_page_config(
    page_title="Plant Disease Detection",
    page_icon="🌱",
    layout="wide"
)

IMG_SIZE = (224, 224)
CLASS_NAMES = ['Healthy', 'Powdery', 'Rust']
MODEL_PATH = 'models/final_model.keras'

# Load model with caching to improve performance
@st.cache_resource
def load_model():
    try:
        model = keras.models.load_model(MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

def preprocess_image(image):

    image = image.resize(IMG_SIZE)
    
    img_array = np.array(image)
    
    img_array = img_array / 255.0
    
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array

def predict_disease(model, image):

    predictions = model.predict(image, verbose=0)
    
    predicted_class_idx = np.argmax(predictions[0])
    predicted_class = CLASS_NAMES[predicted_class_idx]
    confidence = predictions[0][predicted_class_idx] * 100
    
    return predicted_class, confidence, predictions[0]

def get_diagnosis(predicted_class):

    diagnoses = {
        'Healthy': {
            'status': '✅ Healthy Plant',
            'description': 'Your plant appears to be healthy! No disease detected.',
            'recommendations': [
                'Continue regular watering and maintenance',
                'Monitor for any changes in appearance',
                'Ensure adequate sunlight and nutrients',
                'Keep the growing area clean'
            ],
            'color': 'green'
        },
        'Powdery': {
            'status': '⚠️ Powdery Mildew Detected',
            'description': 'Your plant has powdery mildew, a fungal disease that appears as white powdery spots on leaves.',
            'recommendations': [
                'Remove affected leaves immediately',
                'Improve air circulation around plants',
                'Apply fungicide (neem oil or sulfur-based)',
                'Avoid overhead watering',
                'Reduce humidity if growing indoors',
                'Space plants properly to prevent spread'
            ],
            'color': 'orange'
        },
        'Rust': {
            'status': '🔴 Rust Disease Detected',
            'description': 'Your plant has rust disease, a fungal infection that causes orange-brown pustules on leaves.',
            'recommendations': [
                'Remove and destroy infected leaves',
                'Apply appropriate fungicide treatment',
                'Improve air circulation',
                'Water at the base of plants, not leaves',
                'Clean up fallen leaves and debris',
                'Consider resistant plant varieties for future planting'
            ],
            'color': 'red'
        }
    }
    
    return diagnoses.get(predicted_class, diagnoses['Healthy'])

def main():

    # Title and description
    st.title("🌱 Plant Disease Detection System")
    st.markdown("""
    Upload an image of a plant leaf to detect if it's healthy or has a disease.
    This model can identify:
    - **Healthy** plants
    - **Powdery Mildew** disease
    - **Rust** disease
    """)
    
    model = load_model()
    
    if model is None:
        st.error("Failed to load model. Please check if the model file exists.")
        return
    
    with st.sidebar:
        st.header("About")
        st.info("""
        This AI-powered tool uses deep learning to detect plant diseases from leaf images.
        
        **How to use:**
        1. Upload a clear image of a plant leaf
        2. Wait for the analysis
        3. Review the diagnosis and recommendations
        
        **Best results:**
        - Use well-lit photos
        - Focus on the leaf
        - Avoid blurry images
        """)
        
        st.header("Model Info")
        st.write(f"- Input Size: {IMG_SIZE[0]}x{IMG_SIZE[1]}")
        st.write(f"- Classes: {len(CLASS_NAMES)}")
        st.write(f"- Model: CNN")
    
    uploaded_file = st.file_uploader(
        "Choose a plant leaf image...",
        type=['jpg', 'jpeg', 'png'],
        help="Upload a clear image of a plant leaf"
    )
    
    if uploaded_file is not None:

        col1, col2 = st.columns(2)
        
        with col1:

            st.subheader("Uploaded Image")
            image = Image.open(uploaded_file)
            st.image(image, use_column_width=True)
        
        with col2:

            st.subheader("Analysis Results")
            
            with st.spinner('Analyzing image...'):

                processed_image = preprocess_image(image)
                
                predicted_class, confidence, all_predictions = predict_disease(model, processed_image)
                
                diagnosis = get_diagnosis(predicted_class)
            
            st.markdown(f"### {diagnosis['status']}")
            st.markdown(f"**Confidence:** {confidence:.2f}%")
            
            st.progress(confidence / 100)
            
            st.markdown("---")
            st.markdown("### Diagnosis")
            st.write(diagnosis['description'])
            
            st.markdown("### Recommendations")
            for rec in diagnosis['recommendations']:
                st.write(f"- {rec}")
        
        st.markdown("---")
        st.subheader("Detailed Prediction Probabilities")
        
        prob_cols = st.columns(3)
        for idx, (class_name, prob) in enumerate(zip(CLASS_NAMES, all_predictions)):
            with prob_cols[idx]:
                st.metric(
                    label=class_name,
                    value=f"{prob * 100:.2f}%"
                )
    
    else:

        st.info("Please upload an image to get started")
        
        st.markdown("---")
        st.subheader("Example Use Cases")
        
        example_cols = st.columns(3)
        with example_cols[0]:
            st.markdown("**Healthy Plant**")
            st.write("Green, vibrant leaves without spots or discoloration")
        
        with example_cols[1]:
            st.markdown("**Powdery Mildew**")
            st.write("White powdery coating on leaf surfaces")
        
        with example_cols[2]:
            st.markdown("**Rust Disease**")
            st.write("Orange-brown pustules or spots on leaves")

if __name__ == "__main__":
    main()
