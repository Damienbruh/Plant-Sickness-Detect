"""
FastAPI application for plant disease detection
Provides REST API endpoints for image classification
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image
import io
import numpy as np
from pathlib import Path

from model import load_trained_model, predict_image
from data_utils import preprocess_image, get_class_names

# Initialize FastAPI app
app = FastAPI(
    title="Plant Disease Detection API",
    description="API for detecting plant diseases using deep learning",
    version="1.0.0"
)

# Add CORS middleware to allow frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for model and classes
MODEL = None
CLASS_NAMES = None
MODEL_PATH = Path(__file__).parent.parent / "models" / "best_model.keras"
DATA_PATH = Path(__file__).parent.parent / "Dataset" / "Train" / "Train"


def load_model_and_classes():
    """Load the trained model and class names"""
    global MODEL, CLASS_NAMES
    
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}")
    
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found at {DATA_PATH}")
    
    MODEL = load_trained_model(str(MODEL_PATH))
    CLASS_NAMES = get_class_names(str(DATA_PATH))
    
    return MODEL, CLASS_NAMES


@app.on_event("startup")
async def startup_event():
    """Load model when API starts"""
    try:
        load_model_and_classes()
        print("Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {e}")
        raise


@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Plant Disease Detection API",
        "version": "1.0.0",
        "endpoints": {
            "predict": "/predict (POST)",
            "health": "/health (GET)",
            "classes": "/classes (GET)"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    model_loaded = MODEL is not None
    classes_loaded = CLASS_NAMES is not None
    
    return {
        "status": "healthy" if (model_loaded and classes_loaded) else "unhealthy",
        "model_loaded": model_loaded,
        "classes_loaded": classes_loaded,
        "num_classes": len(CLASS_NAMES) if CLASS_NAMES else 0
    }


@app.get("/classes")
async def get_classes():
    """Get list of available disease classes"""
    if CLASS_NAMES is None:
        raise HTTPException(status_code=503, detail="Classes not loaded")
    
    return {
        "classes": CLASS_NAMES,
        "num_classes": len(CLASS_NAMES)
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Predict plant disease from uploaded image
    
    Args:
        file: Uploaded image file (JPG, PNG, etc.)
        
    Returns:
        JSON with prediction results
    """
    # Validate model is loaded
    if MODEL is None or CLASS_NAMES is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Validate file type
    if not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="File must be an image (JPG, PNG, etc.)"
        )
    
    try:
        # Read and preprocess image
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes))
        
        # Convert to RGB if necessary
        if image.mode != "RGB":
            image = image.convert("RGB")
        
        # Preprocess image
        image_array = preprocess_image(image)
        
        # Make prediction
        result = predict_image(MODEL, image_array, CLASS_NAMES)
        
        return JSONResponse(content={
            "success": True,
            "filename": file.filename,
            "prediction": {
                "class": result["class"],
                "confidence": round(result["confidence"] * 100, 2),
                "all_probabilities": {
                    cls: round(prob * 100, 2)
                    for cls, prob in result["all_probabilities"].items()
                }
            }
        })
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing image: {str(e)}"
        )


@app.post("/predict/batch")
async def predict_batch(files: list[UploadFile] = File(...)):
    """
    Predict plant disease for multiple images
    
    Args:
        files: List of uploaded image files
        
    Returns:
        JSON with predictions for all images
    """
    if MODEL is None or CLASS_NAMES is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if len(files) > 10:
        raise HTTPException(
            status_code=400,
            detail="Maximum 10 images allowed per batch"
        )
    
    results = []
    
    for file in files:
        try:
            if not file.content_type.startswith("image/"):
                results.append({
                    "filename": file.filename,
                    "success": False,
                    "error": "File must be an image"
                })
                continue
            
            # Read and preprocess image
            image_bytes = await file.read()
            image = Image.open(io.BytesIO(image_bytes))
            
            if image.mode != "RGB":
                image = image.convert("RGB")
            
            image_array = preprocess_image(image)
            
            # Make prediction
            result = predict_image(MODEL, image_array, CLASS_NAMES)
            
            results.append({
                "filename": file.filename,
                "success": True,
                "prediction": {
                    "class": result["class"],
                    "confidence": round(result["confidence"] * 100, 2)
                }
            })
            
        except Exception as e:
            results.append({
                "filename": file.filename,
                "success": False,
                "error": str(e)
            })
    
    return JSONResponse(content={
        "total": len(files),
        "results": results
    })


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
