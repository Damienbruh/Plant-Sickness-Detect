# Plant Disease Detection API

FastAPI web service for detecting plant diseases using deep learning.

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Ensure you have a trained model at `models/best_model.keras`

## Running the API

### Development Mode
```bash
cd src
python app.py
```

### Production Mode with Uvicorn
```bash
cd src
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

The API will be available at `http://localhost:8000`

## API Documentation

Once running, visit:
- Interactive API docs: `http://localhost:8000/docs`
- Alternative docs: `http://localhost:8000/redoc`

## Endpoints

### GET /
Basic API information

### GET /health
Check API health status

### GET /classes
Get list of available disease classes

### POST /predict
Predict disease from a single image

**Example using curl:**
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@path/to/plant_image.jpg"
```

**Example using Python:**
```python
import requests

url = "http://localhost:8000/predict"
files = {"file": open("plant_image.jpg", "rb")}
response = requests.post(url, files=files)
print(response.json())
```

**Response:**
```json
{
  "success": true,
  "filename": "plant_image.jpg",
  "prediction": {
    "class": "Healthy",
    "confidence": 98.33,
    "all_probabilities": {
      "Healthy": 98.33,
      "Powdery": 1.45,
      "Rust": 0.22
    }
  }
}
```

### POST /predict/batch
Predict diseases for multiple images (max 10)

**Example using Python:**
```python
import requests

url = "http://localhost:8000/predict/batch"
files = [
    ("files", open("image1.jpg", "rb")),
    ("files", open("image2.jpg", "rb")),
]
response = requests.post(url, files=files)
print(response.json())
```

## Testing the API

You can test the API using the interactive documentation at `http://localhost:8000/docs` or use the following Python script:

```python
import requests

# Health check
response = requests.get("http://localhost:8000/health")
print(response.json())

# Get available classes
response = requests.get("http://localhost:8000/classes")
print(response.json())

# Make prediction
files = {"file": open("test_image.jpg", "rb")}
response = requests.post("http://localhost:8000/predict", files=files)
print(response.json())
```

## CORS

The API allows cross-origin requests from any domain. For production, update the CORS settings in `app.py` to restrict allowed origins.

## Error Handling

The API returns appropriate HTTP status codes:
- 200: Success
- 400: Bad request (invalid file type, etc.)
- 500: Server error (processing failed)
- 503: Service unavailable (model not loaded)
