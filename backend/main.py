import fastapi
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import numpy as np
import pandas as pd
import joblib
import re
from PIL import Image
import io
import torch
import clip  # For generating embeddings
import base64
import warnings
import os
import sys
from sklearn.utils import Bunch
import sklearn.compose._column_transformer as ct
warnings.filterwarnings('ignore')

# --- Compatibility Patch for sklearn ---
# This ensures that `_RemainderColsList` is available when joblib loads the models
try:
    from sklearn.compose._column_transformer import _RemainderColsList
except (ImportError, AttributeError):
    class _RemainderColsList(list):
        pass
    ct._RemainderColsList = _RemainderColsList
    if 'sklearn.compose._column_transformer' in sys.modules:
        sys.modules['sklearn.compose._column_transformer']._RemainderColsList = _RemainderColsList
# --- End Patch ---

app = FastAPI(title="Product Price Prediction API")

# --- CORS middleware ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Global variables for models ---
text_model = None       # The TF-IDF + Num/Cat pipeline
image_model = None      # The CLIP + Num/Cat pipeline
clip_model = None       # The raw CLIP model (for generating embeddings)
clip_preprocess = None
device = None

# --- MODEL PATHS ---
# Assumes a 'models' folder in the same directory as this main.py
MODEL_PATH = './models/' 
TEXT_MODEL_PATH = os.path.join(MODEL_PATH, 'text_model.pkl')
IMAGE_MODEL_PATH = os.path.join(MODEL_PATH, 'image_model.pkl')

# ============================================================================
# STARTUP EVENT: Load all models
# ============================================================================

@app.on_event("startup")
async def load_models():
    global text_model, image_model, clip_model, clip_preprocess, device
    
    try:
        # Load the two saved model pipelines
        print(f"Loading text model from {TEXT_MODEL_PATH}...")
        if not os.path.exists(TEXT_MODEL_PATH):
            raise FileNotFoundError(f"Model not found at {TEXT_MODEL_PATH}")
        text_model = joblib.load(TEXT_MODEL_PATH)
        print("✓ Text model (TF-IDF) loaded")

        print(f"Loading image model from {IMAGE_MODEL_PATH}...")
        if not os.path.exists(IMAGE_MODEL_PATH):
            raise FileNotFoundError(f"Model not found at {IMAGE_MODEL_PATH}")
        image_model = joblib.load(IMAGE_MODEL_PATH)
        print("✓ Image model (CLIP) loaded")
        
        # Set device
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Load the raw CLIP model (to generate embeddings for the image_model)
        print(f"Loading CLIP model 'ViT-B/32' on {device}...")
        clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)
        print("✓ Raw CLIP model loaded")
        
    except FileNotFoundError as e:
        print(f"FATAL ERROR: Model file not found: {e}")
    except Exception as e:
        print(f"Error loading models: {e}")
        import traceback
        traceback.print_exc()

# ============================================================================
# Feature Engineering Functions (Matched *EXACTLY* to Training Script)
# ============================================================================

def extract_brand(name):
    """Extract brand from item name"""
    if not name:
        return "unknown"
    words = name.split()
    if words and len(words[0]) > 2:
        return words[0]
    return "unknown"

def extract_pack_size(text):
    """Extract pack size from item name"""
    if not text:
        return 1
    patterns = [r'\(Pack of (\d+)\)', r'Pack of (\d+)', r'(\d+)-Pack']
    for p in patterns:
        m = re.search(p, text, re.IGNORECASE)
        if m: return int(m.group(1))
    return 1

def clean_text(t):
    """Clean text for TF-IDF"""
    if not t:
        return ""
    t = t.lower()
    t = re.sub(r'\(pack of \d+\)', '', t)
    t = re.sub(r'pack of \d+', '', t)
    t = re.sub(r'\d+-pack', '', t)
    t = re.sub(r'\d+ count', '', t)
    t = re.sub(r'\d+\.?\d*\s*(oz|ounce|lb|pound|gram|kg|ml|liter|fl oz)', '', t)
    t = re.sub(r'[^a-z0-9\s]', ' ', t)
    t = re.sub(r'\s+', ' ', t).strip()
    return t

def get_clip_embedding(image: Image.Image):
    """Generate CLIP embedding from image"""
    global clip_model, clip_preprocess, device
    if clip_model is None:
        return np.zeros(512) # Return default on failure
    
    try:
        image_tensor = clip_preprocess(image).unsqueeze(0).to(device)
        with torch.no_grad():
            embedding = clip_model.encode_image(image_tensor)
        embedding_np = embedding.cpu().numpy().flatten()
        
        # Ensure correct dimension (512 for ViT-B/32)
        if len(embedding_np) == 512:
            return embedding_np
        else:
            print(f"Warning: CLIP embedding dim is {len(embedding_np)}, expected 512.")
            return np.zeros(512)
    except Exception as e:
        print(f"Error generating CLIP embedding: {e}")
        return np.zeros(512)

# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "text_model_pipeline_loaded": text_model is not None,
        "image_model_pipeline_loaded": image_model is not None,
        "clip_embed_generator_loaded": clip_model is not None,
        "device": device
    }

@app.post("/predict")
async def predict_price(
    item_name: str = Form(...),
    description: str = Form(""),
    value: float = Form(...),
    measurement_type: str = Form(...), # Not used by model, but received
    unit: str = Form(""),               # Not used by model, but received
    image: UploadFile = File(...)
):
    global text_model, image_model
    
    if text_model is None or image_model is None:
        raise HTTPException(status_code=500, detail="One or more models are not loaded.")

    try:
        # --- 1. Process Image & Text ---
        image_bytes = await image.read()
        pil_image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        
        # Return a base64 string for the frontend
        buffered = io.BytesIO()
        pil_image.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode()
        
        full_text = f"{item_name} {description}".strip()
        full_text_cleaned = clean_text(full_text)
        
        # --- 2. Generate Embeddings (as required by image_model) ---
        print("Generating CLIP embedding...")
        clip_embedding = get_clip_embedding(pil_image) # 1D array of 512
        
        # --- 3. Build Feature Dictionary (MUST contain all features for *both* models) ---
        print("Building feature set...")
        pack_size = extract_pack_size(item_name)
        
        # This is the *exact* logic from the training script
        parsed_value = float(value) if not np.isnan(value) else 0.0
        
        features = {
            # Numerical Features (used by both)
            'parsed_value': parsed_value,
            'pack_size': pack_size,
            'total_volume': parsed_value, # Training script used total_volume = parsed_value
            'brand_count': 1,  # Default for inference
            'item_name_length': len(item_name),
            'desc_length': len(description),
            'has_description': 1 if description else 0,
            'has_clip': 1,
            'log_pack_size': np.log1p(pack_size),
            'log_total_volume': np.log1p(parsed_value),
            
            # Categorical Features (used by both)
            'brand': extract_brand(item_name),
            
            # Text Feature (used by text_model)
            'full_text_cleaned': full_text_cleaned,
        }
        
        # Add CLIP features (used by image_model)
        for i, val in enumerate(clip_embedding):
            features[f'clip_{i}'] = val
        
        # --- 4. Create DataFrame ---
        df = pd.DataFrame([features])
        
        # --- 5. Make Predictions ---
        print("Predicting with text_model...")
        # text_model will select num/cat/tfidf columns and ignore clip_... columns
        log_price_text = text_model.predict(df)[0]
        
        print("Predicting with image_model...")
        # image_model will select num/cat/clip_... columns and ignore full_text_cleaned
        log_price_image = image_model.predict(df)[0]
        avg_log_price = (log_price_text + log_price_image) / 2.0
        predicted_price = np.expm1(avg_log_price)
        predicted_price = max(0.0, predicted_price)
        
        print(f"Text LogPrice: {log_price_text:.4f}, Image LogPrice: {log_price_image:.4f}")
        print(f"Blended LogPrice: {avg_log_price:.4f}")
        print(f"Final Predicted price (from np.expm1): ${predicted_price:.2f}")
        return JSONResponse(content={
            "success": True,
            "predicted_price": round(float(predicted_price), 2),
            "item_name": item_name,
            "description": description,
            "image_base64": img_base64,
            "features": {
                "pack_size": int(pack_size),
                "brand": features['brand'],
                "parsed_value": float(parsed_value),
                "log_total_volume": float(features['log_total_volume'])
            }
        })
        
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"Error in prediction: {error_details}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    if not os.path.exists(TEXT_MODEL_PATH) or not os.path.exists(IMAGE_MODEL_PATH):
        print(f"--- WARNING ---")
        print(f"One or more model files are missing from: {os.path.abspath(MODEL_PATH)}")
        print(f"Missing: {'text_model.pkl' if not os.path.exists(TEXT_MODEL_PATH) else ''}")
        print(f"Missing: {'image_model.pkl' if not os.path.exists(IMAGE_MODEL_PATH) else ''}")
        print(f"---------------")
    else:
        print(f"Found both model files in: {os.path.abspath(MODEL_PATH)}")
    
    print("Starting Uvicorn server...")
    uvicorn.run(app, host="0.0.0.0", port=8000)