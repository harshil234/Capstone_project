# AdVision Model Location Summary

## Overview

Your AdVision project has been successfully configured to work with the following model locations:

## Model Locations

### 🤗 Hugging Face Models (Located in `huggingface/hub/`)

1. **Stable Diffusion v1.4 (CompVis)**
   - **Path**: `huggingface/hub/models--CompVis--stable-diffusion-v1-4/snapshots/133a221b8aa7292a167afc5127cb63fb5005638b/`
   - **Purpose**: Image generation for ad creatives
   - **Status**: ✅ Available

2. **Stable Diffusion v1.5 (RunwayML)**
   - **Path**: `huggingface/hub/models--runwayml--stable-diffusion-v1-5/snapshots/451f4fe16113bff5a5d2269ed5ad43b0592e9a14/`
   - **Purpose**: Enhanced image generation with improved quality
   - **Status**: ✅ Available

3. **BART Large (Facebook)**
   - **Path**: `huggingface/hub/models--facebook--bart-large/snapshots/4422181a85458e2acc57a4c1f845b9dfd60bd411/`
   - **Purpose**: Text generation and language understanding
   - **Status**: ✅ Available

### 📊 ML Models (Located in `models/`)

All 7 ML models are available in the `models/` directory:
- `ctr_model.pkl` (3.9 MB) - Click-through rate prediction
- `cpm_model.pkl` (3.9 MB) - Cost per mille prediction
- `roi_classifier.pkl` (2.6 MB) - ROI classification
- `style_model.pkl` (2.4 MB) - Ad style analysis
- `cta_model.pkl` (2.4 MB) - Call-to-action optimization
- `thumbnail_model.pkl` (1.1 MB) - Thumbnail analysis
- `image_model.pkl` (2.5 MB) - Image performance prediction

### 🧠 GGUF Model (Located in `models/`)

- **Path**: `models/capybarahermes-2.5-mistral-7b.Q4_K_M.gguf`
- **Size**: 4.1 GB
- **Purpose**: Local LLM for prompt enhancement and text generation
- **Status**: ✅ Available

## Application Integration

### Configuration Updates

The application has been updated to properly reference these model locations:

```python
class Config:
    # Hugging Face model paths (relative to project root)
    HUGGING_FACE_MODELS = {
        'stable_diffusion_v1_4': 'huggingface/hub/models--CompVis--stable-diffusion-v1-4/snapshots/133a221b8aa7292a167afc5127cb63fb5005638b',
        'stable_diffusion_v1_5': 'huggingface/hub/models--runwayml--stable-diffusion-v1-5/snapshots/451f4fe16113bff5a5d2269ed5ad43b0592e9a14',
        'bart_large': 'huggingface/hub/models--facebook--bart-large/snapshots/4422181a85458e2acc57a4c1f845b9dfd60bd411'
    }
    
    # GGUF model path
    GGUF_MODEL_PATH = 'models/capybarahermes-2.5-mistral-7b.Q4_K_M.gguf'
```

### Model Loading Process

1. **Startup**: The `ModelManager` class checks all model paths during application startup
2. **Status Reporting**: Detailed model status is logged to console and available via `/api/health`
3. **Fallback Strategy**: Mock models ensure functionality even if some models are missing
4. **Integration**: Models are used for image generation, text processing, and predictions

## Verification

### Model Status Check

Run the verification script to check model availability:

```bash
python verify_models.py
```

**Current Status**: 🎉 **100% - Excellent! All models are available.**

### Health Check API

Visit `/api/health` to get real-time model status:

```json
{
  "status": "healthy",
  "health_score": 100.0,
  "huggingface_models": {
    "stable_diffusion_v1_4": true,
    "stable_diffusion_v1_5": true,
    "bart_large": true
  },
  "gguf_model_available": true,
  "model_paths": {...}
}
```

## Running the Project

### Quick Start

1. **Verify Models**: `python verify_models.py`
2. **Start Application**: `python app.py`
3. **Check Health**: Visit `http://localhost:5000/api/health`
4. **Access Web Interface**: Visit `http://localhost:5000`

### Model Usage

- **Image Generation**: Uses Stable Diffusion v1.5 (primary) and v1.4 (backup)
- **Text Generation**: Uses BART Large for ad copy generation
- **Local Processing**: Uses GGUF model for prompt enhancement
- **Predictions**: Uses ML models for CTR, CPM, and ROI predictions

## File Structure

```
Advision/
├── huggingface/hub/           # Hugging Face models
│   ├── models--CompVis--stable-diffusion-v1-4/
│   ├── models--runwayml--stable-diffusion-v1-5/
│   └── models--facebook--bart-large/
├── models/                    # ML models and GGUF
│   ├── *.pkl                 # 7 ML models
│   └── capybarahermes-2.5-mistral-7b.Q4_K_M.gguf
├── app.py                    # Main application (updated)
├── verify_models.py          # Model verification script
├── MODEL_SETUP_GUIDE.md      # Detailed setup guide
└── MODEL_LOCATION_SUMMARY.md # This file
```

## Key Benefits

✅ **Complete Model Suite**: All 11 models are available and properly configured  
✅ **Robust Fallbacks**: Application works even if some models are missing  
✅ **Easy Verification**: Simple script to check model status  
✅ **Clear Documentation**: Comprehensive guides for setup and troubleshooting  
✅ **Production Ready**: Proper error handling and status reporting  

## Next Steps

1. **Test the Application**: Run `python app.py` and test all features
2. **Monitor Performance**: Check `/api/health` for system status
3. **Customize Models**: Modify `Config` class to use different model paths if needed
4. **Scale Up**: Consider GPU acceleration for better performance

Your AdVision project is now fully configured and ready to run with all models properly located and integrated! 🚀 