# AdVision Model Setup Guide

## Overview

This guide explains the model structure and organization in the AdVision project, specifically focusing on the three main Hugging Face models and their integration.

## Model Directory Structure

```
Advision/
├── huggingface/
│   └── hub/
│       ├── models--CompVis--stable-diffusion-v1-4/
│       │   └── snapshots/
│       │       └── 133a221b8aa7292a167afc5127cb63fb5005638b/
│       │           ├── feature_extractor/
│       │           ├── model_index.json
│       │           ├── safety_checker/
│       │           ├── scheduler/
│       │           ├── text_encoder/
│       │           ├── tokenizer/
│       │           ├── unet/
│       │           └── vae/
│       ├── models--facebook--bart-large/
│       │   └── snapshots/
│       │       └── 4422181a85458e2acc57a4c1f845b9dfd60bd411/
│       │           ├── config.json
│       │           ├── merges.txt
│       │           ├── model.safetensors
│       │           └── pytorch_model.bin
│       └── models--runwayml--stable-diffusion-v1-5/
│           └── snapshots/
│               └── 451f4fe16113bff5a5d2269ed5ad43b0592e9a14/
│                   ├── feature_extractor/
│                   ├── model_index.json
│                   ├── safety_checker/
│                   ├── scheduler/
│                   ├── text_encoder/
│                   ├── tokenizer/
│                   ├── unet/
│                   └── vae/
└── models/
    ├── ctr_model.pkl
    ├── cpm_model.pkl
    ├── roi_classifier.pkl
    ├── style_model.pkl
    ├── cta_model.pkl
    ├── thumbnail_model.pkl
    ├── image_model.pkl
    └── capybarahermes-2.5-mistral-7b.Q4_K_M.gguf
```

## Model Details

### 1. Stable Diffusion v1.4 (CompVis)
- **Location**: `huggingface/hub/models--CompVis--stable-diffusion-v1-4/`
- **Snapshot**: `133a221b8aa7292a167afc5127cb63fb5005638b`
- **Purpose**: Image generation for ad creatives
- **Components**: UNet, VAE, Text Encoder, Tokenizer, Safety Checker
- **Usage**: Primary image generation pipeline

### 2. Stable Diffusion v1.5 (RunwayML)
- **Location**: `huggingface/hub/models--runwayml--stable-diffusion-v1-5/`
- **Snapshot**: `451f4fe16113bff5a5d2269ed5ad43b0592e9a14`
- **Purpose**: Enhanced image generation with improved quality
- **Components**: UNet, VAE, Text Encoder, Tokenizer, Safety Checker
- **Usage**: Alternative/backup image generation pipeline

### 3. BART Large (Facebook)
- **Location**: `huggingface/hub/models--facebook--bart-large/`
- **Snapshot**: `4422181a85458e2acc57a4c1f845b9dfd60bd411`
- **Purpose**: Text generation, summarization, and language understanding
- **Components**: Model weights, configuration, tokenizer
- **Usage**: Ad copy generation and text processing

### 4. Capybara Hermes GGUF Model
- **Location**: `models/capybarahermes-2.5-mistral-7b.Q4_K_M.gguf`
- **Purpose**: Local LLM for prompt enhancement and text generation
- **Usage**: Enhances image prompts and provides local AI capabilities

## Configuration in Application

The models are configured in the `Config` class in `app.py`:

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

## Model Loading Process

### 1. Application Startup
When the application starts, the `ModelManager` class:
- Checks for the existence of all model files
- Loads available models into memory
- Creates mock models for missing ones
- Logs the status of all models

### 2. Model Status Reporting
The application provides detailed model status through:
- Console logging during startup
- `/api/health` endpoint for real-time status
- Model availability checks before operations

### 3. Fallback Strategy
If models are not available:
- Mock models provide realistic demo data
- Application continues to function
- Users get informative messages about model status

## Integration Points

### Image Generation
- Uses Stable Diffusion v1.5 as primary model
- Falls back to v1.4 if v1.5 is unavailable
- GGUF model enhances prompts before generation
- Mock pipeline provides demo images if models fail

### Text Generation
- BART Large handles ad copy generation
- GGUF model provides local text processing
- Mock responses ensure functionality without models

### Performance Prediction
- Uses local pickle models in `models/` directory
- Provides CTR, CPM, and ROI predictions
- Mock models ensure demo functionality

## Running the Project

### Prerequisites
1. Ensure all model directories exist
2. Verify model file integrity
3. Check available disk space (models are large)

### Startup Process
1. Run `python app.py`
2. Check console output for model status
3. Visit `/api/health` to verify all systems

### Model Verification
```bash
# Check if models exist
ls -la huggingface/hub/models--CompVis--stable-diffusion-v1-4/snapshots/
ls -la huggingface/hub/models--runwayml--stable-diffusion-v1-5/snapshots/
ls -la huggingface/hub/models--facebook--bart-large/snapshots/
ls -la models/capybarahermes-2.5-mistral-7b.Q4_K_M.gguf
```

## Troubleshooting

### Common Issues
1. **Models not found**: Check file paths and permissions
2. **Memory issues**: Models require significant RAM
3. **Loading errors**: Verify model file integrity
4. **Performance issues**: Consider using smaller models

### Solutions
1. **Re-download models**: Use Hugging Face CLI to re-download
2. **Check disk space**: Ensure sufficient storage
3. **Verify paths**: Confirm model paths in configuration
4. **Use mock mode**: Application works with mock models

## Future Enhancements

### Planned Integrations
1. **Direct Hugging Face integration**: Load models directly from Hugging Face
2. **Model optimization**: Quantization and compression
3. **Caching**: Implement model result caching
4. **Batch processing**: Handle multiple requests efficiently

### Performance Improvements
1. **GPU acceleration**: Enable CUDA support
2. **Model serving**: Separate model serving infrastructure
3. **Load balancing**: Distribute model load
4. **Monitoring**: Add model performance metrics

## API Endpoints

### Health Check
- **Endpoint**: `/api/health`
- **Method**: GET
- **Response**: Model status and availability

### Model Information
The health endpoint returns:
```json
{
  "status": "healthy",
  "health_score": 85.7,
  "model_status": {...},
  "huggingface_models": {
    "stable_diffusion_v1_4": true,
    "stable_diffusion_v1_5": true,
    "bart_large": true
  },
  "gguf_model_available": true,
  "model_paths": {...}
}
```

## Conclusion

The AdVision project is designed to work with or without the full model suite. The modular architecture ensures that:
- Core functionality is always available
- Models can be added incrementally
- Performance degrades gracefully
- Users get informative feedback about system status

For optimal performance, ensure all models are properly downloaded and accessible in the specified directories. 