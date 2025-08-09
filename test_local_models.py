#!/usr/bin/env python3
"""
Test script to verify local Hugging Face models work
"""
import os
import sys

def test_local_models():
    print("Testing local Hugging Face models...")
    
    # Check if local models exist
    model_paths = [
        'huggingface/hub/models--runwayml--stable-diffusion-v1-5/snapshots/451f4fe16113bff5a5d2269ed5ad43b0592e9a14',
        'huggingface/hub/models--CompVis--stable-diffusion-v1-4/snapshots/133a221b8aa7292a167afc5127cb63fb5005638b'
    ]
    
    for path in model_paths:
        if os.path.exists(path):
            print(f"✅ Found local model: {path}")
        else:
            print(f"❌ Local model not found: {path}")
    
    # Check GGUF model
    gguf_path = 'D:/ICT/100 activity points/Advision/models/capybarahermes-2.5-mistral-7b.Q4_K_M.gguf'
    if os.path.exists(gguf_path):
        size_gb = os.path.getsize(gguf_path) / (1024**3)
        print(f"✅ Found GGUF model: {gguf_path} ({size_gb:.1f} GB)")
    else:
        print(f"❌ GGUF model not found: {gguf_path}")
    
    # Try to import required libraries
    try:
        import torch
        print(f"✅ PyTorch version: {torch.__version__}")
    except ImportError as e:
        print(f"❌ PyTorch import error: {e}")
        return False
    
    try:
        from diffusers import StableDiffusionPipeline
        print("✅ Diffusers library imported successfully")
    except ImportError as e:
        print(f"❌ Diffusers import error: {e}")
        return False
    
    try:
        from transformers import AutoTokenizer
        print("✅ Transformers library imported successfully")
    except ImportError as e:
        print(f"❌ Transformers import error: {e}")
        return False
    
    print("\n✅ All libraries imported successfully!")
    return True

if __name__ == "__main__":
    success = test_local_models()
    if success:
        print("\n🎉 Local models test passed! Ready to use Hugging Face models.")
    else:
        print("\n❌ Local models test failed. Please check dependencies.")

