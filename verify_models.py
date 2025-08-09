#!/usr/bin/env python3
"""
Model Verification Script for AdVision
=====================================

This script verifies that all model paths are correctly configured
and accessible in the project.
"""

import os
import sys
from pathlib import Path

def check_model_paths():
    """Check all model paths and report their status"""
    
    print("🔍 AdVision Model Verification Report")
    print("=" * 50)
    
    # Project root
    project_root = Path(__file__).parent
    print(f"📁 Project Root: {project_root}")
    print()
    
    # Hugging Face models configuration
    huggingface_models = {
        'stable_diffusion_v1_4': 'huggingface/hub/models--CompVis--stable-diffusion-v1-4/snapshots/133a221b8aa7292a167afc5127cb63fb5005638b',
        'stable_diffusion_v1_5': 'huggingface/hub/models--runwayml--stable-diffusion-v1-5/snapshots/451f4fe16113bff5a5d2269ed5ad43b0592e9a14',
        'bart_large': 'huggingface/hub/models--facebook--bart-large/snapshots/4422181a85458e2acc57a4c1f845b9dfd60bd411'
    }
    
    # ML models configuration
    ml_models = {
        'ctr_model': 'models/ctr_model.pkl',
        'cpm_model': 'models/cpm_model.pkl',
        'roi_model': 'models/roi_classifier.pkl',
        'style_model': 'models/style_model.pkl',
        'cta_model': 'models/cta_model.pkl',
        'thumbnail_model': 'models/thumbnail_model.pkl',
        'image_model': 'models/image_model.pkl'
    }
    
    # GGUF model
    gguf_model = 'models/capybarahermes-2.5-mistral-7b.Q4_K_M.gguf'
    
    print("🤗 Hugging Face Models:")
    print("-" * 30)
    
    hf_status = {}
    for model_name, model_path in huggingface_models.items():
        full_path = project_root / model_path
        exists = full_path.exists()
        hf_status[model_name] = exists
        
        status_emoji = "✅" if exists else "❌"
        print(f"  {status_emoji} {model_name}: {model_path}")
        if exists:
            # Check for key files
            if 'stable-diffusion' in model_name:
                key_files = ['model_index.json', 'unet', 'vae', 'text_encoder']
            else:  # BART
                key_files = ['config.json', 'model.safetensors', 'pytorch_model.bin']
            
            for key_file in key_files:
                key_path = full_path / key_file
                key_exists = key_path.exists()
                key_status = "✅" if key_exists else "⚠️"
                print(f"    {key_status} {key_file}")
    
    print()
    print("📊 ML Models:")
    print("-" * 15)
    
    ml_status = {}
    for model_name, model_path in ml_models.items():
        full_path = project_root / model_path
        exists = full_path.exists()
        ml_status[model_name] = exists
        
        status_emoji = "✅" if exists else "❌"
        size_mb = full_path.stat().st_size / (1024 * 1024) if exists else 0
        print(f"  {status_emoji} {model_name}: {model_path} ({size_mb:.1f} MB)")
    
    print()
    print("🧠 GGUF Model:")
    print("-" * 15)
    
    gguf_path = project_root / gguf_model
    gguf_exists = gguf_path.exists()
    gguf_status_emoji = "✅" if gguf_exists else "❌"
    gguf_size_gb = gguf_path.stat().st_size / (1024 * 1024 * 1024) if gguf_exists else 0
    print(f"  {gguf_status_emoji} Capybara Hermes: {gguf_model} ({gguf_size_gb:.1f} GB)")
    
    print()
    print("📈 Summary:")
    print("-" * 10)
    
    total_hf = len(huggingface_models)
    available_hf = sum(hf_status.values())
    total_ml = len(ml_models)
    available_ml = sum(ml_status.values())
    
    print(f"  Hugging Face Models: {available_hf}/{total_hf} available")
    print(f"  ML Models: {available_ml}/{total_ml} available")
    print(f"  GGUF Model: {'Available' if gguf_exists else 'Not Available'}")
    
    overall_score = ((available_hf + available_ml + (1 if gguf_exists else 0)) / 
                    (total_hf + total_ml + 1)) * 100
    
    print(f"  Overall Score: {overall_score:.1f}%")
    
    if overall_score >= 80:
        print("  🎉 Excellent! Most models are available.")
    elif overall_score >= 60:
        print("  ✅ Good! Core functionality is available.")
    elif overall_score >= 40:
        print("  ⚠️ Fair! Some models are missing but basic functionality works.")
    else:
        print("  ❌ Poor! Many models are missing. Consider downloading them.")
    
    print()
    print("💡 Recommendations:")
    print("-" * 20)
    
    if available_hf < total_hf:
        print("  • Download missing Hugging Face models using:")
        print("    huggingface-cli download <model_name>")
    
    if available_ml < total_ml:
        print("  • Ensure all pickle models are in the models/ directory")
    
    if not gguf_exists:
        print("  • Download the GGUF model for enhanced local processing")
    
    print("  • Check MODEL_SETUP_GUIDE.md for detailed instructions")
    
    return overall_score >= 60

if __name__ == "__main__":
    try:
        success = check_model_paths()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"❌ Error during verification: {e}")
        sys.exit(1) 