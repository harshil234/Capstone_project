#!/usr/bin/env python3
"""
Simple test script to verify AdVision application works
"""
import os
import sys
import time
import requests

def test_app():
    print("Testing AdVision application...")
    
    # Set environment variables to avoid console issues
    os.environ['NO_COLOR'] = '1'
    os.environ['PYTHONIOENCODING'] = 'utf-8'
    os.environ['FLASK_ENV'] = 'production'
    
    try:
        # Import and create app
        from app import create_app
        app = create_app()
        print("✅ App created successfully")
        
        # Test image generation
        from app import ModelManager
        model_manager = ModelManager()
        image_pipeline = model_manager.load_image_pipeline()
        print("✅ Image pipeline loaded")
        
        # Test image generation
        test_prompt = "A modern smartphone advertisement with clean design"
        print(f"Testing image generation with prompt: {test_prompt}")
        
        result = image_pipeline(test_prompt)
        if result and len(result) > 0:
            print("✅ Image generation successful!")
            print(f"Generated {len(result)} image(s)")
        else:
            print("❌ Image generation failed")
            
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return False
    
    return True

if __name__ == "__main__":
    success = test_app()
    if success:
        print("\n🎉 AdVision test passed!")
    else:
        print("\n❌ AdVision test failed!")
