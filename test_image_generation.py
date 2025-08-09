#!/usr/bin/env python3
"""
Test script to verify local image generation works
"""
import os
import sys

def test_image_generation():
    print("Testing local image generation...")
    
    try:
        # Import the ModelManager from the main app
        from app import ModelManager
        
        print("✅ ModelManager imported successfully")
        
        # Create model manager
        model_manager = ModelManager()
        print("✅ ModelManager created successfully")
        
        # Load image pipeline
        image_pipeline = model_manager.load_image_pipeline()
        print("✅ Image pipeline loaded successfully")
        
        # Test image generation
        test_prompt = "A modern smartphone advertisement with clean design, professional lighting"
        print(f"Testing with prompt: {test_prompt}")
        
        result = image_pipeline(test_prompt)
        
        if result and len(result) > 0:
            print("✅ Image generation successful!")
            print(f"Generated {len(result)} image(s)")
            
            # Save the first image
            import uuid
            filename = f"test_ad_{uuid.uuid4().hex[:8]}.png"
            filepath = os.path.join('static', 'generated_ads', filename)
            
            # Ensure directory exists
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            # Save image
            result[0].save(filepath)
            print(f"✅ Image saved to: {filepath}")
            
            return True
        else:
            print("❌ Image generation failed - no images returned")
            return False
            
    except Exception as e:
        print(f"❌ Error during image generation: {str(e)}")
        return False

if __name__ == "__main__":
    success = test_image_generation()
    if success:
        print("\n🎉 Local image generation test passed!")
    else:
        print("\n❌ Local image generation test failed.")
