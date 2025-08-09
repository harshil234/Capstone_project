#!/usr/bin/env python3
"""
Debug script to check what create_hf_api_pipeline returns
"""
import os
import sys

def debug_pipeline():
    print("Debugging pipeline creation...")
    
    try:
        from app import ModelManager
        
        model_manager = ModelManager()
        print("✅ ModelManager created")
        
        # Check what create_hf_api_pipeline returns
        pipeline_class = model_manager.create_hf_api_pipeline()
        print(f"✅ create_hf_api_pipeline returned: {type(pipeline_class)}")
        print(f"✅ Pipeline class name: {pipeline_class.__name__ if hasattr(pipeline_class, '__name__') else 'No name'}")
        
        # Try to instantiate it
        try:
            pipeline_instance = pipeline_class()
            print(f"✅ Pipeline instance created: {type(pipeline_instance)}")
            
            # Test calling it
            result = pipeline_instance("test prompt")
            print(f"✅ Pipeline call result: {type(result)}")
            if result:
                print(f"✅ Result length: {len(result)}")
            
        except Exception as e:
            print(f"❌ Error instantiating pipeline: {e}")
            
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    debug_pipeline()



