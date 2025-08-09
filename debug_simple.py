#!/usr/bin/env python3
"""
Simple debug script to test pipeline step by step
"""
import os
import sys

def debug_simple():
    print("Simple debug test...")
    
    try:
        from app import ModelManager
        
        model_manager = ModelManager()
        print("✅ ModelManager created")
        
        # Test create_hf_api_pipeline directly
        pipeline = model_manager.create_hf_api_pipeline()
        print(f"✅ Pipeline created: {type(pipeline)}")
        
        # Test if it's callable
        if callable(pipeline):
            print("✅ Pipeline is callable")
            
            # Test calling it
            result = pipeline("test prompt")
            print(f"✅ Pipeline call result: {type(result)}")
            if result:
                print(f"✅ Result length: {len(result)}")
            else:
                print("❌ Result is None")
        else:
            print("❌ Pipeline is not callable")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_simple()
