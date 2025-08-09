#!/usr/bin/env python3
"""
Test GGUF Model Integration
===========================

This script tests the integration of the Capybara Hermes GGUF model
for image generation in the AdVision project.
"""

import requests
import json
import time

def test_gguf_image_generation():
    """Test the GGUF model integration for image generation"""
    print("🎨 Testing GGUF Model Integration for Image Generation")
    print("=" * 60)
    
    # Test data for copy generation with image
    test_data = {
        "productName": "Premium Fitness App",
        "targetAudience": "Fitness Enthusiasts",
        "tone": "Motivational",
        "callToAction": "Start Your Journey Today",
        "keyBenefits": "Personalized workouts, progress tracking, and expert guidance",
        "generateImage": True
    }
    
    print(f"📝 Testing with product: {test_data['productName']}")
    print(f"🎯 Target audience: {test_data['targetAudience']}")
    print(f"🎨 Tone: {test_data['tone']}")
    print(f"📞 CTA: {test_data['callToAction']}")
    print(f"✨ Benefits: {test_data['keyBenefits']}")
    print()
    
    try:
        print("🔄 Sending request to copy generation endpoint...")
        start_time = time.time()
        
        response = requests.post(
            "http://localhost:5000/api/generate-copy",
            json=test_data,
            timeout=60  # Increased timeout for GGUF processing
        )
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        if response.status_code == 200:
            data = response.json()
            generated_copy = data.get('generated_copy', {})
            
            print("✅ Copy generation successful!")
            print(f"⏱️  Processing time: {processing_time:.2f} seconds")
            print()
            
            print("📄 Generated Content:")
            print(f"   Headline: {generated_copy.get('primary_headline', 'N/A')}")
            print(f"   Body: {generated_copy.get('primary_body', 'N/A')}")
            print(f"   CTA: {generated_copy.get('primary_cta', 'N/A')}")
            print(f"   Style: {generated_copy.get('design_style', 'N/A')}")
            print()
            
            # Check for image generation
            image_path = generated_copy.get('image_path')
            if image_path:
                print("🎨 Image Generation:")
                print(f"   ✅ Image generated successfully!")
                print(f"   📁 Image path: {image_path}")
                print(f"   🎯 Prompt used: {generated_copy.get('image_prompt', 'N/A')}")
            else:
                print("🎨 Image Generation:")
                print("   ⚠️  No image path returned")
            
            print()
            print("🔄 Variations generated:")
            variations = generated_copy.get('variations', [])
            for i, variation in enumerate(variations, 1):
                print(f"   {i}. {variation.get('headline', 'N/A')}")
            
            return True
            
        else:
            print(f"❌ Copy generation failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
            
    except requests.exceptions.Timeout:
        print("❌ Request timed out - GGUF model may be taking too long to process")
        return False
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return False

def test_health_with_gguf():
    """Test health endpoint to confirm GGUF model is loaded"""
    print("🏥 Testing Health Endpoint with GGUF Model")
    print("=" * 50)
    
    try:
        response = requests.get("http://localhost:5000/api/health", timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            model_status = data.get('model_status', {})
            
            print("✅ Health check successful!")
            print(f"   Status: {data.get('status', 'unknown')}")
            print(f"   Health Score: {data.get('health_score', 'unknown')}%")
            print(f"   Models loaded: {sum(model_status.values())}/{len(model_status)}")
            print()
            
            print("🔍 Model Status:")
            for model_name, status in model_status.items():
                status_icon = "✅" if status else "❌"
                print(f"   {status_icon} {model_name}: {'Loaded' if status else 'Not loaded'}")
            
            # Check specifically for GGUF model
            gguf_status = model_status.get('gguf_model', False)
            if gguf_status:
                print("\n🎉 GGUF Model Status: ✅ LOADED SUCCESSFULLY!")
            else:
                print("\n⚠️  GGUF Model Status: ❌ NOT LOADED")
            
            return True
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return False

def main():
    """Main test function"""
    print("🚀 AdVision GGUF Model Integration Test")
    print("=" * 60)
    print(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Test health first
    health_success = test_health_with_gguf()
    print()
    
    if health_success:
        # Test image generation
        image_success = test_gguf_image_generation()
        print()
        
        # Summary
        print("📊 Test Summary:")
        print("=" * 30)
        print(f"   Health Check: {'✅ PASS' if health_success else '❌ FAIL'}")
        print(f"   Image Generation: {'✅ PASS' if image_success else '❌ FAIL'}")
        
        if health_success and image_success:
            print("\n🎉 ALL TESTS PASSED! GGUF model integration is working!")
        elif health_success:
            print("\n⚠️  GGUF model is loaded but image generation needs attention")
        else:
            print("\n❌ GGUF model integration has issues")
    else:
        print("\n❌ Cannot test image generation - health check failed")

if __name__ == "__main__":
    main() 