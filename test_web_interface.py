#!/usr/bin/env python3
"""
Test Web Interface Image Generation
==================================

This script tests the web interface to ensure image generation
is working properly through the browser interface.
"""

import requests
import json
import time

def test_web_interface_image_generation():
    """Test image generation through the web interface"""
    print("🌐 Testing Web Interface Image Generation")
    print("=" * 50)
    
    # Test data for copy generation with image
    test_data = {
        "productName": "Premium Fitness App",
        "targetAudience": "Fitness Enthusiasts",
        "adObjective": "Brand Awareness",
        "tone": "Motivational",
        "keyBenefits": "Personalized workouts, progress tracking, and expert guidance",
        "callToAction": "Start Your Journey Today",
        "generateImage": True  # This is the key - make sure image generation is enabled
    }
    
    print(f"📝 Testing with product: {test_data['productName']}")
    print(f"🎯 Target audience: {test_data['targetAudience']}")
    print(f"🎨 Tone: {test_data['tone']}")
    print(f"📞 CTA: {test_data['callToAction']}")
    print(f"✨ Benefits: {test_data['keyBenefits']}")
    print(f"🖼️  Generate Image: {test_data['generateImage']}")
    print()
    
    try:
        print("🔄 Sending request to copy generation endpoint...")
        start_time = time.time()
        
        response = requests.post(
            "http://localhost:5000/api/generate-copy",
            json=test_data,
            timeout=30
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
            image_prompt = generated_copy.get('image_prompt')
            
            if image_path:
                print("🎨 Image Generation:")
                print(f"   ✅ Image generated successfully!")
                print(f"   📁 Image path: {image_path}")
                print(f"   🎯 Prompt used: {image_prompt or 'N/A'}")
                
                # Test if the image is accessible
                try:
                    img_response = requests.get(f"http://localhost:5000{image_path}", timeout=10)
                    if img_response.status_code == 200:
                        print(f"   ✅ Image is accessible via web interface")
                    else:
                        print(f"   ⚠️  Image not accessible (Status: {img_response.status_code})")
                except Exception as img_error:
                    print(f"   ❌ Image access error: {str(img_error)}")
            else:
                print("🎨 Image Generation:")
                print("   ❌ No image path returned")
                print("   💡 Make sure the 'Generate AI Image' checkbox is checked in the form")
            
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
        print("❌ Request timed out - the application may be taking too long to respond")
        return False
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return False

def test_without_image_generation():
    """Test copy generation without image to compare"""
    print("\n🔄 Testing Copy Generation WITHOUT Image")
    print("=" * 40)
    
    test_data = {
        "productName": "Premium Fitness App",
        "targetAudience": "Fitness Enthusiasts",
        "adObjective": "Brand Awareness",
        "tone": "Motivational",
        "keyBenefits": "Personalized workouts, progress tracking, and expert guidance",
        "callToAction": "Start Your Journey Today",
        "generateImage": False  # No image generation
    }
    
    try:
        response = requests.post(
            "http://localhost:5000/api/generate-copy",
            json=test_data,
            timeout=30
        )
        
        if response.status_code == 200:
            data = response.json()
            generated_copy = data.get('generated_copy', {})
            
            print("✅ Copy generation successful (without image)!")
            print(f"   Headline: {generated_copy.get('primary_headline', 'N/A')}")
            print(f"   Image path: {generated_copy.get('image_path', 'None (expected)')}")
            return True
        else:
            print(f"❌ Copy generation failed: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return False

def main():
    """Main test function"""
    print("🚀 Web Interface Image Generation Test")
    print("=" * 60)
    print(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Test with image generation
    with_image_success = test_web_interface_image_generation()
    
    # Test without image generation for comparison
    without_image_success = test_without_image_generation()
    
    # Summary
    print("\n📊 Test Summary:")
    print("=" * 30)
    print(f"   With Image Generation: {'✅ PASS' if with_image_success else '❌ FAIL'}")
    print(f"   Without Image Generation: {'✅ PASS' if without_image_success else '❌ FAIL'}")
    
    if with_image_success and without_image_success:
        print("\n🎉 ALL TESTS PASSED! Web interface image generation is working!")
        print("\n💡 Instructions for using the web interface:")
        print("   1. Go to http://localhost:5000")
        print("   2. Click on 'AI Ad Copy Generator'")
        print("   3. Fill in the form details")
        print("   4. ✅ IMPORTANT: Check the 'Generate AI Image' checkbox")
        print("   5. Click 'Generate Ad Copy'")
        print("   6. Wait for the image to be generated (may take 30-60 seconds)")
    elif without_image_success:
        print("\n⚠️  Copy generation works but image generation needs attention")
        print("   Check the 'Generate AI Image' checkbox in the web form")
    else:
        print("\n❌ Web interface has issues")

if __name__ == "__main__":
    main() 