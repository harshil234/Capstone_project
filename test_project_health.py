#!/usr/bin/env python3
"""
AdVision Project Health Check
============================

This script performs a comprehensive health check of the AdVision application
to verify all features are working correctly.
"""

import requests
import json
import time
import os
from datetime import datetime

# Configuration
BASE_URL = "http://localhost:5000"
TEST_IMAGE_PATH = "test_thumbnail.png"

def print_header(title):
    """Print a formatted header"""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")

def print_section(title):
    """Print a formatted section"""
    print(f"\n{'-'*40}")
    print(f"  {title}")
    print(f"{'-'*40}")

def test_health_endpoint():
    """Test the health check endpoint"""
    print_section("Testing Health Endpoint")
    
    try:
        response = requests.get(f"{BASE_URL}/api/health", timeout=10)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Health check successful")
            print(f"   Status: {data.get('status', 'unknown')}")
            print(f"   Health Score: {data.get('health_score', 'unknown')}%")
            print(f"   Version: {data.get('version', 'unknown')}")
            
            # Check model status
            model_status = data.get('model_status', {})
            print(f"   Models loaded: {sum(model_status.values())}/{len(model_status)}")
            
            return True
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Health check error: {str(e)}")
        return False

def test_prediction_endpoint():
    """Test the prediction endpoint"""
    print_section("Testing Prediction Endpoint")
    
    test_data = {
        "budget": 1000,
        "age_group": "25-34",
        "platform": "Facebook",
        "ad_format": "Image",
        "industry": "E-commerce",
        "campaign_duration": 30
    }
    
    try:
        response = requests.post(
            f"{BASE_URL}/api/predict-metrics",
            json=test_data,
            timeout=10
        )
        
        if response.status_code == 200:
            data = response.json()
            predictions = data.get('predictions', {})
            print(f"✅ Prediction successful")
            print(f"   CTR: {predictions.get('ctr_prediction', 'N/A')}%")
            print(f"   CPM: ${predictions.get('cpm_prediction', 'N/A')}")
            print(f"   Impressions: {predictions.get('estimated_impressions', 'N/A')}")
            print(f"   Reach: {predictions.get('estimated_reach', 'N/A')}")
            print(f"   Clicks: {predictions.get('estimated_clicks', 'N/A')}")
            return True
        else:
            print(f"❌ Prediction failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Prediction error: {str(e)}")
        return False

def test_roi_endpoint():
    """Test the ROI calculation endpoint"""
    print_section("Testing ROI Endpoint")
    
    test_data = {
        "ad_spend": 1000,
        "revenue": 3000,
        "cost_of_goods": 1500,
        "cpc": 2.5,
        "conversionRate": 3.5
    }
    
    try:
        response = requests.post(
            f"{BASE_URL}/api/calculate-roi",
            json=test_data,
            timeout=10
        )
        
        if response.status_code == 200:
            data = response.json()
            roi_analysis = data.get('roi_analysis', {})
            print(f"✅ ROI calculation successful")
            print(f"   ROI: {roi_analysis.get('roi_percentage', 'N/A')}%")
            print(f"   ROAS: {roi_analysis.get('roas_percentage', 'N/A')}%")
            print(f"   Net Profit: ${roi_analysis.get('net_profit', 'N/A')}")
            print(f"   Profit Margin: {roi_analysis.get('profit_margin', 'N/A')}%")
            print(f"   Category: {roi_analysis.get('roi_category', 'N/A')}")
            return True
        else:
            print(f"❌ ROI calculation failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ ROI calculation error: {str(e)}")
        return False

def test_copy_generation():
    """Test the AI copy generation endpoint"""
    print_section("Testing Copy Generation")
    
    test_data = {
        "productName": "Test Product",
        "targetAudience": "Young Professionals",
        "tone": "Professional",
        "callToAction": "Get Started Today",
        "keyBenefits": "Amazing benefits and features",
        "generateImage": True
    }
    
    try:
        response = requests.post(
            f"{BASE_URL}/api/generate-copy",
            json=test_data,
            timeout=30
        )
        
        if response.status_code == 200:
            data = response.json()
            generated_copy = data.get('generated_copy', {})
            print(f"✅ Copy generation successful")
            print(f"   Headline: {generated_copy.get('primary_headline', 'N/A')}")
            print(f"   Body: {generated_copy.get('primary_body', 'N/A')[:50]}...")
            print(f"   CTA: {generated_copy.get('primary_cta', 'N/A')}")
            print(f"   Variations: {data.get('copy_variations', 'N/A')}")
            
            # Check if image was generated
            image_path = generated_copy.get('image_path')
            if image_path:
                print(f"   Image: {image_path}")
            else:
                print(f"   Image: Not generated")
            
            return True
        else:
            print(f"❌ Copy generation failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Copy generation error: {str(e)}")
        return False

def test_chatbot():
    """Test the chatbot endpoint"""
    print_section("Testing Chatbot")
    
    test_messages = [
        "What is AdVision?",
        "How does ROI calculation work?",
        "Tell me about CTR predictions"
    ]
    
    success_count = 0
    for message in test_messages:
        try:
            response = requests.post(
                f"{BASE_URL}/api/chat",
                json={"message": message},
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                response_text = data.get('response', '')
                print(f"✅ Chat response received for: '{message[:30]}...'")
                print(f"   Response: {response_text[:50]}...")
                success_count += 1
            else:
                print(f"❌ Chat failed for: '{message[:30]}...'")
        except Exception as e:
            print(f"❌ Chat error for '{message[:30]}...': {str(e)}")
    
    return success_count == len(test_messages)

def test_thumbnail_analysis():
    """Test the thumbnail analysis endpoint"""
    print_section("Testing Thumbnail Analysis")
    
    # Check if test image exists
    if not os.path.exists(TEST_IMAGE_PATH):
        print(f"⚠️  Test image not found: {TEST_IMAGE_PATH}")
        print(f"   Creating a simple test image...")
        
        # Create a simple test image
        try:
            from PIL import Image, ImageDraw
            
            # Create a 300x200 test image
            img = Image.new('RGB', (300, 200), color='white')
            draw = ImageDraw.Draw(img)
            
            # Draw a simple rectangle
            draw.rectangle([50, 50, 250, 150], outline='blue', width=3)
            draw.text((150, 100), "Test Image", fill='blue', anchor="mm")
            
            img.save(TEST_IMAGE_PATH)
            print(f"✅ Test image created: {TEST_IMAGE_PATH}")
        except Exception as e:
            print(f"❌ Failed to create test image: {str(e)}")
            return False
    
    try:
        with open(TEST_IMAGE_PATH, 'rb') as f:
            files = {'thumbnail': f}
            response = requests.post(
                f"{BASE_URL}/api/analyze-thumbnail",
                files=files,
                timeout=15
            )
        
        if response.status_code == 200:
            data = response.json()
            analysis = data.get('analysis', {})
            print(f"✅ Thumbnail analysis successful")
            print(f"   Performance Score: {analysis.get('performance_score', 'N/A')}")
            print(f"   Performance: {analysis.get('performance', 'N/A')}")
            print(f"   Aspect Ratio: {analysis.get('aspect_ratio', 'N/A')}")
            print(f"   Brightness: {analysis.get('brightness', 'N/A')}")
            print(f"   Contrast: {analysis.get('contrast', 'N/A')}")
            
            recommendations = analysis.get('recommendations', [])
            if recommendations:
                print(f"   Recommendations: {len(recommendations)} items")
            
            return True
        else:
            print(f"❌ Thumbnail analysis failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Thumbnail analysis error: {str(e)}")
        return False

def test_frontend_pages():
    """Test frontend pages are accessible"""
    print_section("Testing Frontend Pages")
    
    pages = [
        ("/", "Homepage"),
        ("/chatbot", "Chatbot Page"),
        ("/thumbnail-analyzer", "Thumbnail Analyzer")
    ]
    
    success_count = 0
    for path, name in pages:
        try:
            response = requests.get(f"{BASE_URL}{path}", timeout=10)
            if response.status_code == 200:
                print(f"✅ {name} accessible")
                success_count += 1
            else:
                print(f"❌ {name} failed: {response.status_code}")
        except Exception as e:
            print(f"❌ {name} error: {str(e)}")
    
    return success_count == len(pages)

def main():
    """Main test function"""
    print_header("AdVision Project Health Check")
    print(f"Testing application at: {BASE_URL}")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Test results tracking
    test_results = []
    
    # Run all tests
    tests = [
        ("Health Endpoint", test_health_endpoint),
        ("Prediction Endpoint", test_prediction_endpoint),
        ("ROI Endpoint", test_roi_endpoint),
        ("Copy Generation", test_copy_generation),
        ("Chatbot", test_chatbot),
        ("Thumbnail Analysis", test_thumbnail_analysis),
        ("Frontend Pages", test_frontend_pages)
    ]
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            test_results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} crashed: {str(e)}")
            test_results.append((test_name, False))
    
    # Summary
    print_header("Test Summary")
    
    passed = sum(1 for _, result in test_results if result)
    total = len(test_results)
    
    print(f"Tests Passed: {passed}/{total}")
    print(f"Success Rate: {(passed/total)*100:.1f}%")
    
    print("\nDetailed Results:")
    for test_name, result in test_results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {test_name}: {status}")
    
    # Overall assessment
    if passed == total:
        print(f"\n🎉 ALL TESTS PASSED! The AdVision project is working perfectly!")
    elif passed >= total * 0.8:
        print(f"\n✅ MOSTLY WORKING! {passed}/{total} tests passed. Minor issues detected.")
    elif passed >= total * 0.5:
        print(f"\n⚠️  PARTIALLY WORKING! {passed}/{total} tests passed. Some issues need attention.")
    else:
        print(f"\n❌ MAJOR ISSUES! Only {passed}/{total} tests passed. Significant problems detected.")
    
    print(f"\nHealth check completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main() 