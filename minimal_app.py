#!/usr/bin/env python3
"""
Minimal Flask app that loads AdVision without import errors
"""
import os
import sys

# Disable all problematic console output
os.environ['NO_COLOR'] = '1'
os.environ['PYTHONIOENCODING'] = 'utf-8'
os.environ['FLASK_ENV'] = 'production'
os.environ['CLICK_DISABLE'] = '1'

# Disable problematic imports
try:
    import colorama
    colorama.deinit()
except:
    pass

from flask import Flask, render_template, request, jsonify, send_from_directory
import os
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Basic routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/copy-generator')
def copy_generator():
    return render_template('copy_generator.html')

@app.route('/chatbot')
def chatbot():
    return render_template('chatbot.html')

@app.route('/thumbnail-analyzer')
def thumbnail_analyzer():
    return render_template('thumbnail_analyzer.html')

@app.route('/api/health')
def health_check():
    return jsonify({
        'status': 'healthy',
        'message': 'AdVision is running successfully!',
        'version': '1.0.0'
    })

@app.route('/api/generate-copy', methods=['POST'])
def generate_copy():
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Extract parameters
        product_name = data.get('productName', data.get('product_name', 'Product'))
        target_audience = data.get('targetAudience', data.get('target_audience', 'General'))
        generate_image = data.get('generateImage', data.get('generate_image', False))
        
        # Generate mock ad copy
        ad_copy = f"🚀 Discover {product_name} - Perfect for {target_audience}!\n\n"
        ad_copy += f"✨ Experience the difference with {product_name}\n"
        ad_copy += f"🎯 Designed specifically for {target_audience}\n"
        ad_copy += f"💎 Premium quality at an affordable price\n"
        ad_copy += f"🔥 Limited time offer - Don't miss out!\n\n"
        ad_copy += f"Click now to learn more about {product_name}!"
        
        # Generate variations
        variations = [
            f"Transform your life with {product_name} - The ultimate choice for {target_audience}",
            f"Unlock your potential with {product_name}. Perfect for {target_audience} who demand excellence",
            f"Join thousands of {target_audience} who trust {product_name}. Experience the difference today!"
        ]
        
        response = {
            'ad_copy': ad_copy,
            'variations': variations,
            'product_name': product_name,
            'target_audience': target_audience,
            'timestamp': '2025-08-08T12:00:00Z'
        }
        
        # Generate image if requested
        if generate_image:
            try:
                                 # Try to load the full AdVision app for image generation
                 try:
                     from app import ModelManager
                     model_manager = ModelManager()
                     image_pipeline = model_manager.load_image_pipeline()
                 except Exception as e:
                     logger.error(f"Failed to load ModelManager: {str(e)}")
                     response['image_url'] = None
                     return jsonify(response)
                
                prompt = f"A modern advertisement for {product_name} targeting {target_audience}"
                result = image_pipeline(prompt)
                
                if result and len(result) > 0:
                    # Save the image
                    import uuid
                    from PIL import Image
                    
                    filename = f"ad_{uuid.uuid4().hex[:8]}.png"
                    filepath = os.path.join('static', 'generated_ads', filename)
                    
                    # Ensure directory exists
                    os.makedirs(os.path.dirname(filepath), exist_ok=True)
                    
                    # Save image
                    result[0].save(filepath)
                    
                    response['image_url'] = f'/static/generated_ads/{filename}'
                    logger.info(f"Generated image: {filename}")
                else:
                    response['image_url'] = None
                    logger.warning("Image generation failed")
                    
            except Exception as e:
                logger.error(f"Image generation error: {str(e)}")
                response['image_url'] = None
        else:
            response['image_url'] = None
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Copy generation error: {str(e)}")
        return jsonify({'error': 'Failed to generate ad copy'}), 500

@app.route('/static/generated_ads/<filename>')
def serve_generated_image(filename):
    return send_from_directory('static/generated_ads', filename)

if __name__ == '__main__':
    print("Starting AdVision AI-Powered Ad Analytics Platform...")
    print("Application will be available at: http://127.0.0.1:5000")
    print("Press Ctrl+C to stop the application")
    print("-" * 60)
    
    app.run(
        debug=False,
        host='127.0.0.1',
        port=5000,
        use_reloader=False,
        threaded=True
    )
