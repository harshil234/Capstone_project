#!/usr/bin/env python3
"""
Minimal Flask app to run AdVision without console issues
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

def create_simple_app():
    """Create a minimal Flask app that imports AdVision"""
    from flask import Flask
    
    app = Flask(__name__)
    
    # Import AdVision components
    try:
        from app import create_app as create_advision_app
        advision_app = create_advision_app()
        
        # Register all routes from AdVision
        for rule in advision_app.url_map.iter_rules():
            app.add_url_rule(
                rule.rule,
                endpoint=rule.endpoint,
                view_func=advision_app.view_functions[rule.endpoint],
                methods=rule.methods
            )
        
        print("AdVision loaded successfully!")
        
    except Exception as e:
        print(f"Error loading AdVision: {e}")
        
        @app.route('/')
        def index():
            return "AdVision is starting up..."
    
    return app

if __name__ == '__main__':
    print("Starting AdVision AI-Powered Ad Analytics Platform...")
    print("Application will be available at: http://127.0.0.1:5000")
    print("Press Ctrl+C to stop the application")
    print("-" * 60)
    
    app = create_simple_app()
    app.run(
        debug=False,
        host='127.0.0.1',
        port=5000,
        use_reloader=False,
        threaded=True
    )



