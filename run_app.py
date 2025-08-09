#!/usr/bin/env python3
"""
Simple launcher for AdVision that completely bypasses console issues
"""
import os
import sys
import subprocess

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

def main():
    print("Starting AdVision AI-Powered Ad Analytics Platform...")
    print("Application will be available at: http://127.0.0.1:5000")
    print("Press Ctrl+C to stop the application")
    print("-" * 60)
    
    try:
        # Import and run the Flask app directly
        from app import create_app
        
        app = create_app()
        app.run(
            debug=False,
            host='127.0.0.1',
            port=5000,
            use_reloader=False,
            threaded=True
        )
    except Exception as e:
        print(f"Error starting AdVision: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()


