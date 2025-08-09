#!/usr/bin/env python3
"""
Simple launcher for AdVision - bypasses all console issues
"""

import os
import sys
import warnings

# Suppress all warnings and set environment
warnings.filterwarnings('ignore')
os.environ['NO_COLOR'] = '1'
os.environ['PYTHONIOENCODING'] = 'utf-8'

def main():
    try:
        print("Starting AdVision...")
        
        # Import and create the app
        from app import create_app
        
        app = create_app()
        print("AdVision is running at: http://127.0.0.1:5000")
        
        # Run the app
        app.run(debug=False, host='127.0.0.1', port=5000, use_reloader=False)
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()

