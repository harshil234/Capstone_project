#!/usr/bin/env python3
"""
Simple launcher for AdVision - bypasses Windows console issues
"""

import os
import sys
import warnings
import subprocess

# Suppress all warnings
warnings.filterwarnings('ignore')

def main():
    try:
        # Set environment variables to avoid console issues
        os.environ['NO_COLOR'] = '1'
        os.environ['PYTHONIOENCODING'] = 'utf-8'
        
        print("🚀 Starting AdVision AI-Powered Ad Analytics Platform...")
        print("📱 Application will be available at: http://127.0.0.1:5000")
        print("🛑 Press Ctrl+C to stop the application")
        print("-" * 60)
        
        # Import and run the app directly
        from app import create_app
        
        app = create_app()
        app.run(debug=False, host='127.0.0.1', port=5000, use_reloader=False)
        
    except KeyboardInterrupt:
        print("\n🛑 AdVision application stopped by user")
    except Exception as e:
        print(f"❌ Error starting AdVision: {e}")
        print("Trying alternative method...")
        
        # Alternative method using subprocess
        try:
            subprocess.run([sys.executable, "-c", 
                "from app import create_app; app = create_app(); app.run(debug=False, host='127.0.0.1', port=5000)"],
                check=True)
        except subprocess.CalledProcessError as e:
            print(f"❌ Alternative method also failed: {e}")
            sys.exit(1)

if __name__ == '__main__':
    main()

