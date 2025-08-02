# Advision - Social Media Ad Performance Predictor

A comprehensive web application for predicting ad performance metrics, calculating ROI, generating ad copy, and analyzing thumbnails using machine learning.

## Features

- 📊 **Ad Metrics Prediction**: Predict CTR, impressions, reach, and other key metrics
- 💰 **ROI Calculator**: Calculate return on investment with detailed breakdowns
- ✍️ **AI Ad Copy Generator**: Generate engaging ad copy using AI
- 🤖 **Chatbot Assistant**: Get marketing insights and advice
- 🖼️ **Thumbnail Analyzer**: Analyze image performance with ML insights
- 🏥 **System Health Check**: Monitor service status and model availability

## Quick Start

### Option 1: Using the startup script (Recommended)

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the application**:
   ```bash
   python run.py
   ```

3. **Open your browser** and go to: `http://localhost:5000`

### Option 2: Manual startup

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Set environment variables** (optional):
   ```bash
   export FLASK_ENV=development
   export HUGGING_FACE_TOKEN=your_token_here
   ```

3. **Run the Flask app**:
   ```bash
   python app.py
   ```

4. **Open your browser** and go to: `http://localhost:5000`

## Demo Mode

If the Flask server is not running, the application will automatically switch to **demo mode**, where:
- All buttons and features work with simulated data
- No backend server is required
- Perfect for testing the interface

## API Endpoints

When the server is running, the following API endpoints are available:

- `GET /api/health` - System health check
- `POST /api/predict-metrics` - Predict ad performance metrics
- `POST /api/calculate-roi` - Calculate ROI
- `POST /api/generate-copy` - Generate ad copy
- `POST /api/chat` - Chat with AI assistant
- `POST /api/analyze-thumbnail` - Analyze thumbnail images

## Project Structure

```
Advision/
├── app.py                 # Main Flask application
├── run.py                 # Startup script
├── requirements.txt       # Python dependencies
├── models/               # ML model files
├── static/               # Static assets
│   ├── css/
│   ├── js/
│   ├── uploads/
│   └── generated_ads/
└── templates/            # HTML templates
    ├── index.html
    ├── chatbot.html
    ├── thumbnail_analyzer.html
    └── base.html
```

## Dependencies

- **Flask**: Web framework
- **Flask-CORS**: Cross-origin resource sharing
- **pandas & numpy**: Data processing
- **scikit-learn**: Machine learning models
- **Pillow**: Image processing
- **matplotlib**: Data visualization
- **llama-cpp-python**: Local LLM support
- **huggingface-hub**: AI model integration

## Troubleshooting

### Buttons not working?
- Make sure the Flask server is running
- Check the browser console for errors
- The app will work in demo mode even without the server

### Import errors?
- Install missing packages: `pip install -r requirements.txt`
- Use the startup script to check dependencies: `python run.py`

### Port already in use?
- Change the port in `app.py` or kill the process using port 5000
- Alternative: `python app.py --port 5001`

## Development

The application is built with:
- **Backend**: Flask (Python)
- **Frontend**: HTML5, CSS3, JavaScript (Bootstrap 5)
- **ML**: scikit-learn, custom models
- **AI**: Local LLM integration

## License

This project is created by Harshil, Kaushal, Sachin & Thirupathi.

---

**Note**: This application includes both real ML model integration and demo mode functionality. When the Flask server is not running, all features work with simulated data for demonstration purposes. 