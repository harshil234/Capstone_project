import os

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'dev-secret-key-change-in-production'
    # Fixed: Use get() method for environment variable
    HUGGING_FACE_TOKEN = os.environ.get('HUGGING_FACE_TOKEN') or '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'
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB
    
    # Base directory and folders
    BASE_DIR = r"D:\ICT\Advision"
    MODEL_FOLDER = os.path.join(BASE_DIR, 'models')
    UPLOAD_FOLDER = os.path.join(BASE_DIR, 'static', 'uploads')
    GENERATED_ADS_FOLDER = os.path.join(BASE_DIR, 'static', 'generated_ads')
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'webp'}
    
    # Model paths - MAKE SURE THESE ARE CLASS ATTRIBUTES
    BASE_DIR = r"D:\ICT\Advision"
    MODEL_FOLDER = os.path.join(BASE_DIR, 'models')
    UPLOAD_FOLDER = os.path.join(BASE_DIR, 'static', 'uploads')
    GENERATED_ADS_FOLDER = os.path.join(BASE_DIR, 'static', 'generated_ads')
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'webp'}
    
    # FIXED: All model paths are now class attributes
    CTR_MODEL_PATH = os.path.join(MODEL_FOLDER, 'ctr_model.pkl')
    CPM_MODEL_PATH = os.path.join(MODEL_FOLDER, 'cpm_model.pkl')
    ROI_MODEL_PATH = os.path.join(MODEL_FOLDER, 'roi_classifier.pkl')
    STYLE_MODEL_PATH = os.path.join(MODEL_FOLDER, 'style_model.pkl')
    CTA_MODEL_PATH = os.path.join(MODEL_FOLDER, 'cta_model.pkl')
    IMAGE_MODEL_PATH = os.path.join(MODEL_FOLDER, 'image_model.pkl')
    THUMBNAIL_MODEL_PATH = os.path.join(MODEL_FOLDER, 'thumbnail_model.pkl')
    # FIXED: Added missing LLM_MODEL_PATH
    LLM_MODEL_PATH = os.path.join(MODEL_FOLDER, 'capybarahermes-2.5-mistral-7b.Q4_K_M.gguf')

class ProductionConfig(Config):
    """Production configuration"""
    DEBUG = False
    FLASK_ENV = 'production'
    
class DevelopmentConfig(Config):
    DEBUG = True
    FLASK_ENV = 'development'



from flask import Flask

def create_app(config_class):
    app = Flask(__name__)
    
    # Load the config
    app.config.from_object(config_class)
    
    base_dir = r"D:\ICT\Advision\models"
    app.config.CTR_MODEL_PATH = os.path.join(base_dir, 'ctr_model.pkl')
    app.config.CPM_MODEL_PATH = os.path.join(base_dir, 'cpm_model.pkl')
    app.config.ROI_MODEL_PATH = os.path.join(base_dir, 'roi_classifier.pkl')
    app.config.STYLE_MODEL_PATH = os.path.join(base_dir, 'style_model.pkl')
    app.config.CTA_MODEL_PATH = os.path.join(base_dir, 'cta_model.pkl')
    app.config.IMAGE_MODEL_PATH = os.path.join(base_dir, 'image_model.pkl')
    app.config.THUMBNAIL_MODEL_PATH = os.path.join(base_dir, 'thumbnail_model.pkl')
    app.config.LLM_MODEL_PATH = os.path.join(base_dir, 'capybarahermes-2.5-mistral-7b.Q4_K_M.gguf')
    
    model_manager = ModelManager(app.config)
    
    
    return app


class ModelManager:
    def __init__(self, config):
        self.config = config
        self._load_models()
    
    def _load_models(self):
        base_dir = r"D:\ICT\Advision\models"
    
        model_paths = {
            'ctr': os.path.join(base_dir, 'ctr_model.pkl'),
            'cpm': os.path.join(base_dir, 'cpm_model.pkl'),
            'roi': os.path.join(base_dir, 'roi_classifier.pkl'),
            'style': os.path.join(base_dir, 'style_model.pkl'),
            'cta': os.path.join(base_dir, 'cta_model.pkl'),
            'image': os.path.join(base_dir, 'image_model.pkl'),
            'thumbnail': os.path.join(base_dir, 'thumbnail_model.pkl'),
            'llm': os.path.join(base_dir, 'capybarahermes-2.5-mistral-7b.Q4_K_M.gguf')
        }


import logging
import joblib
import os
import pandas as pd
import numpy as np
from typing import Dict, Optional

logger = logging.getLogger(__name__)

def create_mock_models():
    """Create simple mock models for development when real models aren't available"""
    try:
        from sklearn.dummy import DummyRegressor, DummyClassifier
        
        mock_models = {
            'ctr': DummyRegressor(strategy='constant', constant=2.5),
            'cpm': DummyRegressor(strategy='constant', constant=5.0),
            'roi': DummyClassifier(strategy='constant', constant='good'),
            'style': DummyClassifier(strategy='constant', constant='professional'),
            'cta': DummyClassifier(strategy='constant', constant='Learn More'),
            'image': DummyClassifier(strategy='constant', constant='product_focus'),
            'thumbnail': DummyClassifier(strategy='constant', constant='good')
        }
        
        dummy_data = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'feature2': [2, 4, 6, 8, 10]
        })
        dummy_target = [1, 2, 3, 4, 5]
        
        for name, model in mock_models.items():
            if name in ['roi', 'style', 'cta', 'image', 'thumbnail']:
                model.fit(dummy_data, ['good', 'average', 'good', 'excellent', 'good'])
            else:
                model.fit(dummy_data, dummy_target)
        
        return mock_models
    except ImportError:
        logger.warning("scikit-learn not available, creating basic mock models")
        return {}

class ModelManager:
    def __init__(self, config):
        self.config = config
        self.models: Dict[str, Optional[object]] = {}
        self.image_pipe = None
        self.llm = None
        self._load_models()
        self._load_image_pipeline()
        self._load_llm()
    
    def _load_models(self):
        """Load all ML models"""
        model_configs = {
            'ctr': self.config.get('CTR_MODEL_PATH'),
            'cpm': self.config.get('CPM_MODEL_PATH'),
            'roi': self.config.get('ROI_MODEL_PATH'),
            'style': self.config.get('STYLE_MODEL_PATH'),
            'cta': self.config.get('CTA_MODEL_PATH'),
            'image': self.config.get('IMAGE_MODEL_PATH'),
            'thumbnail': self.config.get('THUMBNAIL_MODEL_PATH')
        }
        
        real_models_loaded = 0
        for name, path in model_configs.items():
            try:
                if os.path.exists(path):
                    self.models[name] = joblib.load(path)
                    logger.info(f"✅ Loaded {name} model successfully")
                    real_models_loaded += 1
                else:
                    logger.warning(f"⚠️ Model file not found: {path}")
                    self.models[name] = None
            except Exception as e:
                logger.error(f"❌ Failed to load {name} model: {e}")
                self.models[name] = None
        
        
        if real_models_loaded == 0:
            logger.info("🔄 Loading mock models for development...")
            mock_models = create_mock_models()
            for name, model in mock_models.items():
                if self.models.get(name) is None:
                    self.models[name] = model
                    logger.info(f"✅ Loaded mock {name} model")
    
    def _load_image_pipeline(self):
        """Load Stable Diffusion pipeline once"""
        hugging_face_token = self.config.get('HUGGING_FACE_TOKEN')
        if not hugging_face_token:
            logger.warning("⚠️ HUGGING_FACE_TOKEN not set, skipping image pipeline")
            return
    
        try:
            import torch
            from huggingface_hub import login
            from diffusers import StableDiffusionPipeline
            
            login(hugging_face_token)
            
            device = "cuda" if torch.cuda.is_available() else "cpu"
            torch_dtype = torch.float16 if device == "cuda" else torch.float32
            
            
            self.image_pipe = StableDiffusionPipeline.from_pretrained(
                "CompVis/stable-diffusion-v1-4",  
                torch_dtype=torch_dtype,
                safety_checker=None,  
                requires_safety_checker=False
            ).to(device)
            
            logger.info(f"✅ Stable Diffusion loaded on {device}")
            
        except Exception as e:
            logger.error(f"❌ Failed to load Stable Diffusion: {e}")
            self.image_pipe = None
    
    def _load_llm(self):
        """Load LLM model"""
        llm_model_path = self.config.get('LLM_MODEL_PATH')
        
        if not llm_model_path or not os.path.exists(llm_model_path):
            logger.warning(f"⚠️ LLM model not found at {llm_model_path}")
            return
        
        try:
            from llama_cpp import Llama
            
           
            self.llm = Llama(
                model_path=llm_model_path,
                n_ctx=2048,
                n_threads=max(1, os.cpu_count() // 2),
                verbose=False
            )
            logger.info("✅ LLM model loaded successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to load LLM: {e}")
            self.llm = None
    
    def get_model(self, name: str):
        """Get a model by name"""
        return self.models.get(name)
    
    def is_model_available(self, name: str) -> bool:
        """Check if a model is available"""
        return self.models.get(name) is not None
    
    def get_required_models(self, required: list) -> tuple:
        """Check if all required models are available"""
        missing = [name for name in required if not self.is_model_available(name)]
        available_models = {name: self.get_model(name) for name in required if self.is_model_available(name)}
        return available_models, missing

# utils.py
import time
import logging
from functools import wraps
from flask import request, jsonify
from datetime import datetime
import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import base64
from io import BytesIO

logger = logging.getLogger(__name__)

def rate_limit(max_requests=10, window=60):
    """Rate limiting decorator"""
    requests_log = {}
    
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            client_ip = request.remote_addr
            current_time = time.time()
            
            # Clean old requests
            if client_ip in requests_log:
                requests_log[client_ip] = [
                    req_time for req_time in requests_log[client_ip]
                    if current_time - req_time < window
                ]
            else:
                requests_log[client_ip] = []
            
            # Check rate limit
            if len(requests_log[client_ip]) >= max_requests:
                return jsonify({
                    "error": "Rate limit exceeded",
                    "message": f"Maximum {max_requests} requests per {window} seconds"
                }), 429
            
            # Add current request
            requests_log[client_ip].append(current_time)
            return f(*args, **kwargs)
        return decorated_function
    return decorator

def allowed_file(filename, allowed_extensions):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions

def validate_json_request(required_fields=None):
    """Validate JSON request"""
    if not request.is_json:
        return {"error": "Content-Type must be application/json"}, 400
    
    data = request.get_json()
    if not data:
        return {"error": "Invalid JSON payload"}, 400
    
    if required_fields:
        missing_fields = [field for field in required_fields if field not in data]
        if missing_fields:
            return {
                "error": "Missing required fields",
                "missing_fields": missing_fields
            }, 400
    
    return data, None

# prediction_services.py
import pandas as pd
import numpy as np
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class PredictionService:
    def __init__(self, model_manager):
        self.model_manager = model_manager
    
    def calculate_ad_predictions(self, data):
        """Use ML models to predict CTR, CPM, and compute derived metrics"""
        required_models = ['ctr', 'cpm']
        available_models, missing = self.model_manager.get_required_models(required_models)
        
        if missing:
            logger.warning(f"Some models missing: {missing}. Using fallback calculations.")
        
        # Prepare input data
        try:
            input_df = pd.DataFrame([{
                'adSpend': data['adSpend'],
                'targetAge': data['targetAge'],
                'platform': data['platform'],
                'adFormat': data['adFormat'],
                'industry': data['industry'],
                'campaignDuration': data['campaignDuration']
            }])
        except Exception as e:
            logger.error(f"Error preparing input data: {e}")
            input_df = pd.DataFrame([{
                'feature1': data['adSpend'] / 1000,
                'feature2': data['campaignDuration']
            }])
        
        # Predict CTR and CPM using models or fallbacks
        if 'ctr' in available_models:
            try:
                predicted_ctr = available_models['ctr'].predict(input_df)[0]
            except Exception as e:
                logger.error(f"CTR model prediction failed: {e}")
                predicted_ctr = self._fallback_ctr_prediction(data)
        else:
            predicted_ctr = self._fallback_ctr_prediction(data)
        
        if 'cpm' in available_models:
            try:
                predicted_cpm = available_models['cpm'].predict(input_df)[0]
            except Exception as e:
                logger.error(f"CPM model prediction failed: {e}")
                predicted_cpm = self._fallback_cpm_prediction(data)
        else:
            predicted_cpm = self._fallback_cpm_prediction(data)
        
        # Ensure reasonable values
        predicted_ctr = max(0.5, min(10.0, predicted_ctr))
        predicted_cpm = max(1.0, min(50.0, predicted_cpm))
        
        # Derived metrics
        impressions = data['adSpend'] / predicted_cpm * 1000
        clicks = impressions * (predicted_ctr / 100)
        reach = impressions * 0.7  # simulated reach factor
        
        return {
            'ctr': round(predicted_ctr, 2),
            'impressions': int(impressions),
            'reach': int(reach),
            'clicks': int(clicks),
            'cpm': round(predicted_cpm, 2),
            'confidence_score': round(min(90, 60 + (data['adSpend'] / 1000) * 5), 1)
        }
    
    def _fallback_ctr_prediction(self, data):
        """Fallback CTR prediction based on platform and format"""
        base_ctr = 2.0
        
        # Platform adjustments
        platform_multipliers = {
            'facebook': 1.0,
            'instagram': 1.2,
            'twitter': 0.8,
            'linkedin': 0.9,
            'youtube': 1.1
        }
        
        # Format adjustments
        format_multipliers = {
            'video': 1.3,
            'image': 1.0,
            'carousel': 1.1,
            'story': 1.2
        }
        
        platform_mult = platform_multipliers.get(data.get('platform', ''), 1.0)
        format_mult = format_multipliers.get(data.get('adFormat', ''), 1.0)
        
        return base_ctr * platform_mult * format_mult
    
    def _fallback_cpm_prediction(self, data):
        """Fallback CPM prediction"""
        base_cpm = 5.0
        
        # Platform cost adjustments
        platform_costs = {
            'facebook': 1.0,
            'instagram': 1.2,
            'twitter': 0.7,
            'linkedin': 2.0,
            'youtube': 1.5
        }
        
        platform_mult = platform_costs.get(data.get('platform', ''), 1.0)
        return base_cpm * platform_mult
    
    def perform_roi_calculation(self, data):
        """Calculate ROI using ML model or fallback logic"""
        # Calculate basic financial metrics first
        total_clicks = data['adSpend'] / data['cpc']
        conversions = total_clicks * (data['conversionRate'] / 100)
        revenue = conversions * data['productPrice']
        profit = revenue * (data['profitMargin'] / 100)
        repeat_revenue = revenue * (data.get('repeatPurchases', 0) / 100)
        total_revenue = revenue + repeat_revenue
        total_profit = total_revenue * (data['profitMargin'] / 100)
        roi_value = ((total_profit - data['adSpend']) / data['adSpend']) * 100
        
        roi_model = self.model_manager.get_model('roi')
        if roi_model:
            try:
                input_df = pd.DataFrame([{
                    'adSpend': data['adSpend'],
                    'cpc': data['cpc'],
                    'conversionRate': data['conversionRate'],
                    'productPrice': data['productPrice'],
                    'profitMargin': data['profitMargin'],
                    'repeatPurchases': data.get('repeatPurchases', 0)
                }])
                
                prediction = roi_model.predict(input_df)[0]
                if hasattr(roi_model, 'predict_proba'):
                    proba = roi_model.predict_proba(input_df)[0]
                    confidence = round(max(proba) * 100, 1)
                else:
                    confidence = 85.0
                
                roi_category = prediction
            except Exception as e:
                logger.error(f"ROI model prediction failed: {e}")
                roi_category = self._classify_roi_performance(roi_value)
                confidence = 75.0
        else:
            roi_category = self._classify_roi_performance(roi_value)
            confidence = 75.0
        
        # Get insights and recommendations
        explanation = self._get_roi_insights(roi_category, roi_value)
        
        return {
            "roi_percentage": round(roi_value, 1),
            "roi_category": roi_category,
            "confidence_score": f"{confidence}%",
            "total_clicks": int(total_clicks),
            "conversions": int(conversions),
            "revenue": round(total_revenue, 2),
            "profit": round(total_profit, 2),
            "assessment": f"ROI: {roi_value:.1f}% - {roi_category.title()} Performance",
            "recommendation": explanation["strategy"],
            "insight_summary": explanation["message"]
        }
    
    def _classify_roi_performance(self, roi_value):
        """Classify ROI performance based on value"""
        if roi_value > 200:
            return "excellent"
        elif roi_value > 50:
            return "good"
        elif roi_value > 0:
            return "moderate"
        else:
            return "poor"
    
    def _get_roi_insights(self, category, roi_value):
        """Get insights and recommendations based on ROI category"""
        insights_map = {
            "excellent": {
                "message": f"Outstanding ROI of {roi_value:.1f}%! Your campaign is performing exceptionally well.",
                "strategy": "Consider scaling budget or replicating this strategy across other segments.",
            },
            "good": {
                "message": f"Good ROI of {roi_value:.1f}%. Your campaign is profitable with optimization potential.",
                "strategy": "Try A/B testing creatives or adjusting targeting to maximize returns.",
            },
            "moderate": {
                "message": f"Moderate ROI of {roi_value:.1f}%. Your campaign is profitable but could be optimized.",
                "strategy": "Improve conversion rate or reduce CPC to boost ROI.",
            },
            "poor": {
                "message": f"ROI of {roi_value:.1f}% indicates underperformance.",
                "strategy": "Re-evaluate targeting, creative, and landing page performance. Consider pausing and optimizing.",
            }
        }
        
        return insights_map.get(category, {
            "message": "ROI analysis completed.",
            "strategy": "Please review campaign metrics and adjust strategy accordingly."
        })

# text_generation.py
import logging
import pandas as pd
import numpy as np
from datetime import datetime
import os

logger = logging.getLogger(__name__)

class TextGenerationService:
    def __init__(self, model_manager):
        self.model_manager = model_manager
    
    def create_ad_copy(self, data):
        """Generate ad copy using ML models and templates"""
        required_models = ['style', 'cta', 'image']
        available_models, missing = self.model_manager.get_required_models(required_models)
        
        if missing:
            logger.warning(f"Some models missing: {missing}. Using fallbacks.")
        
        # Prepare input
        try:
            input_df = pd.DataFrame([{
                'productName': data['productName'],
                'targetAudience': data['targetAudience'],
                'adObjective': data['adObjective'],
                'tone': data['tone']
            }])
        except Exception as e:
            logger.error(f"Error preparing input: {e}")
            input_df = pd.DataFrame([{'feature1': 1, 'feature2': 2}])
        
        # ML Predictions with fallbacks
        if 'style' in available_models:
            try:
                style = available_models['style'].predict(input_df)[0]
            except:
                style = 'professional'
        else:
            style = 'professional'
            
        if 'cta' in available_models and not data.get('callToAction'):
            try:
                cta = available_models['cta'].predict(input_df)[0]
            except:
                cta = 'Learn More'
        else:
            cta = data.get('callToAction') or 'Learn More'
            
        if 'image' in available_models:
            try:
                image_type = available_models['image'].predict(input_df)[0]
            except:
                image_type = 'product_focus'
        else:
            image_type = 'product_focus'
        
        # Text generation
        benefits = data.get('keyBenefits', '').split(',') if data.get('keyBenefits') else []
        headlines = self._generate_headlines_advanced(
            data['productName'],
            data['targetAudience'],
            data['adObjective'],
            data['tone']
        )
        body_text = self._generate_body_text_advanced(
            data['productName'],
            data['targetAudience'],
            benefits,
            data['tone']
        )
        
        variations = []
        for i in range(3):
            variations.append({
                'headline': headlines[i % len(headlines)],
                'body': self._generate_body_variation(
                    data['productName'],
                    data['targetAudience'],
                    benefits,
                    data['tone'],
                    i
                ),
                'cta': self._generate_cta_variation(
                    data['adObjective'],
                    data['tone'],
                    i
                )
            })
        
        # Generate image 
        image_path = None
        image_prompt = None
        if data.get('generateImage', True):  # Default to True
            image_prompt = f"{data['tone']} ad showcasing {data['productName']} for {data['targetAudience']}. Professional background, ad style."
            
            if self.model_manager.image_pipe:
                try:
                    image_path = self._generate_ad_image(data, image_prompt)
                except Exception as e:
                    logger.error(f"AI image generation failed: {e}")
            

            if not image_path:
                image_path = self._generate_placeholder_image(data, image_prompt)
                if image_path:
                    image_prompt = f"Placeholder image for {data['productName']} - {data['targetAudience']}"
        
        return {
            'primary_headline': headlines[0],
            'primary_body': body_text,
            'primary_cta': cta,
            'design_style': style,
            'recommended_image_type': image_type,
            'image_prompt': image_prompt,
            'image_path': image_path,
            'variations': variations,
            'performance_tips': self._generate_performance_tips(data['adObjective'], data['tone']),
            'estimated_performance': self._estimate_copy_performance(data['adObjective'], data['tone'], len(benefits))
        }
    
    def _generate_ad_image(self, data, prompt):
        """Generate ad image using Stable Diffusion"""
        if not self.model_manager.image_pipe:
            return None
        
        try:
            os.makedirs(self.model_manager.config.get('GENERATED_ADS_FOLDER'), exist_ok=True)
            
            # Use faster generation parameters
            ad_image = self.model_manager.image_pipe(
                prompt,
                num_inference_steps=20,  # Reduced from default 50
                guidance_scale=7.5,      # Standard guidance
                width=512,               # Smaller size for speed
                height=512
            ).images[0]
            
            image_filename = f"{data['productName'].replace(' ', '_')}_{int(datetime.now().timestamp())}.png"
            image_path = os.path.join(self.model_manager.config.get('GENERATED_ADS_FOLDER'), image_filename)
            ad_image.save(image_path)
            
            return f"static/generated_ads/{image_filename}"
        except Exception as e:
            logger.error(f"Image generation failed: {e}")
            return None
    
    def _generate_placeholder_image(self, data, prompt):
        """Generate a simple placeholder image using PIL (much faster)"""
        try:
            from PIL import Image, ImageDraw, ImageFont
            import random
            
            os.makedirs(self.model_manager.config.get('GENERATED_ADS_FOLDER'), exist_ok=True)
            
            width, height = 512, 512
            
            # Generate a gradient background
            img = Image.new('RGB', (width, height), color='white')
            draw = ImageDraw.Draw(img)
            
            # Create a simple gradient
            for y in range(height):
                r = int(200 + (y / height) * 55)
                g = int(220 + (y / height) * 35)
                b = int(240 + (y / height) * 15)
                draw.line([(0, y), (width, y)], fill=(r, g, b))
            
            # Add some decorative elements
            colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
            for i in range(5):
                x = random.randint(50, width-50)
                y = random.randint(50, height-50)
                size = random.randint(20, 60)
                color = random.choice(colors)
                draw.ellipse([x, y, x+size, y+size], fill=color, outline='white', width=2)
            
            # Add text
            try:
                # Try to use a default font
                font_large = ImageFont.load_default()
                font_small = ImageFont.load_default()
            except:
                font_large = ImageFont.load_default()
                font_small = ImageFont.load_default()
            
            # Add product name
            product_text = data['productName'][:20]  # Limit length
            text_bbox = draw.textbbox((0, 0), product_text, font=font_large)
            text_width = text_bbox[2] - text_bbox[0]
            text_x = (width - text_width) // 2
            text_y = height // 2 - 30
            
            # Add background for text
            draw.rectangle([text_x-10, text_y-10, text_x+text_width+10, text_y+40], 
                          fill='rgba(255,255,255,0.8)', outline='#333', width=2)
            draw.text((text_x, text_y), product_text, fill='#333', font=font_large)
            
            # Add audience text
            audience_text = f"For {data['targetAudience'][:15]}"
            text_bbox = draw.textbbox((0, 0), audience_text, font=font_small)
            text_width = text_bbox[2] - text_bbox[0]
            text_x = (width - text_width) // 2
            text_y = height // 2 + 20
            
            draw.text((text_x, text_y), audience_text, fill='#666', font=font_small)
            
            # Save the image
            image_filename = f"{data['productName'].replace(' ', '_')}_placeholder_{int(datetime.now().timestamp())}.png"
            image_path = os.path.join(self.model_manager.config.get('GENERATED_ADS_FOLDER'), image_filename)
            img.save(image_path)
            
            return f"static/generated_ads/{image_filename}"
            
        except Exception as e:
            logger.error(f"Placeholder image generation failed: {e}")
            return None
    
    def _generate_headlines_advanced(self, product, audience, objective, tone):
        """Generate multiple headline variations"""
        templates = {
            'awareness': {
                'professional': [
                    f"Introducing {product}: The Solution {audience} Trust",
                    f"Discover Why {audience} Choose {product}",
                    f"{product} - Setting New Standards for {audience}"
                ],
                'casual': [
                    f"Hey {audience}! Meet Your New Favorite {product}",
                    f"{product} is Here and {audience} Are Loving It!",
                    f"Finally, a {product} Made Just for {audience}"
                ],
                'humorous': [
                    f"{audience}, Your Search for the Perfect {product} Ends Here!",
                    f"Warning: {product} May Cause Extreme Satisfaction in {audience}",
                    f"{audience} Can't Stop Talking About This {product}"
                ],
                'urgent': [
                    f"Limited Time: {product} Available for {audience}",
                    f"{audience} - Don't Miss Out on {product}",
                    f"Breaking: {product} Now Available for {audience}"
                ],
                'inspirational': [
                    f"Transform Your Experience with {product}, {audience}",
                    f"Unlock Your Potential with {product}",
                    f"{audience}, It's Time to Elevate with {product}"
                ]
            },
            'traffic': {
                'professional': [
                    f"Learn More About {product} for {audience}",
                    f"Explore {product} Solutions for {audience}",
                    f"Visit Our {product} Center for {audience}"
                ],
                'casual': [
                    f"Check Out This Amazing {product} for {audience}",
                    f"Come See What {product} Can Do for {audience}",
                    f"Take a Look at {product} - Perfect for {audience}"
                ]
            },
            'conversions': {
                'professional': [
                    f"Get {product} Today - Trusted by {audience}",
                    f"Start Your {product} Journey Now",
                    f"Experience {product} - Built for {audience}"
                ],
                'urgent': [
                    f"Limited Offer: Get {product} Now",
                    f"Don't Wait - {product} Available Today",
                    f"Last Chance: {product} for {audience}"
                ]
            }
        }
        
        # Get templates or fallback
        if objective in templates and tone in templates[objective]:
            return templates[objective][tone]
        elif objective in templates:
            first_tone = list(templates[objective].keys())[0]
            return templates[objective][first_tone]
        else:
            return [f"Discover {product}", f"{product} for {audience}", f"Join {audience} Using {product}"]
    
    def _generate_body_text_advanced(self, product, audience, benefits, tone):
        """Generate compelling body text"""
        benefit_text = ", ".join([b.strip() for b in benefits]) if benefits else "amazing benefits"
        
        templates = {
            'professional': f"{product} delivers {benefit_text} specifically designed for {audience}. Our proven solution provides measurable results that matter to your success.",
            'casual': f"Ready to experience {benefit_text}? {product} makes it super easy for {audience} like you to get exactly what you need!",
            'humorous': f"Tired of disappointing products? {product} brings you {benefit_text} with a smile (and maybe a little magic)!",
            'urgent': f"Time is running out! {product} offers {benefit_text} that {audience} need right now. Don't wait – act today!",
            'inspirational': f"Your journey to success starts here. {product} empowers {audience} with {benefit_text} to achieve extraordinary results."
        }
        
        return templates.get(tone, templates['professional'])
    
    def _generate_body_variation(self, product, audience, benefits, tone, variation_index):
        """Generate body text variations"""
        variations = [
            self._generate_body_text_advanced(product, audience, benefits, tone),
            f"Join thousands of {audience} who trust {product} for exceptional results.",
            f"See why {audience} everywhere recommend {product} to their friends and colleagues."
        ]
        return variations[variation_index % len(variations)]
    
    def _generate_cta_variation(self, objective, tone, variation_index):
        """Generate CTA variations"""
        cta_map = {
            'awareness': ['Learn More', 'Discover Now', 'Find Out More'],
            'traffic': ['Visit Website', 'Click Here', 'Explore Now'],
            'conversions': ['Get Started', 'Try Now', 'Order Today'],
            'engagement': ['Join Discussion', 'Share Now', 'Connect'],
            'leads': ['Get Free Guide', 'Download Now', 'Sign Up']
        }
        
        ctas = cta_map.get(objective, ['Learn More', 'Get Started', 'Try Now'])
        return ctas[variation_index % len(ctas)]
    
    def _generate_performance_tips(self, objective, tone):
        """Generate performance optimization tips"""
        tips = [
            "Use high-quality visuals that match your brand",
            "Test different audiences to find your best performers",
            "A/B test headlines and CTAs regularly",
            "Monitor performance and adjust budgets accordingly",
            "Use consistent branding across channels"
        ]
        
        if objective == 'conversions':
            tips.extend(["Include social proof and testimonials", "Create urgency with limited-time offers"])
        if tone == 'urgent':
            tips.append("Use countdown timers and scarcity messaging")
        
        return tips
    
    def _estimate_copy_performance(self, objective, tone, benefit_count):
        """Estimate copy performance based on inputs"""
        base_score = 60
        if objective == 'conversions': base_score += 10
        elif objective == 'awareness': base_score += 5
        if tone in ['urgent', 'inspirational']: base_score += 8
        elif tone == 'humorous': base_score += 5
        base_score += min(benefit_count * 3, 15)
        
        ctr_improve = f"{max(5, min(25, base_score - 50))}%"
        return {
            'estimated_ctr_improvement': ctr_improve,
            'engagement_score': min(95, base_score),
            'optimization_potential': 'High' if base_score > 75 else 'Medium' if base_score > 60 else 'Low'
        }

# chat_service.py
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class ChatService:
    def __init__(self, model_manager):
        self.model_manager = model_manager
    
    def generate_response(self, message):
        """Generate chatbot response using local LLM or fallback"""
        if self.model_manager.llm:
            return self._generate_llm_response(message)
        else:
            return self._generate_fallback_response(message)
    
    def _generate_llm_response(self, message):
        """Generate response using local LLM"""
        prompt = f"<|im_start|>user\n{message}\n<|im_end|>\n<|im_start|>assistant\n"
        
        try:
            response = self.model_manager.llm(
                prompt,
                max_tokens=512,
                stop=["<|im_end|>", "</s>"],
                temperature=0.7
            )
            
            bot_response = response["choices"][0]["text"].strip()
            return bot_response
            
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            return self._generate_fallback_response(message)
    
    def _generate_fallback_response(self, message):
        """Generate fallback responses when LLM is not available"""
        message_lower = message.lower()
        
        # Marketing-related responses
        if any(word in message_lower for word in ['ctr', 'click-through', 'click rate']):
            return "CTR (Click-Through Rate) is crucial for campaign success. Typical good CTRs range from 2-5% depending on your industry and platform. Focus on compelling headlines and targeted audiences to improve CTR."
        
        elif any(word in message_lower for word in ['roi', 'return on investment', 'profitability']):
            return "ROI calculation is essential for campaign optimization. Calculate it as: (Revenue - Ad Spend) / Ad Spend × 100. A good ROI is typically 200%+ for most businesses. Monitor your conversion rates and customer lifetime value."
        
        elif any(word in message_lower for word in ['ad copy', 'copywriting', 'headlines']):
            return "Great ad copy should have: 1) Attention-grabbing headlines, 2) Clear value propositions, 3) Strong call-to-actions, 4) Emotional triggers, 5) Social proof when possible. A/B test different variations to find what works best."
        
        elif any(word in message_lower for word in ['targeting', 'audience', 'demographics']):
            return "Effective targeting involves: Understanding your ideal customer demographics, interests, and behaviors. Start broad and narrow down based on performance data. Use lookalike audiences and retargeting for better results."
        
        elif any(word in message_lower for word in ['budget', 'spending', 'cost']):
            return "Budget optimization tips: Start with smaller budgets to test, allocate more to high-performing campaigns, monitor CPC and CPM regularly, and set clear KPIs. Generally, allocate 70% to proven campaigns and 30% to testing new ones."
        
        elif any(word in message_lower for word in ['platform', 'facebook', 'instagram', 'google', 'youtube']):
            return "Platform selection depends on your audience: Facebook/Instagram for B2C and visual products, LinkedIn for B2B, Google Ads for high-intent searches, YouTube for video content. Each platform has different strengths and costs."
        
        elif any(word in message_lower for word in ['conversion', 'landing page', 'optimize']):
            return "Conversion optimization involves: 1) Clear and relevant landing pages, 2) Fast loading times, 3) Mobile optimization, 4) Strong CTAs, 5) Trust signals like testimonials, 6) A/B testing different elements."
        
        elif any(word in message_lower for word in ['analytics', 'tracking', 'metrics']):
            return "Key metrics to track: CTR, CPC, CPM, conversion rate, ROAS, customer acquisition cost, lifetime value. Set up proper tracking with Google Analytics and platform pixels for accurate measurement."
        
        elif any(word in message_lower for word in ['hello', 'hi', 'hey']):
            return "Hello! I'm your AI marketing assistant. I can help you with ad campaign optimization, ROI calculations, copywriting tips, audience targeting, and general marketing strategy. What would you like to know?"
        
        elif any(word in message_lower for word in ['help', 'assist', 'support']):
            return "I can help you with: 📊 Ad performance predictions, 💰 ROI calculations, ✍️ Ad copywriting, 🎯 Audience targeting, 📈 Campaign optimization, 🖼️ Creative analysis. What specific area would you like assistance with?"
        
        elif any(word in message_lower for word in ['thank', 'thanks']):
            return "You're welcome! I'm here to help you succeed with your marketing campaigns. Feel free to ask any questions about advertising, optimization, or strategy!"
        
        else:
            return f"I understand you're asking about '{message}'. As your marketing AI assistant, I can help with campaign optimization, ROI analysis, ad copy creation, and strategic advice. Could you be more specific about what aspect of marketing you'd like help with?"

# image_analysis.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import base64
from io import BytesIO
from PIL import Image
import logging

logger = logging.getLogger(__name__)

class ImageAnalysisService:
    def __init__(self, model_manager):
        self.model_manager = model_manager
    
    def analyze_thumbnail(self, image_file):
        """Analyze thumbnail performance"""
        try:
            image = Image.open(image_file.stream).convert("RGB")
        except Exception:
            raise ValueError("Invalid image file - could not process the uploaded image")
        
        # Extract features
        width, height = image.size
        aspect_ratio = width / height
        
        gray = image.convert("L")
        pixels = np.array(gray).flatten()
        brightness = pixels.mean()
        brightness_std = pixels.std()
        
        r, g, b = image.split()
        r_mean = np.array(r).mean()
        g_mean = np.array(g).mean()
        b_mean = np.array(b).mean()
        
        contrast = brightness_std / brightness if brightness > 0 else 0
        color_balance = abs(r_mean - g_mean) + abs(g_mean - b_mean) + abs(b_mean - r_mean)
        
        ml_prediction = None
        confidence = None
        thumbnail_model = self.model_manager.get_model('thumbnail')
        
        if thumbnail_model:
            try:
                input_features = pd.DataFrame([{
                    "width": width,
                    "height": height,
                    "aspect_ratio": aspect_ratio,
                    "brightness": brightness,
                    "contrast": contrast,
                    "color_balance": color_balance,
                    "r_mean": r_mean,
                    "g_mean": g_mean,
                    "b_mean": b_mean
                }])
                
                ml_prediction = thumbnail_model.predict(input_features)[0]
                if hasattr(thumbnail_model, "predict_proba"):
                    proba = thumbnail_model.predict_proba(input_features)[0]
                    confidence = round(max(proba) * 100, 1)
            except Exception as e:
                logger.error(f"ML prediction failed: {e}")
        
        # Rule-based scoring
        performance_score, performance_factors = self._calculate_performance_score(
            brightness, contrast, aspect_ratio, width, height, color_balance
        )
        
        performance = self._get_performance_label(performance_score)
        
        # Create visualization
        analysis_chart = self._create_analysis_chart(image, pixels, r, g, b, brightness, performance_score)
        
        return {
            "width": width,
            "height": height,
            "aspect_ratio": round(aspect_ratio, 2),
            "brightness": round(brightness, 2),
            "contrast": round(contrast, 2),
            "color_balance": round(color_balance, 2),
            "performance_score": performance_score,
            "performance": performance,
            "performance_factors": performance_factors,
            "rgb_means": {
                "red": round(r_mean, 2),
                "green": round(g_mean, 2),
                "blue": round(b_mean, 2)
            },
            "analysis_chart": analysis_chart,
            "ml_prediction": ml_prediction,
            "ml_confidence": f"{confidence}%" if confidence else None,
            "timestamp": datetime.now().isoformat()
        }
    
    def _calculate_performance_score(self, brightness, contrast, aspect_ratio, width, height, color_balance):
        """Calculate rule-based performance score"""
        performance_score = 0
        performance_factors = []
        
        if brightness > 130:
            performance_score += 30
            performance_factors.append("✅ Good brightness")
        else:
            performance_factors.append("❌ Low brightness")
        
        if contrast > 0.3:
            performance_score += 25
            performance_factors.append("✅ Good contrast")
        else:
            performance_factors.append("❌ Low contrast")
        
        if 1.7 <= aspect_ratio <= 1.8:
            performance_score += 20
            performance_factors.append("✅ Optimal aspect ratio")
        else:
            performance_factors.append("⚠️ Non-standard aspect ratio")
        
        if width >= 1280 and height >= 720:
            performance_score += 15
            performance_factors.append("✅ High resolution")
        else:
            performance_factors.append("⚠️ Low resolution")
        
        if color_balance < 30:
            performance_score += 10
            performance_factors.append("✅ Balanced colors")
        else:
            performance_factors.append("⚠️ Color imbalance")
        
        return performance_score, performance_factors
    
    def _get_performance_label(self, score):
        """Get performance label based on score"""
        if score >= 80:
            return "🌟 Excellent"
        elif score >= 60:
            return "👍 Good"
        elif score >= 40:
            return "⚠️ Average"
        else:
            return "❌ Poor"
    
    def _create_analysis_chart(self, image, pixels, r, g, b, brightness, performance_score):
        """Create analysis visualization"""
        try:
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10, 8))
            
            # Brightness distribution
            ax1.hist(pixels, bins=50, color='skyblue', edgecolor='black', alpha=0.7)
            ax1.set_title('Brightness Distribution')
            ax1.axvline(brightness, color='red', linestyle='--', label=f'Mean: {brightness:.1f}')
            ax1.legend()
            
            # RGB distribution
            ax2.hist(np.array(r).flatten(), bins=50, color='red', alpha=0.5, label='Red')
            ax2.hist(np.array(g).flatten(), bins=50, color='green', alpha=0.5, label='Green')
            ax2.hist(np.array(b).flatten(), bins=50, color='blue', alpha=0.5, label='Blue')
            ax2.set_title('RGB Distribution')
            ax2.legend()
            
            # Performance metrics
            metrics = ['Brightness', 'Contrast', 'Aspect Ratio', 'Resolution', 'Color Balance']
            scores = [
                min(brightness/255*100, 100),
                min(25, 100),  # Simplified 
                80,  # Simplified
                75,  # Simplified
                70   # Simplified
            ]
            ax3.bar(metrics, scores, color=['red', 'green', 'blue', 'orange', 'purple'])
            ax3.set_title('Performance Metrics')
            ax3.set_ylim(0, 100)
            ax3.tick_params(axis='x', rotation=45)
            
            # Overall score pie chart
            ax4.pie([performance_score, 100-performance_score], 
                    labels=['Score', 'Remaining'], 
                    colors=['green', 'lightgray'], 
                    startangle=90)
            ax4.set_title(f'Overall Score: {performance_score}/100')
            
            plt.tight_layout()
            buffer = BytesIO()
            plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
            buffer.seek(0)
            encoded_img = base64.b64encode(buffer.read()).decode()
            buffer.close()
            plt.close()
            
            return f"data:image/png;base64,{encoded_img}"
            
        except Exception as e:
            logger.error(f"Chart generation failed: {e}")
            return None

# app.py - Main Flask Application
import os
import logging
from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
from datetime import datetime
import pandas as pd


# ---------------------- Logging Setup ----------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ---------------------- Flask App Initialization ----------------------
def create_app(config_class=DevelopmentConfig):
    app = Flask(__name__, static_folder='static', template_folder='templates')
    app.config.from_object(config_class)
    
    # Enable CORS
    CORS(app)
    
    # Ensure required directories exist
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(app.config['GENERATED_ADS_FOLDER'], exist_ok=True)
    os.makedirs(app.config['MODEL_FOLDER'], exist_ok=True)
    
    # Initialize services
    model_manager = ModelManager(app.config)
    prediction_service = PredictionService(model_manager)
    text_generation_service = TextGenerationService(model_manager)
    chat_service = ChatService(model_manager)
    image_analysis_service = ImageAnalysisService(model_manager)
    
    # ---------------------- Web Routes ----------------------
    
    @app.route('/')
    def home():
        return render_template('index.html')
    
    @app.route('/chatbot')
    def chatbot():
        return render_template('chatbot.html')
    
    @app.route('/thumbnail-analyzer')
    def thumbnail_analyzer():
        return render_template('thumbnail_analyzer.html')
    
    @app.route('/api/health')
    def health_check():
        """Health check endpoint"""
        model_status = {
            'ctr_model': model_manager.is_model_available('ctr'),
            'cpm_model': model_manager.is_model_available('cpm'),
            'roi_model': model_manager.is_model_available('roi'),
            'style_model': model_manager.is_model_available('style'),
            'cta_model': model_manager.is_model_available('cta'),
            'image_model': model_manager.is_model_available('image'),
            'thumbnail_model': model_manager.is_model_available('thumbnail'),
            'llm_available': model_manager.llm is not None,
            'image_pipeline_available': model_manager.image_pipe is not None
        }
        
        return jsonify({
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "version": "2.0.0",
            "models": model_status,
            "config": {
                "max_content_length": app.config['MAX_CONTENT_LENGTH'],
                "allowed_extensions": list(app.config['ALLOWED_EXTENSIONS'])
            }
        })
    
    # ---------------------- Ad Metrics Prediction API ----------------------
    
    @app.route('/api/predict-metrics', methods=['POST'])
    @rate_limit(max_requests=20, window=60)
    def predict_metrics():
        try:
            required_fields = ['adSpend', 'targetAge', 'platform', 'adFormat', 'industry', 'campaignDuration']
            data, error = validate_json_request(required_fields)
            if error:
                return jsonify(error[0]), error[1]
            
            # Validate data types and ranges
            try:
                ad_spend = float(data.get('adSpend', 0))
                campaign_duration = int(data.get('campaignDuration', 30))
                
                if ad_spend <= 0:
                    return jsonify({"error": "Ad spend must be greater than 0"}), 400
                if campaign_duration <= 0:
                    return jsonify({"error": "Campaign duration must be greater than 0"}), 400
                    
            except ValueError as e:
                return jsonify({"error": "Invalid numeric values", "details": str(e)}), 400
            
            # Get predictions
            predictions = prediction_service.calculate_ad_predictions({
                'adSpend': ad_spend,
                'targetAge': data.get('targetAge', ''),
                'platform': data.get('platform', ''),
                'adFormat': data.get('adFormat', ''),
                'industry': data.get('industry', ''),
                'campaignDuration': campaign_duration
            })
            
            return jsonify({
                "predictions": predictions,
                "input_data": data,
                "timestamp": datetime.now().isoformat(),
                "model_version": "2.0.0"
            })
            
        except RuntimeError as e:
            return jsonify({"error": str(e)}), 503
        except Exception as e:
            logger.error(f"Error in predict metrics: {e}")
            return jsonify({
                "error": "Prediction failed",
                "details": str(e)
            }), 500
    
    # ---------------------- ROI Calculator API ----------------------
    
    @app.route('/api/calculate-roi', methods=['POST'])
    @rate_limit(max_requests=20, window=60)
    def calculate_roi():
        try:
            required_fields = ['adSpend', 'cpc', 'conversionRate', 'productPrice', 'profitMargin']
            data, error = validate_json_request(required_fields)
            if error:
                return jsonify(error[0]), error[1]
            
            # Validate data types and ranges
            try:
                ad_spend = float(data.get('adSpend', 0))
                cpc = float(data.get('cpc', 0))
                conversion_rate = float(data.get('conversionRate', 0))
                product_price = float(data.get('productPrice', 0))
                profit_margin = float(data.get('profitMargin', 0))
                repeat_purchases = float(data.get('repeatPurchases', 0))
                
                if ad_spend <= 0 or cpc <= 0 or product_price <= 0:
                    return jsonify({"error": "Ad spend, CPC, and product price must be greater than 0"}), 400
                if conversion_rate < 0 or conversion_rate > 100:
                    return jsonify({"error": "Conversion rate must be between 0 and 100"}), 400
                if profit_margin < 0 or profit_margin > 100:
                    return jsonify({"error": "Profit margin must be between 0 and 100"}), 400
                    
            except ValueError as e:
                return jsonify({"error": "Invalid numeric values", "details": str(e)}), 400
            
            # Calculate ROI
            roi_results = prediction_service.perform_roi_calculation({
                'adSpend': ad_spend,
                'cpc': cpc,
                'conversionRate': conversion_rate,
                'productPrice': product_price,
                'profitMargin': profit_margin,
                'repeatPurchases': repeat_purchases
            })
            
            return jsonify({
                "roi_analysis": roi_results,
                "input_data": data,
                "timestamp": datetime.now().isoformat()
            })
            
        except RuntimeError as e:
            return jsonify({"error": str(e)}), 503
        except Exception as e:
            logger.error(f"Error in ROI calculation: {e}")
            return jsonify({
                "error": "ROI calculation failed",
                "details": str(e)
            }), 500
    
    # ---------------------- AI Ad Copy Generator API ----------------------
    
    @app.route('/api/generate-copy', methods=['POST'])
    @rate_limit(max_requests=15, window=60)
    def generate_copy():
        try:
            required_fields = ['productName', 'targetAudience', 'adObjective', 'tone']
            data, error = validate_json_request(required_fields)
            if error:
                return jsonify(error[0]), error[1]
            
            # Validate string lengths
            if len(data.get('productName', '')) > 100:
                return jsonify({"error": "Product name too long (max 100 characters)"}), 400
            if len(data.get('targetAudience', '')) > 100:
                return jsonify({"error": "Target audience too long (max 100 characters)"}), 400
            
            # Generate ad copy
            generated_copy = text_generation_service.create_ad_copy({
                'productName': data.get('productName', ''),
                'targetAudience': data.get('targetAudience', ''),
                'adObjective': data.get('adObjective', ''),
                'tone': data.get('tone', ''),
                'keyBenefits': data.get('keyBenefits', ''),
                'callToAction': data.get('callToAction', '')
            })
            
            return jsonify({
                "generated_copy": generated_copy,
                "input_data": data,
                "timestamp": datetime.now().isoformat(),
                "copy_variations": 3
            })
            
        except Exception as e:
            logger.error(f"Error in copy generation: {e}")
            return jsonify({
                "error": "Copy generation failed",
                "details": str(e)
            }), 500
    
    # ---------------------- Chatbot API ----------------------
    
    @app.route('/api/chat', methods=['POST'])
    @rate_limit(max_requests=30, window=60)
    def chat():
        try:
            data, error = validate_json_request(['message'])
            if error:
                return jsonify(error[0]), error[1]
            
            message = data.get('message', '').strip()
            if not message:
                return jsonify({"error": "Message cannot be empty"}), 400
            if len(message) > 1000:
                return jsonify({"error": "Message too long (max 1000 characters)"}), 400
            
            # Generate response
            bot_response = chat_service.generate_response(message)
            
            return jsonify({
                "response": bot_response,
                "timestamp": datetime.now().isoformat()
            })
            
        except RuntimeError as e:
            return jsonify({"error": str(e)}), 503
        except Exception as e:
            logger.error(f"Error in chat endpoint: {e}")
            return jsonify({
                "error": "Chat service unavailable",
                "details": str(e)
            }), 500
    
    # ---------------------- Thumbnail Analyzer API ----------------------
    
    @app.route('/api/analyze-thumbnail', methods=['POST'])
    @rate_limit(max_requests=20, window=60)
    def analyze_thumbnail():
        try:
            if 'thumbnail' not in request.files:
                return jsonify({"error": "No thumbnail image uploaded"}), 400
            
            image_file = request.files['thumbnail']
            if image_file.filename == '':
                return jsonify({"error": "No file selected"}), 400
            if not allowed_file(image_file.filename, app.config['ALLOWED_EXTENSIONS']):
                return jsonify({
                    "error": "Invalid file type", 
                    "allowed_types": list(app.config['ALLOWED_EXTENSIONS'])
                }), 400
            
            # Analyze image
            analysis_result = image_analysis_service.analyze_thumbnail(image_file)
            
            return jsonify(analysis_result)
            
        except ValueError as e:
            return jsonify({"error": str(e)}), 400
        except Exception as e:
            logger.error(f"Error in thumbnail analysis: {e}")
            return jsonify({
                "error": "Image analysis failed", 
                "details": str(e)
            }), 500
    
    # ---------------------- Static File Serving ----------------------
    
    @app.route('/static/generated_ads/<filename>')
    def generated_ads(filename):
        return send_from_directory(app.config['GENERATED_ADS_FOLDER'], filename)
    
    # ---------------------- Error Handlers ----------------------
    
    @app.errorhandler(413)
    def too_large(e):
        return jsonify({
            "error": "File too large",
            "message": f"Maximum file size is {app.config['MAX_CONTENT_LENGTH'] // (1024*1024)}MB"
        }), 413
    
    @app.errorhandler(404)
    def not_found(e):
        return jsonify({
            "error": "Endpoint not found",
            "message": "The requested endpoint does not exist",
            "available_endpoints": [
                "/api/health",
                "/api/predict-metrics",
                "/api/calculate-roi",
                "/api/generate-copy",
                "/api/chat",
                "/api/analyze-thumbnail"
            ]
        }), 404
    
    @app.errorhandler(500)
    def internal_error(e):
        logger.error(f"Internal server error: {e}")
        return jsonify({
            "error": "Internal server error",
            "message": "An unexpected error occurred"
        }), 500
    
    @app.errorhandler(400)
    def bad_request(e):
        return jsonify({
            "error": "Bad request",
            "message": "Invalid request format or parameters"
        }), 400
    
    return app

# ---------------------- Application Entry Point ----------------------
if __name__ == '__main__':
    # Set environment
    env = os.environ.get('FLASK_ENV', 'development')
    
    if env == 'production':
        app = create_app(ProductionConfig)
        logger.info("🚀 Starting in PRODUCTION mode")
    else:
        app = create_app(DevelopmentConfig)
        logger.info("🔧 Starting in DEVELOPMENT mode")
    
    # Validate critical configurations
    if not app.config.get('HUGGING_FACE_TOKEN'):
        logger.warning("⚠️  HUGGING_FACE_TOKEN not set - some features may be unavailable")
    
    # Start server
    port = int(os.environ.get('PORT', 5000))
    host = os.environ.get('HOST', '0.0.0.0')
    
    logger.info(f"🌐 Server starting on http://{host}:{port}")
    app.run(debug=app.config['DEBUG'], host=host, port=port)