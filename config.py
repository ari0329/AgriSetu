"""
Configuration management for AgriSetu WhatsApp Bot
Loads environment variables and provides default values
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env file if exists
env_path = Path('.') / '.env'
load_dotenv(dotenv_path=env_path)

class Config:
    """Application configuration"""
    
    # Flask Settings
    FLASK_ENV = os.getenv('FLASK_ENV', 'production')
    PORT = int(os.getenv('PORT', 10000))  # Render default port
    BASE_URL = os.getenv('BASE_URL', 'https://agrisetu-13.onrender.com')
    SECRET_KEY = os.getenv('SECRET_KEY', 'dev-secret-key-change-in-production')
    
    # Twilio Configuration
    TWILIO_ACCOUNT_SID = os.getenv('TWILIO_ACCOUNT_SID')
    TWILIO_AUTH_TOKEN = os.getenv('TWILIO_AUTH_TOKEN')
    TWILIO_WHATSAPP_NUMBER = os.getenv('TWILIO_WHATSAPP_NUMBER', 'whatsapp:+14155238886')
    
    # ThingESP Configuration
    THINGESP_USERNAME = os.getenv('THINGESP_USERNAME', 'Noctum')
    THINGESP_PROJECT = os.getenv('THINGESP_PROJECT', 'Agrisetu')
    THINGESP_TOKEN = os.getenv('THINGESP_TOKEN')  # Get from environment!
    THINGESP_API_URL = os.getenv('THINGESP_API_URL', 
        f'https://thingesp.com/api/users/{THINGESP_USERNAME}/projects/{THINGESP_PROJECT}/webhooks/twilio')
    
    # Model Paths
    MODEL_DIR = Path(__file__).parent / 'models'
    CROP_MODEL_PATH = MODEL_DIR / 'crop_model.pkl'
    LABEL_ENCODER_PATH = MODEL_DIR / 'label_encoder.pkl'
    MONTH_MODEL_PATH = MODEL_DIR / 'month_model.pkl'
    MONTH_LOOKUP_PATH = MODEL_DIR / 'crop_month_lookup.pkl'
    SCALER_PATH = MODEL_DIR / 'scaler.pkl'           # ← ADDED: scaler for feature normalization
    MODEL_METADATA_PATH = MODEL_DIR / 'model_metadata.pkl'  # ← ADDED: optional metadata
    
    # Report Settings
    REPORTS_DIR = Path(__file__).parent / 'reports'
    LOGS_DIR = Path(__file__).parent / 'logs'
    
    # Trigger Keywords (comma-separated)
    TRIGGER_KEYWORDS = os.getenv('TRIGGER_KEYWORDS', 'prediction,predict,crop,report')
    
    # Logging
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
    
    @classmethod
    def validate(cls):
        """Validate required configuration"""
        required_vars = [
            'TWILIO_ACCOUNT_SID',
            'TWILIO_AUTH_TOKEN',
            'THINGESP_TOKEN'
        ]
        
        missing = []
        for var in required_vars:
            if not getattr(cls, var):
                missing.append(var)
        
        if missing:
            print(f"⚠️ Missing required environment variables: {', '.join(missing)}")
            print("Please set them in Render Dashboard > Environment")
            # Don't raise error on Render - allow fallback
        
        # Create directories if they don't exist
        cls.REPORTS_DIR.mkdir(exist_ok=True)
        cls.LOGS_DIR.mkdir(exist_ok=True)
        
        return True

# Validate on import
Config.validate()