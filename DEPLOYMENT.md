# 🚀 AgriSetu Deployment Guide

Complete step-by-step guide to deploy AgriSetu WhatsApp Bot on Render.com

## 📋 Prerequisites

1. **GitHub Account** (https://github.com)
2. **Render Account** (https://render.com) - Free tier works
3. **Twilio Account** (https://twilio.com) - Free trial available
4. **ThingESP Account** (https://thingesp.com) - Already have
5. **Trained ML Models** (.pkl files)

## 🔧 Step 1: Prepare Your Project

### 1.1 Clone/Create Project Structure
```bash
# Create project directory
mkdir agrisetu
cd agrisetu

# Create all files from this guide
# (app.py, config.py, pdf_generator.py, etc.)

# Create directories
mkdir models reports logs

# Copy your trained models to models/
cp /path/to/crop_model.pkl models/
cp /path/to/label_encoder.pkl models/
cp /path/to/month_model.pkl models/
cp /path/to/crop_month_lookup.pkl models/



agrisetu/
│
├── app.py                      # Main Flask application with Twilio webhook
├── pdf_generator.py            # Enhanced PDF generation (merged from live_agrisetu.py)
├── thingesp_client.py          # ThingESP API client with fallback
├── config.py                   # Configuration and environment variables
├── requirements.txt            # Python dependencies
├── Procfile                    # Render deployment configuration
├── runtime.txt                 # Python version specification
├── .env.example                # Example environment variables
├── .gitignore                  # Git ignore file
│
├── models/                     # ML Models directory
│   ├── crop_model.pkl
│   ├── label_encoder.pkl
│   ├── month_model.pkl
│   └── crop_month_lookup.pkl
│
├── reports/                    # Generated PDF reports (auto-created)
│   └── .gitkeep
│
└── logs/                       # Application logs (auto-created)
    └── .gitkeep