"""
AgriSetu WhatsApp Bot - Main Flask Application
Handles Twilio webhooks and orchestrates the prediction flow
"""

import os
import logging
import traceback
import threading
from datetime import datetime
from pathlib import Path
from flask import Flask, request, send_file, jsonify
from twilio.twiml.messaging_response import MessagingResponse
from twilio.rest import Client
from twilio.request_validator import RequestValidator

from config import Config
from thingesp_client import get_sensor_data
from pdf_generator import generate_pdf

# ================== SETUP LOGGING ==================
log_file = Config.LOGS_DIR / f"agrisetu_{datetime.now().strftime('%Y%m%d')}.log"

logging.basicConfig(
    level=getattr(logging, Config.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# ================== FLASK APP ==================
app = Flask(__name__)
app.config['SECRET_KEY'] = Config.SECRET_KEY

# ================== TWILIO CLIENT ==================
twilio_client = None
if Config.TWILIO_ACCOUNT_SID and Config.TWILIO_AUTH_TOKEN:
    try:
        twilio_client = Client(Config.TWILIO_ACCOUNT_SID, Config.TWILIO_AUTH_TOKEN)
        logger.info("✅ Twilio client initialized")
    except Exception as e:
        logger.error(f"❌ Twilio client initialization failed: {e}")
else:
    logger.error("❌ Twilio credentials missing")

# ================== HELPER FUNCTIONS ==================
def validate_twilio_request() -> bool:
    """Validate that request comes from Twilio"""
    if Config.FLASK_ENV == 'development':
        return True  # Skip validation in development
    
    validator = RequestValidator(Config.TWILIO_AUTH_TOKEN)
    url = f"{Config.BASE_URL}/whatsapp"
    params = request.form.to_dict()
    signature = request.headers.get('X-Twilio-Signature', '')
    
    return validator.validate(url, params, signature)

def is_trigger_message(message: str) -> bool:
    """Check if message contains trigger keyword"""
    triggers = Config.TRIGGER_KEYWORDS.lower().split(',')
    message_lower = message.lower().strip()
    
    return any(trigger in message_lower for trigger in triggers)

def send_whatsapp_message(to: str, body: str) -> bool:
    """Send text message via WhatsApp"""
    try:
        if not twilio_client:
            logger.error("Twilio client not initialized")
            return False
            
        message = twilio_client.messages.create(
            from_=Config.TWILIO_WHATSAPP_NUMBER,
            to=to,
            body=body
        )
        logger.info(f"✅ Message sent to {to}: {message.sid}")
        return True
    except Exception as e:
        logger.error(f"Failed to send message: {e}")
        return False

def send_whatsapp_pdf(to: str, pdf_path: str, caption: str) -> bool:
    """Send PDF file via WhatsApp"""
    try:
        if not twilio_client:
            logger.error("Twilio client not initialized")
            return False
            
        # Create public URL for PDF
        pdf_filename = Path(pdf_path).name
        media_url = f"{Config.BASE_URL}/reports/{pdf_filename}"
        
        message = twilio_client.messages.create(
            from_=Config.TWILIO_WHATSAPP_NUMBER,
            to=to,
            body=caption,
            media_url=[media_url]
        )
        logger.info(f"✅ PDF sent to {to}: {message.sid}")
        return True
    except Exception as e:
        logger.error(f"Failed to send PDF: {e}")
        return False

def cleanup_old_reports(max_age_hours: int = 24):
    """Delete PDF reports older than max_age_hours"""
    try:
        cutoff = datetime.now().timestamp() - (max_age_hours * 3600)
        for pdf_file in Config.REPORTS_DIR.glob("*.pdf"):
            if pdf_file.stat().st_mtime < cutoff:
                pdf_file.unlink()
                logger.info(f"🗑️ Deleted old report: {pdf_file.name}")
    except Exception as e:
        logger.error(f"Cleanup error: {e}")

def process_prediction_background(sender: str):
    """
    Process prediction request in background thread
    """
    try:
        logger.info(f"📡 Starting background processing for {sender}")
        
        # 1. Fetch real-time sensor data
        logger.info(f"📡 Fetching sensor data...")
        sensor_data = get_sensor_data()
        logger.info(f"📊 Sensor data received: {sensor_data}")
        
        # 2. Generate PDF report
        logger.info(f"📄 Generating PDF report...")
        pdf_path, crop, growth_months = generate_pdf(sensor_data)
        logger.info(f"📄 PDF generated: {pdf_path}, Crop: {crop}, Months: {growth_months}")
        
        # 3. Send PDF via WhatsApp
        logger.info(f"📤 Sending PDF to {sender}...")
        caption = f"🌱 *Crop Prediction Report*\n\n*Recommended Crop:* {crop}\n*Growth Duration:* {growth_months} months\n\n📊 Detailed report attached below."
        
        success = send_whatsapp_pdf(sender, pdf_path, caption)
        
        if success:
            logger.info(f"✅ Successfully processed prediction for {sender}")
            # Cleanup old reports periodically
            cleanup_old_reports()
        else:
            # Fallback - send text-only prediction
            fallback_msg = f"🌱 *Crop Recommendation:* {crop}\n\n⏱️ *Growth Duration:* {growth_months} months\n\n⚠️ PDF delivery failed. Please try again."
            send_whatsapp_message(sender, fallback_msg)
            
    except Exception as e:
        logger.error(f"❌ Prediction error: {e}")
        logger.error(traceback.format_exc())
        
        # Send error message to user
        error_msg = "❌ *Sorry, something went wrong.*\n\nWe couldn't generate your report. Please try again in a few minutes."
        send_whatsapp_message(sender, error_msg)

# ================== ROUTES ==================
@app.route('/')
def home():
    """Health check endpoint"""
    return jsonify({
        'status': 'online',
        'service': 'AgriSetu WhatsApp Bot',
        'version': '1.0.0',
        'timestamp': datetime.now().isoformat()
    })

@app.route('/health')
def health():
    """Detailed health check"""
    # Import here to avoid circular import
    from pdf_generator import pdf_generator as pg
    
    return jsonify({
        'status': 'healthy',
        'twilio_configured': twilio_client is not None,
        'models_loaded': pg.models_loaded if pg else False,
        'reports_count': len(list(Config.REPORTS_DIR.glob("*.pdf"))),
    })

@app.route('/whatsapp', methods=['POST', 'GET'])
def whatsapp_webhook():
    """
    Twilio WhatsApp webhook endpoint
    Receives messages and responds with crop predictions
    """
    
    # Log all incoming data for debugging
    logger.info(f"📨 Webhook called - Method: {request.method}")
    logger.info(f"📨 Form data: {dict(request.form)}")
    logger.info(f"📨 Args: {dict(request.args)}")
    
    # Validate request is from Twilio (skip in development or if GET request)
    if request.method == 'POST' and Config.FLASK_ENV == 'production':
        if not validate_twilio_request():
            logger.warning("⚠️ Invalid Twilio signature")
            return "Unauthorized", 401
    
    # Get message details
    if request.method == 'POST':
        incoming_msg = request.values.get('Body', '').strip()
        sender = request.values.get('From', '')
    else:
        # Handle GET requests (for testing)
        incoming_msg = request.args.get('Body', '').strip()
        sender = request.args.get('From', 'whatsapp:+918123456789')
    
    message_sid = request.values.get('MessageSid', 'unknown')
    
    logger.info(f"📨 Message from {sender}: '{incoming_msg}'")
    
    # Create TwiML response
    resp = MessagingResponse()
    msg = resp.message()
    
    # Check if this is a trigger message
    if is_trigger_message(incoming_msg):
        logger.info(f"🎯 Trigger detected from {sender}")
        
        # Send acknowledgment
        msg.body("🌱 *Processing your request...*\n\nFetching real-time sensor data and generating your crop prediction report. This will take a few seconds.")
        
        # Process in background thread
        thread = threading.Thread(target=process_prediction_background, args=(sender,))
        thread.daemon = True
        thread.start()
        logger.info(f"🚀 Background thread started for {sender}")
        
    else:
        # Help message
        triggers_display = ', '.join([f'"{t}"' for t in Config.TRIGGER_KEYWORDS.split(',')])
        msg.body(f"👋 *Welcome to AgriSetu!*\n\nSend {triggers_display} to get an AI-powered crop recommendation report.\n\n📊 You'll receive a detailed PDF with:\n• Real-time sensor data\n• Crop prediction\n• Growth duration\n• Water level status")
    
    logger.info(f"📤 Returning TwiML response: {str(resp)}")
    return str(resp)

@app.route('/reports/<filename>')
def serve_report(filename):
    """Serve generated PDF reports"""
    try:
        # Security: Only serve PDF files
        if not filename.endswith('.pdf'):
            return "Invalid file type", 403
        
        file_path = Config.REPORTS_DIR / filename
        
        if not file_path.exists():
            return "Report not found", 404
        
        return send_file(
            file_path,
            mimetype='application/pdf',
            as_attachment=False,
            download_name=filename
        )
        
    except Exception as e:
        logger.error(f"Error serving report: {e}")
        return "Error serving file", 500

@app.errorhandler(404)
def not_found(e):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def server_error(e):
    return jsonify({'error': 'Internal server error'}), 500

# ================== MAIN ==================
if __name__ == '__main__':
    logger.info("🚀 Starting AgriSetu WhatsApp Bot...")
    logger.info(f"📁 Reports directory: {Config.REPORTS_DIR}")
    logger.info(f"📁 Logs directory: {Config.LOGS_DIR}")
    logger.info(f"🔑 Trigger keywords: {Config.TRIGGER_KEYWORDS}")
    
    app.run(
        host='0.0.0.0',
        port=Config.PORT,
        debug=(Config.FLASK_ENV == 'development')
    )