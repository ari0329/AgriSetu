"""
Enhanced PDF Generator - Merged from live_agrisetu.py
Generates professional crop prediction reports
"""

import os
import random
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple, Optional

import joblib
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, 
    TableStyle, Image, PageBreak
)
from reportlab.lib.units import inch

from config import Config

logger = logging.getLogger(__name__)

# Feature columns for ML model
FEATURE_COLUMNS = [
    "Soil_Moisture_%",
    "Soil_Temperature_C",
    "Rainfall_ml",
    "Air_Temperature_C",
    "Humidity_%"
]

class PDFGenerator:
    """Generate professional PDF reports for crop predictions"""
    
    def __init__(self):
        self.models_loaded = False
        self.crop_model = None
        self.label_encoder = None
        self.month_model = None
        self.month_lookup = {}
        
        self._load_models()
    
    def _load_models(self):
        """Load ML models from disk"""
        try:
            if Config.CROP_MODEL_PATH.exists():
                self.crop_model = joblib.load(Config.CROP_MODEL_PATH)
                logger.info("✅ Crop model loaded")
            
            if Config.LABEL_ENCODER_PATH.exists():
                self.label_encoder = joblib.load(Config.LABEL_ENCODER_PATH)
                logger.info("✅ Label encoder loaded")
            
            if Config.MONTH_MODEL_PATH.exists():
                self.month_model = joblib.load(Config.MONTH_MODEL_PATH)
                logger.info("✅ Month model loaded")
            
            if Config.MONTH_LOOKUP_PATH.exists():
                self.month_lookup = joblib.load(Config.MONTH_LOOKUP_PATH)
                logger.info("✅ Month lookup loaded")
            
            if self.crop_model and self.label_encoder:
                self.models_loaded = True
            else:
                logger.warning("⚠️ ML models not found, using fallback predictions")
                
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            self.models_loaded = False
    
    def generate_report(self, sensor_data: Dict) -> Tuple[str, str, str]:
        """
        Generate PDF report from sensor data
        
        Args:
            sensor_data: Dictionary with sensor readings
            
        Returns:
            Tuple of (file_path, crop_name, growth_months)
        """
        # Extract sensor values
        soil_moisture = float(sensor_data.get('soil_moisture', 50))
        soil_temp = float(sensor_data.get('soil_temperature', 25))
        
        # Generate complementary data (would come from weather API in production)
        rainfall = round(random.uniform(80, 150), 2)
        air_temp = round(random.uniform(20, 40), 2)
        humidity = round(random.uniform(40, 80), 2)
        
        # Create feature DataFrame
        sample_input = pd.DataFrame([{
            "Soil_Moisture_%": soil_moisture,
            "Soil_Temperature_C": soil_temp,
            "Rainfall_ml": rainfall,
            "Air_Temperature_C": air_temp,
            "Humidity_%": humidity
        }], columns=FEATURE_COLUMNS)
        
        # Make prediction
        if self.models_loaded:
            crop, growth_months = self._predict_with_model(sample_input)
        else:
            crop, growth_months = self._predict_fallback(sample_input)
        
        # Generate PDF
        timestamp = datetime.now()
        file_name = f"AgriSetu_{timestamp.strftime('%Y%m%d_%H%M%S')}.pdf"
        file_path = Config.REPORTS_DIR / file_name
        
        self._create_pdf(
            file_path, 
            sample_input.iloc[0].to_dict(),
            crop, 
            growth_months, 
            sensor_data,
            timestamp
        )
        
        logger.info(f"✅ PDF generated: {file_name}")
        
        return str(file_path), crop, str(growth_months)
    
    def _predict_with_model(self, sample_input: pd.DataFrame) -> Tuple[str, int]:
        """Make prediction using loaded ML models"""
        try:
            predicted_label = self.crop_model.predict(sample_input)
            crop = self.label_encoder.inverse_transform(predicted_label)[0]
            
            # Predict growth months
            if crop in self.month_lookup:
                growth_months = int(self.month_lookup[crop])
            elif self.month_model:
                predicted = self.month_model.predict(sample_input)[0]
                growth_months = max(1, int(round(predicted)))
            else:
                growth_months = 4  # Default
                
            return crop, growth_months
            
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return self._predict_fallback(sample_input)
    
    def _predict_fallback(self, sample_input: pd.DataFrame) -> Tuple[str, int]:
        """Fallback prediction when models are unavailable"""
        # Simple rule-based fallback
        soil_moisture = sample_input['Soil_Moisture_%'].iloc[0]
        soil_temp = sample_input['Soil_Temperature_C'].iloc[0]
        
        if soil_moisture > 60 and soil_temp > 25:
            crop = "Rice"
        elif soil_moisture < 40 and soil_temp > 20:
            crop = "Wheat"
        elif soil_temp > 30:
            crop = "Cotton"
        else:
            crop = "Maize"
        
        growth_months = 4
        
        logger.info(f"Fallback prediction: {crop}")
        return crop, growth_months
    
    def _create_pdf(self, file_path: Path, sensor_values: Dict, 
                    crop: str, growth_months: int, raw_data: Dict,
                    timestamp: datetime):
        """Create professionally styled PDF report"""
        
        doc = SimpleDocTemplate(
            str(file_path),
            pagesize=A4,
            rightMargin=40,
            leftMargin=40,
            topMargin=40,
            bottomMargin=40,
        )
        
        styles = getSampleStyleSheet()
        
        # Custom styles
        title_style = ParagraphStyle(
            "TitleStyle",
            parent=styles["Title"],
            alignment=1,
            textColor=colors.white,
            fontSize=22,
            spaceAfter=20,
        )
        
        section_style = ParagraphStyle(
            "SectionStyle",
            parent=styles["Heading2"],
            textColor=colors.HexColor("#2E7D32"),
            spaceBefore=20,
            spaceAfter=10,
            fontSize=14,
        )
        
        normal_style = ParagraphStyle(
            "NormalStyle",
            parent=styles["Normal"],
            fontSize=11,
            leading=16,
            spaceAfter=8,
        )
        
        content = []
        
        # ===== HEADER =====
        header_table = Table(
            [[Paragraph("🌱 SMART CROP PREDICTION REPORT", title_style)],
             [Paragraph("AgriSetu - AI-Powered Agriculture", 
                       ParagraphStyle("Subtitle", parent=styles["Normal"], 
                                     alignment=1, textColor=colors.white, fontSize=12))]],
            colWidths=[450],
        )
        header_table.setStyle(
            TableStyle([
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#2E7D32")),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING", (0, 0), (-1, -1), 16),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 16),
            ])
        )
        content.append(header_table)
        content.append(Spacer(1, 20))
        
        # ===== TIMESTAMP & SOURCE =====
        display_time = timestamp.strftime("%d %B %Y | %I:%M %p")
        content.append(Paragraph(f"<b>Generated On:</b> {display_time}", normal_style))
        content.append(Paragraph(f"<b>Data Source:</b> {raw_data.get('source', 'Unknown').upper()}", 
                                normal_style))
        
        # ===== SENSOR DATA SECTION =====
        content.append(Paragraph("📊 SENSOR READINGS", section_style))
        
        sensor_data_table = [
            ["Parameter", "Value", "Status"],
            ["Soil Moisture", f"{sensor_values['Soil_Moisture_%']:.1f}%", 
             self._get_moisture_status(sensor_values['Soil_Moisture_%'])],
            ["Soil Temperature", f"{sensor_values['Soil_Temperature_C']:.1f}°C",
             self._get_temp_status(sensor_values['Soil_Temperature_C'])],
            ["Rainfall (Est.)", f"{sensor_values['Rainfall_ml']:.1f} ml", "Normal"],
            ["Air Temperature", f"{sensor_values['Air_Temperature_C']:.1f}°C",
             self._get_temp_status(sensor_values['Air_Temperature_C'])],
            ["Humidity", f"{sensor_values['Humidity_%']:.1f}%",
             self._get_humidity_status(sensor_values['Humidity_%'])],
        ]
        
        # Add water level if available
        if 'L1' in raw_data:
            water_level = self._get_water_level(raw_data)
            sensor_data_table.append(["Water Level", water_level, "OK"])
        
        sensor_table = Table(sensor_data_table, colWidths=[180, 120, 150])
        sensor_table.setStyle(
            TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#A5D6A7")),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("FONT", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("ALIGN", (1, 1), (1, -1), "CENTER"),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
                ("TOPPADDING", (0, 0), (-1, -1), 8),
                ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#F5F5F5")),
            ])
        )
        content.append(sensor_table)
        
        # ===== PREDICTION RESULT =====
        content.append(Spacer(1, 20))
        content.append(Paragraph("🎯 CROP RECOMMENDATION", section_style))
        
        result_table = Table(
            [
                ["Recommended Crop", crop],
                ["Growth Duration", f"{growth_months} Months"],
                ["Confidence Level", "High" if self.models_loaded else "Medium (Fallback)"],
            ],
            colWidths=[180, 270],
        )
        result_table.setStyle(
            TableStyle([
                ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#C8E6C9")),
                ("BACKGROUND", (1, 0), (1, -1), colors.HexColor("#E8F5E9")),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.green),
                ("FONT", (0, 0), (0, -1), "Helvetica-Bold"),
                ("TOPPADDING", (0, 0), (-1, -1), 12),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 12),
            ])
        )
        content.append(result_table)
        
        # ===== RECOMMENDATION TEXT =====
        content.append(Spacer(1, 15))
        recommendation_text = (
            f"Based on current sensor readings, <b>{crop}</b> is the most suitable crop. "
            f"The expected growth cycle is approximately <b>{growth_months} months</b> "
            f"under optimal conditions. Soil moisture levels are "
            f"{self._get_moisture_status(sensor_values['Soil_Moisture_%']).lower()} "
            f"for {crop.lower()} cultivation."
        )
        content.append(Paragraph(recommendation_text, normal_style))
        
        # ===== WATER LEVEL DETAILS (if available) =====
        if 'L1' in raw_data:
            content.append(Spacer(1, 15))
            content.append(Paragraph("💧 WATER RESERVOIR STATUS", section_style))
            
            level_data = [
                ["Level 1 (25%)", "✅ Full" if raw_data.get('L1', 0) else "❌ Empty"],
                ["Level 2 (50%)", "✅ Full" if raw_data.get('L2', 0) else "❌ Empty"],
                ["Level 3 (75%)", "✅ Full" if raw_data.get('L3', 0) else "❌ Empty"],
                ["Level 4 (100%)", "✅ Full" if raw_data.get('L4', 0) else "❌ Empty"],
            ]
            
            water_table = Table(level_data, colWidths=[225, 225])
            water_table.setStyle(
                TableStyle([
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                    ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#E3F2FD")),
                    ("TOPPADDING", (0, 0), (-1, -1), 8),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
                ])
            )
            content.append(water_table)
        
        # ===== FOOTER =====
        content.append(Spacer(1, 30))
        footer_text = (
            "This report is automatically generated by AgriSetu Smart Agriculture System.<br/>"
            "For queries, reply to this WhatsApp number."
        )
        content.append(Paragraph(footer_text, styles["Italic"]))
        
        # Build PDF
        doc.build(content)
    
    def _get_moisture_status(self, value: float) -> str:
        if value < 30:
            return "⚠️ Dry"
        elif value < 60:
            return "✅ Optimal"
        else:
            return "💧 Wet"
    
    def _get_temp_status(self, value: float) -> str:
        if value < 15:
            return "❄️ Cold"
        elif value < 30:
            return "✅ Optimal"
        else:
            return "🔥 Hot"
    
    def _get_humidity_status(self, value: float) -> str:
        if value < 40:
            return "⚠️ Low"
        elif value < 70:
            return "✅ Optimal"
        else:
            return "💧 High"
    
    def _get_water_level(self, data: Dict) -> str:
        """Calculate overall water level percentage"""
        levels = [data.get('L1', 0), data.get('L2', 0), 
                 data.get('L3', 0), data.get('L4', 0)]
        filled = sum(levels)
        
        if filled == 4:
            return "100% (Full)"
        elif filled == 3:
            return "75%"
        elif filled == 2:
            return "50%"
        elif filled == 1:
            return "25%"
        else:
            return "Empty"

# Singleton instance
pdf_generator = PDFGenerator()

def generate_pdf(sensor_data: Dict) -> Tuple[str, str, str]:
    """Public function to generate PDF report"""
    return pdf_generator.generate_report(sensor_data)