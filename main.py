import os
from flask import Flask, render_template, request, send_file, jsonify, flash
from werkzeug.utils import secure_filename
import torch
from transformers import BertTokenizer, BertModel
import nltk
from nltk import sent_tokenize
import pickle
from docx import Document
from PyPDF2 import PdfReader
import json
import logging
from logging.handlers import RotatingFileHandler
from markupsafe import escape

# Import your model and prediction functions here
from model import BiLSTMClassifier, load_model, load_label_encoders, predict_from_paragraph
from mitre_info import enrich_analysis

# Set up logging
log_directory = 'logs'
os.makedirs(log_directory, exist_ok=True)
log_file = os.path.join(log_directory, 'app.log')

logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Create a file handler
file_handler = RotatingFileHandler(log_file, maxBytes=10485760, backupCount=5)  # 10MB per file, keep 5 old versions
file_handler.setLevel(logging.INFO)
file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(file_formatter)

# Add the file handler to the logger
logger.addHandler(file_handler)

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'  # Set a secret key for flash messages

# Configure upload folder
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'docx', 'json'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB max file size

# Ensure the upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load your model and encoders
try:
    bilstm_model = load_model('models/bilstm_model_full.pth')
    tactic_encoder, technique_encoder = load_label_encoders('models/label_encoders.pkl')
    tokenizer = BertTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')
    bert_model = BertModel.from_pretrained('allenai/scibert_scivocab_uncased')
    logger.info("Models and encoders loaded successfully")
except Exception as e:
    logger.error(f"Error loading models and encoders: {str(e)}")
    raise

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_text(file_path):
    _, extension = os.path.splitext(file_path)
    try:
        if extension == '.txt':
            with open(file_path, 'r') as file:
                return file.read()
        elif extension == '.docx':
            doc = Document(file_path)
            return ' '.join([paragraph.text for paragraph in doc.paragraphs])
        elif extension == '.pdf':
            reader = PdfReader(file_path)
            return ' '.join([page.extract_text() for page in reader.pages])
        elif extension == '.json':
            with open(file_path, 'r') as file:
                data = json.load(file)
                if isinstance(data, dict):
                    return data.get('text', '')
                elif isinstance(data, list):
                    return ' '.join([item.get('text', '') for item in data if isinstance(item, dict)])
                else:
                    logger.warning(f"Unexpected JSON structure in file: {file_path}")
                    return ''
    except json.JSONDecodeError as e:
        logger.error(f"JSON Decode Error in file {file_path}: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Error extracting text from file: {str(e)}")
        raise

def validate_analysis_object(analysis):
    """Ensures all required fields exist in the analysis object"""
    required_fields = {
        'tactic_distribution': {},
        'technique_distribution': {},
        'tactic_confidences': {},
        'technique_confidences': {},
        'most_common_tactic': 'Unknown',
        'most_common_technique': 'Unknown',
        'most_common_pair': ('Unknown', 'Unknown')
    }
    
    if not isinstance(analysis, dict):
        analysis = {}
    
    for field, default_value in required_fields.items():
        if field not in analysis or not analysis[field]:
            analysis[field] = default_value
    
    return analysis

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        try:
            # Check if file exists in request
            if 'file' not in request.files:
                flash("No file part in the request", "error")
                return render_template('upload.html')
            
            file = request.files['file']
            if file.filename == '':
                flash("No selected file", "error")
                return render_template('upload.html')
            
            # Validate file type
            if not allowed_file(file.filename):
                flash("File type not allowed. Please upload a .txt, .pdf, .docx, or .json file", "error")
                return render_template('upload.html')
            
            # Save and process file
            try:
                filename = secure_filename(file.filename)
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(file_path)
                logger.info(f"File saved: {file_path}")
                
                # Extract text
                text = extract_text(file_path)
                if not text or len(text.strip()) == 0:
                    raise ValueError("No text could be extracted from the file")
                logger.info(f"Text extracted from file, length: {len(text)}")
                
                # Generate predictions
                predictions, analysis = predict_from_paragraph(bilstm_model, bert_model, text, tokenizer, tactic_encoder, technique_encoder)
                
                if not isinstance(predictions, list):
                    predictions = [predictions]  # Convert to list if single prediction
                
                # Restructure predictions for template
                formatted_predictions = []
                for pred in predictions:
                    if isinstance(pred, dict):
                        pred_text = pred.get('text', '')
                        pred_tactic = pred.get('tactic', '')
                        pred_technique = pred.get('technique', '')
                    elif isinstance(pred, tuple):
                        pred_text = pred[0] if len(pred) > 0 else ''
                        pred_tactic = pred[1] if len(pred) > 1 else ''
                        pred_technique = pred[2] if len(pred) > 2 else ''
                    else:
                        continue  # Skip invalid predictions
                        
                    formatted_predictions.append({
                        'text': pred_text,
                        'tactic': pred_tactic,
                        'technique': pred_technique,
                        'tactic_confidence': analysis.get('tactic_confidences', {}).get(pred_tactic, 0),
                        'technique_confidence': analysis.get('technique_confidences', {}).get(pred_technique, 0)
                    })
                
                # Ensure all required analysis components exist
                analysis = validate_analysis_object(analysis)
                analysis['predictions'] = formatted_predictions
                
                # Get enriched analysis with MITRE ATT&CK information
                enriched_analysis = enrich_analysis(analysis)
                
                # Escape descriptions in enriched_analysis
                for item, data in enriched_analysis.items():
                    if 'info' in data and 'description' in data['info']:
                        data['info']['description'] = escape(data['info']['description'])
                
                logger.info(f"Predictions generated: {len(formatted_predictions)} items")
                
                return render_template('report.html', 
                                    analysis=analysis,
                                    enriched_analysis=enriched_analysis,
                                    tactic_distribution=json.dumps(analysis['tactic_distribution']),
                                    technique_distribution=json.dumps(analysis['technique_distribution']),
                                    filename=filename)
                
            except Exception as e:
                logger.error(f"File processing error: {str(e)}")
                flash(f"Error processing file: {str(e)}", "error")
                return render_template('upload.html')
            
            finally:
                # Clean up uploaded file
                try:
                    if os.path.exists(file_path):
                        os.remove(file_path)
                except Exception as e:
                    logger.error(f"Error removing temporary file: {str(e)}")
                    
        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}")
            flash("An unexpected error occurred. Please try again.", "error")
            return render_template('upload.html')
    
    return render_template('upload.html')

@app.errorhandler(413)
def too_large(e):
    flash("File is too large. Maximum size is 16 MB.", "error")
    return render_template('upload.html'), 413

if __name__ == '__main__':
    app.run(debug=True)