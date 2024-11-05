import time
import torch
import torch.nn as nn
import pickle
from transformers import BertTokenizer, BertModel
from collections import Counter
import nltk
import logging
import torch.nn.functional as F

nltk.download('punkt')  # Download the sentence tokenizer
logger = logging.getLogger(__name__)

class BiLSTMClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, tactic_output_dim, technique_output_dim):
        super(BiLSTMClassifier, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc_tactic = nn.Linear(hidden_dim * 2, tactic_output_dim)
        self.fc_technique = nn.Linear(hidden_dim * 2, technique_output_dim)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        lstm_out = lstm_out[:, -1, :]  # Take the last hidden state
        tactic_logits = self.fc_tactic(lstm_out)
        technique_logits = self.fc_technique(lstm_out)
        return tactic_logits, technique_logits

def load_model(path='bilstm_model_full.pth'):
    model = torch.load(path)
    model.eval()  # Set to evaluation mode
    logger.info(f"Model loaded from {path}")
    return model

def load_label_encoders(path='label_encoders.pkl'):
    with open(path, 'rb') as f:
        encoders = pickle.load(f)
    tactic_encoder = encoders['tactic']
    technique_encoder = encoders['technique']
    logger.info(f"Label encoders loaded from {path}")
    return tactic_encoder, technique_encoder

def predict_custom_text(bilstm_model, bert_model, tokenizer, tactic_encoder, technique_encoder, texts, device='cpu'):
    logger.info(f"Predicting for {len(texts)} sentences...")

    # Tokenize the input texts
    inputs = tokenizer(texts, return_tensors='pt', padding=True, truncation=True, max_length=512).to(device)
    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']

    # Get BERT embeddings
    with torch.no_grad():
        bert_outputs = bert_model(input_ids=input_ids, attention_mask=attention_mask)
    bert_embeddings = bert_outputs.last_hidden_state

    # Pass the embeddings to the BiLSTM model for predictions
    bilstm_model = bilstm_model.to(device)
    tactic_logits, technique_logits = bilstm_model(bert_embeddings)

    # Get the predicted classes and confidence scores
    tactic_probs = F.softmax(tactic_logits, dim=1)
    technique_probs = F.softmax(technique_logits, dim=1)

    # Detach tensors and convert to NumPy
    predicted_tactics = torch.argmax(tactic_probs, dim=1).detach().cpu().numpy()
    predicted_techniques = torch.argmax(technique_probs, dim=1).detach().cpu().numpy()

    tactic_confidences = tactic_probs.max(dim=1).values.detach().cpu().numpy() * 100
    technique_confidences = technique_probs.max(dim=1).values.detach().cpu().numpy() * 100

    # Decode the predicted labels
    predicted_tactic_labels = tactic_encoder.inverse_transform(predicted_tactics)
    predicted_technique_labels = technique_encoder.inverse_transform(predicted_techniques)

    results = []
    for i in range(len(texts)):
        results.append((
            texts[i],
            predicted_tactic_labels[i],
            predicted_technique_labels[i],
            float(tactic_confidences[i]),  # Convert to standard float
            float(technique_confidences[i])  # Convert to standard float
        ))

    logger.info("Predictions generated.")
    return results

def analyze_predictions(predictions):
    tactics = Counter()
    techniques = Counter()
    tactic_technique_pairs = Counter()
    tactic_confidences = {}
    technique_confidences = {}

    for _, tactic, technique, tactic_conf, technique_conf in predictions:
        tactics[tactic] += 1
        techniques[technique] += 1
        tactic_technique_pairs[(tactic, technique)] += 1
        
        # Update confidence scores (keep the highest)
        tactic_confidences[tactic] = max(tactic_confidences.get(tactic, 0), float(tactic_conf))  # Convert to float
        technique_confidences[technique] = max(technique_confidences.get(technique, 0), float(technique_conf))  # Convert to float

    most_common_tactic = tactics.most_common(1)[0][0] if tactics else 'None'
    most_common_technique = techniques.most_common(1)[0][0] if techniques else 'None'
    most_common_pair = tactic_technique_pairs.most_common(1)[0][0] if tactic_technique_pairs else ('None', 'None')

    analysis = {
        'most_common_tactic': most_common_tactic,
        'most_common_technique': most_common_technique,
        'most_common_pair': most_common_pair,
        'tactic_distribution': dict(tactics),
        'technique_distribution': dict(techniques),
        'tactic_technique_pairs': dict(tactic_technique_pairs),
        'tactic_confidences': {tactic: float(conf) for tactic, conf in tactic_confidences.items()},  # Convert to float
        'technique_confidences': {technique: float(conf) for technique, conf in technique_confidences.items()},  # Convert to float
    }

    logger.info("Analysis complete.")
    return analysis

import re

import re

import re

def predict_from_paragraph(bilstm_model, bert_model, paragraph, tokenizer, tactic_encoder, technique_encoder, device='cpu'):
    def is_likely_list(text):
        lines = text.split('\n')
        if len(lines) < 2:
            return False
        
        # Check for repeated patterns or similar line structures
        non_empty_lines = [line.strip() for line in lines if line.strip()]
        if len(non_empty_lines) < 2:
            return False

        # Check for capitalization at the start of each line
        capitalized_lines = sum(1 for line in non_empty_lines if line[0].isupper())
        capitalization_ratio = capitalized_lines / len(non_empty_lines)
        
        # Check for absence of punctuation at the end of lines
        no_punctuation_lines = sum(1 for line in non_empty_lines if not line[-1] in '.!?')
        no_punctuation_ratio = no_punctuation_lines / len(non_empty_lines)
        
        return capitalization_ratio > 0.8 and no_punctuation_ratio > 0.7

    if isinstance(paragraph, str):
        if is_likely_list(paragraph):
            # Split the string into list items
            items = [item.strip() for item in paragraph.split('\n') if item.strip()]
            logger.info(f"Detected list input with {len(items)} items")
        else:
            # If not a list, tokenize into sentences as before
            items = nltk.sent_tokenize(paragraph)
            logger.info(f"Paragraph split into {len(items)} sentences")
    elif isinstance(paragraph, list):
        items = paragraph
        logger.info(f"Received list input with {len(items)} items")
    else:
        raise ValueError("Input must be a string or a list")

    start_time = time.time()
    all_predictions = []

    # Process all items, regardless of the number
    batch_size = 8
    for i in range(0, len(items), batch_size):
        batch_items = items[i:i + batch_size]
        predictions = predict_custom_text(bilstm_model, bert_model, tokenizer, tactic_encoder, technique_encoder, batch_items, device)
        all_predictions.extend(predictions)

    elapsed_time = time.time() - start_time
    logger.info(f"Prediction completed in {elapsed_time:.2f} seconds")
    
    analysis = analyze_predictions(all_predictions)
    return all_predictions, analysis