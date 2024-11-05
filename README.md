# ML-Based Threat Detection System

## Overview
This project implements a machine learning-based system for detecting and classifying potential security threats. It uses a combination of Natural Language Processing (NLP) techniques and a BiLSTM (Bidirectional Long Short-Term Memory) neural network to analyze text inputs and predict associated tactics and techniques, likely based on the MITRE ATT&CK framework.

## Features
- Text input processing (supports plain text, PDF, and DOCX files)
- ML-based threat detection and classification
- Web interface for easy interaction
- Batch processing capabilities
- Detailed analysis and reporting of detected threats

## Technical Stack
- Python 3.x
- PyTorch for deep learning
- Transformers library (BERT model)
- Flask for web interface
- NLTK for text processing

## Main Components

### 1. Model (`model.py`)
- Defines the BiLSTM classifier
- Implements prediction logic
- Handles text preprocessing and batch processing

### 2. Web Application (`app.py`)
- Implements Flask web server
- Manages file uploads and processing
- Renders results and analysis

### 3. MITRE ATT&CK Integration (`mitre_info.py`)
- Enriches analysis with MITRE ATT&CK framework information

## Setup and Installation

1. Clone the repository
2. Create a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```
3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
4. Download necessary NLTK data:
   ```python
   import nltk
   nltk.download('punkt')
   ```

## Usage

1. Start the Flask server:
   ```
   python app.py
   ```
2. Open a web browser and navigate to `http://localhost:5000`
3. Upload a text file, PDF, or DOCX document for analysis
4. View the analysis results, including predicted tactics and techniques

## Model Training (if applicable)

Details on how to train or fine-tune the model are not provided in the given code snippets. If this is a feature of your project, please provide more information.

## Logging

The application uses Python's logging module to log information and errors. Logs are stored in the `logs` directory.

## Future Improvements

- Implement user authentication for the web interface
- Add support for more file formats
- Improve model accuracy through further training or fine-tuning
- Implement real-time monitoring capabilities

## Contributing

Please read CONTRIBUTING.md for details on our code of conduct and the process for submitting pull requests.

## License

This project is licensed under the [] - see the LICENSE.md file for details.
