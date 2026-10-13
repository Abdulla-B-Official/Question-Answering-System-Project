# Extractive Question-Answering System 

## Overview

The Extractive Question-Answering System is an AI-powered natural language processing application designed to extract precise answer spans from text passages given a specific question.

Instead of relying on slow or unverified answers, the system uses a fine-tuned `deepset/roberta-base-squad2` transformer model to locate start and end token indices, generate accurate extractive text answers, and compute live confidence probabilities using PyTorch softmax logit normalization.

The project is structured with a clean, modular architecture separating backend processing utilities from frontend assets, deployed locally via a lightweight Flask web application.

## Key Features

* **Modular Inference Engine (`backend/inference.py`)**: Manages local transformer model weights, input tokenization, and forward-pass inference execution.
* **Softmax Confidence Scoring**: Computes reliable percentage-based confidence scores by applying softmax probabilities across start and end token logits.
* **Evaluation Framework (`backend/evaluator.py`)**: Calculates quantitative performance metrics including Exact Match (EM) and Token-F1 against dataset splits.
* **Preprocessing Pipeline (`backend/preprocessing.py`)**: Cleans, sanitizes, and normalizes input text passages and questions.
* **Interactive SQuAD Sampling**: Automatically pulls random passages and up to 5 validation questions directly from benchmark datasets.
* **Responsive Web Interface (`frontend/`)**: Modern dark-themed user interface featuring dropdown question selection, dynamic inputs, and real-time confidence score badges.

## How It Works

```
               INPUT (Context & Query)
                        │
                        ▼
               ┌─────────────────┐
               │    Flask API    │
               └────────┬────────┘
                        │
                        ▼
               ┌─────────────────┐
               │  RoBERTa Model  │
               │  (safetensors)  │
               └────────┬────────┘
                        │
                        ▼
               ┌─────────────────┐
               │ Token Span Logits│
               └────────┬────────┘
                        │
               ┌────────┴────────┐
               ▼                 ▼
          Start Logit       End Logit
               │                 │
               └────────┬────────┘
                        │
                        ▼
               ┌─────────────────┐
               │ Softmax Softmax │
               │ Confidence Calc │
               └────────┬────────┘
                        │
                        ▼
               ┌─────────────────┐
               │ Extracted Answer│
               └─────────────────┘

```

## Methodology & Pipeline

The project follows a structured engineering workflow from prototype development to modular deployment:

```
01. Model Selection & Fine-Tuning
        ↓
02. Modular Backend Architecture Design
        ↓
03. Text Preprocessing & Normalization
        ↓
04. Softmax Logit Confidence Estimation
        ↓
05. Quantitative Evaluation (EM / F1)
        ↓
06. Flask Web API Integration
        ↓
07. Interactive Frontend UI Design
        ↓
08. GitHub Version Control & Deployment

```

## Technology Stack

| **Technology** | **Purpose** |
| --- | --- |
| Python | Core Programming Language |
| PyTorch | Deep Learning Backend Framework |
| Hugging Face Transformers | Tokenizers & QA Models (RoBERTa) |
| Datasets | Benchmark SQuAD Data Loading |
| NumPy & Pandas | Data Handling & Metrics Calculation |
| Flask | Backend Web API Framework |
| HTML5 / CSS3 / JavaScript | Frontend User Interface & Asynchronous API Integration |
| Git / GitHub | Version Control & Repository Management |

## Project Structure

```
Question_Answering_System/
│
├── app.py                      # Main entry point for Flask web server
├── requirements.txt            # Project dependencies
├── .gitignore                  # Git ignore rules
│
├── backend/
│   ├── __init__.py
│   ├── evaluator.py            # SQuAD Exact Match & F1 metrics
│   ├── inference.py            # Model loading & softmax confidence scoring
│   └── preprocessing.py        # Text sanitization and normalization
│
├── frontend/
│   ├── index.html              # Web interface layout
│   ├── style.css               # Modern dark-themed styling
│   └── script.js               # Frontend API integration & UI handlers
│
└── saved_qa_model/             # Fine-tuned model directory
    ├── config.json
    ├── model.safetensors
    ├── tokenizer.json
    └── tokenizer_config.json

```

## Installation & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/Abdulla-B-Official/Question-Answering-System-Project.git
cd Question-Answering-System-Project

```

### 2. Create a Virtual Environment

* Windows (Command Prompt):
```bash
python -m venv venv
venv\Scripts\activate.bat

```


* Windows (PowerShell):
```bash
python -m venv venv
.\venv\Scripts\Activate.ps1

```


* Linux / macOS:
```bash
python3 -m venv venv
source venv/bin/activate

```



### 3. Install Dependencies

```bash
pip install -r requirements.txt

```

*(Note: Ensure NumPy is pinned to `<2` for PyTorch 2.2.0 compatibility).*

## Running the Application

### 1. Launch the Flask Web Server

From your project root directory, run:

```bash
python app.py

```

Open your browser and navigate to:
👉 `[http://127.0.0.1:5000/](http://127.0.0.1:5000/)`

### 2. Run Evaluation Script

To check local evaluation metrics against validation splits:

```bash
python -m backend.evaluator

```

## Future Enhancements

* **Top-K Alternative Answers**: Extend inference to return alternative answer spans ranked by confidence score.
* **Model Quantization**: Optimize inference latency using ONNX Runtime or 4-bit quantization.
* **Cloud Hosting**: Deploy the containerized Flask application to Render or Hugging Face Spaces.
* **Expanded Evaluation**: Integrate broader QA benchmark datasets (e.g., Natural Questions, HotpotQA).

## Project Repository

[Question-Answering System Project Repository](https://github.com/Abdulla-B-Official/Question-Answering-System-Project)

Built with: Python · PyTorch · Hugging Face Transformers · Flask · HTML/CSS/JavaScript · Git

Extract Smarter. Answer Accurately. Automate Comprehension.
