"""
Module: preprocessing.py
Description: Handles cleaning of raw user inputs and text normalization 
for quantitative evaluation metrics (Exact Match and Token F1).
"""

import re

def clean_input_text(text: str) -> str:
    """
    Cleans raw text inputs (context passages or questions) by removing 
    excessive whitespaces, tabs, and newline anomalies.
    
    Args:
        text (str): Raw string input from the user interface.
        
    Returns:
        str: Sanitized and trimmed single-line string.
    """
    if not text:
        return ""
    
    # Replace any sequence of whitespace characters with a single space and trim edges
    cleaned = re.sub(r'\s+', ' ', text).strip()
    return cleaned

def normalize_text(text: str) -> str:
    """
    Normalizes text for objective evaluation by converting all characters 
    to lowercase and stripping punctuation/symbols.
    
    Args:
        text (str): Ground truth or predicted answer string.
        
    Returns:
        str: Alphanumeric-only normalized text string.
    """
    if not text:
        return ""
    
    # Keep only alphanumeric characters and whitespace for fair string matching
    return "".join(ch for ch in text.lower() if ch.isalnum() or ch.isspace())