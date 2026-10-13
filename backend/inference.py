"""
Module: inference.py
Description: Manages model loading, softmax confidence score calculation, 
and multi-answer extractive inference.
"""

import os
import torch
import torch.nn.functional as F
from transformers import AutoModelForQuestionAnswering, AutoTokenizer
from backend.preprocessing import clean_input_text

class QAInferenceEngine:
    def __init__(self, model_dir="./saved_qa_model"):
        print("Initializing QA Inference Engine...")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        if not os.path.exists(model_dir):
            raise FileNotFoundError(f"Model directory not found at: {model_dir}")
            
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        except Exception:
            self.tokenizer = AutoTokenizer.from_pretrained("deepset/roberta-base-squad2")

        self.model = AutoModelForQuestionAnswering.from_pretrained(model_dir)
        self.model.to(self.device)
        self.model.eval()
        print(f"QA Model loaded successfully on device: {self.device}")

    def predict_with_confidence(self, context: str, question: str) -> dict:
        """
        Extracts the answer span and computes a normalized confidence score (%) 
        based on softmax probabilities of the start and end token logits.
        """
        cleaned_context = clean_input_text(context)
        cleaned_question = clean_input_text(question)

        if not cleaned_context or not cleaned_question:
            raise ValueError("Context and question cannot be empty.")

        inputs = self.tokenizer(
            cleaned_question, 
            cleaned_context, 
            return_tensors="pt", 
            truncation=True, 
            max_length=384
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        # Apply softmax to calculate confidence probabilities
        start_probs = F.softmax(outputs.start_logits, dim=-1)
        end_probs = F.softmax(outputs.end_logits, dim=-1)

        start_idx = torch.argmax(outputs.start_logits).item()
        end_idx = torch.argmax(outputs.end_logits).item()

        # Multiply start and end token probabilities for combined span confidence
        confidence = (start_probs[0][start_idx].item() * end_probs[0][end_idx].item()) * 100

        answer = self.tokenizer.decode(inputs.input_ids[0, start_idx:end_idx + 1])

        if not answer.strip() or start_idx > end_idx:
            return {"answer": "I couldn't find a clear answer in the provided context.", "confidence": 0.0}

        return {
            "answer": answer.strip(),
            "confidence": round(confidence, 2)
        }