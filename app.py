"""
Module: app.py
Description: Root Flask server serving routes, JSON APIs, and dataset sampling.
"""

import random
from flask import Flask, render_template, request, jsonify
from datasets import load_dataset
from backend.inference import QAInferenceEngine

app = Flask(__name__, template_folder="frontend", static_folder="frontend", static_url_path="")

qa_engine = QAInferenceEngine(model_dir="./saved_qa_model")

# Cache a small slice of validation records in memory for instant random sampling
print("Loading validation sample pool for auto-generation...")
val_pool = load_dataset("rajpurkar/squad", split="validation[:50]")

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/random-samples", methods=["GET"])
def get_random_samples():
    """
    Randomly selects a context passage and up to 5 questions linked to it 
    from the dataset pool, returning them to the frontend.
    """
    try:
        # Pick a random sample from the pool
        sample = random.choice(val_pool)
        context = sample["context"]
        
        # Find all questions belonging to this exact context passage in the pool
        matching_rows = [row for row in val_pool if row["context"] == context]
        
        # Extract up to 5 unique questions
        questions = []
        for row in matching_rows:
            if row["question"] not in questions:
                questions.append(row["question"])
            if len(questions) >= 5:
                break
                
        return jsonify({
            "success": True,
            "context": context,
            "questions": questions
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400

@app.route("/api/predict", methods=["POST"])
def predict():
    """API endpoint returning answer and confidence score."""
    try:
        data = request.get_json()
        context = data.get("context", "")
        question = data.get("question", "")

        result = qa_engine.predict_with_confidence(context, question)
        return jsonify({
            "success": True, 
            "answer": result["answer"], 
            "confidence": result["confidence"]
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True, port=5000)