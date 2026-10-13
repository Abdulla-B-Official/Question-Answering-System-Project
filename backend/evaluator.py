"""
Module: evaluator.py
Description: Evaluates fine-tuned model performance across validation datasets 
computing Exact Match (EM), Precision, Recall, and Token-Level F1 scores.
"""

import torch
import collections
from datasets import load_dataset
from transformers import AutoModelForQuestionAnswering, AutoTokenizer
from backend.preprocessing import normalize_text

class QAEvaluator:
    def __init__(self, model_dir="./saved_qa_model"):
        """Initializes model weights and evaluation tokenizer locally."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForQuestionAnswering.from_pretrained(model_dir).to(self.device)
        self.model.eval()

    def _compute_metrics(self, prediction: str, ground_truth: str):
        """
        Calculates token-level Precision, Recall, and F1 overlap between 
        the predicted answer span and the ground truth reference text.
        """
        pred_toks = normalize_text(prediction).split()
        truth_toks = normalize_text(ground_truth).split()

        if len(pred_toks) == 0 or len(truth_toks) == 0:
            score = 1.0 if pred_toks == truth_toks else 0.0
            return score, score, score

        # Count token occurrences to find overlapping words
        common = collections.Counter(pred_toks) & collections.Counter(truth_toks)
        num_same = sum(common.values())

        if num_same == 0:
            return 0.0, 0.0, 0.0

        precision = num_same / len(pred_toks)
        recall = num_same / len(truth_toks)
        f1 = (2 * precision * recall) / (precision + recall)
        return precision, recall, f1

    def run_evaluation(self, sample_limit=100):
        """
        Runs evaluation loop over a specified subset of the validation dataset.
        
        Args:
            sample_limit (int): Number of evaluation records to process.
            
        Returns:
            dict: Average percentage scores for EM, Precision, Recall, and F1.
        """
        print(f"Loading validation dataset for evaluation (limit: {sample_limit} samples)...")
        dataset = load_dataset("rajpurkar/squad", split=f"validation[:{sample_limit}]")

        exact_matches = 0
        precisions = []
        recalls = []
        f1_scores = []
        total = len(dataset)

        for row in dataset:
            context = row["context"]
            question = row["question"]
            true_answers = row["answers"]["text"]

            # Tokenize sample and transfer to GPU/CPU device
            inputs = self.tokenizer(question, context, return_tensors="pt", truncation=True, max_length=384).to(self.device)
            with torch.no_grad():
                outputs = self.model(**inputs)

            start_idx = torch.argmax(outputs.start_logits).item()
            end_idx = torch.argmax(outputs.end_logits).item()
            pred_ans = self.tokenizer.decode(inputs.input_ids[0, start_idx:end_idx + 1])

            # Check if prediction matches any valid ground-truth answer exactly
            em = max(1 if normalize_text(pred_ans) == normalize_text(t) else 0 for t in true_answers)
            exact_matches += em

            # Compute highest precision, recall, and F1 against acceptable references
            best_p, best_r, best_f1 = 0.0, 0.0, 0.0
            for t in true_answers:
                p, r, f1 = self._compute_metrics(pred_ans, t)
                if f1 > best_f1:
                    best_p, best_r, best_f1 = p, r, f1

            precisions.append(best_p)
            recalls.append(best_r)
            f1_scores.append(best_f1)

        results = {
            "exact_match": (exact_matches / total) * 100,
            "precision": (sum(precisions) / total) * 100,
            "recall": (sum(recalls) / total) * 100,
            "f1_score": (sum(f1_scores) / total) * 100
        }
        return results

if __name__ == "__main__":
    evaluator = QAEvaluator()
    metrics = evaluator.run_evaluation(sample_limit=100)
    print("\n=== Local Evaluation Results ===")
    print(f"Exact Match (EM):  {metrics['exact_match']:.2f}%")
    print(f"Average Precision: {metrics['precision']:.2f}%")
    print(f"Average Recall:    {metrics['recall']:.2f}%")
    print(f"Overall F1 Score:  {metrics['f1_score']:.2f}%")