"""Comprehensive evaluation of the fine-tuned model"""

import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
from rouge_score import rouge_scorer
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict
import json
import numpy as np
from tqdm import tqdm
import os
from datetime import datetime
from src import config


class ModelEvaluator:
    def __init__(self, model_path: str = "data/models/final_model"):
        """Initialize with both fine-tuned and base models"""

        print("🔄 Loading models for evaluation...")

        # Load fine-tuned model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.finetuned_model = T5ForConditionalGeneration.from_pretrained(model_path).to(self.device)
        self.tokenizer = T5Tokenizer.from_pretrained(model_path)

        # Load base model for comparison
        self.base_model = T5ForConditionalGeneration.from_pretrained(config.MODEL_NAME).to(self.device)
        self.base_tokenizer = T5Tokenizer.from_pretrained(config.MODEL_NAME)

        # ROUGE scorer
        self.rouge_scorer = rouge_scorer.RougeScorer(
            ['rouge1', 'rouge2', 'rougeL'], use_stemmer=True
        )

        self.results = {
            'finetuned': {'rouge1': [], 'rouge2': [], 'rougeL': [], 'tech_preservation': []},
            'base': {'rouge1': [], 'rouge2': [], 'rougeL': [], 'tech_preservation': []}
        }

        self.bad_predictions = []

    def generate_summary(self, text: str, model, tokenizer, max_length: int = 150):
        """Generate summary using a model"""

        inputs = tokenizer(
            f"summarize: {text}",
            max_length=512,
            truncation=True,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=max_length,
                num_beams=4,
                early_stopping=True,
                no_repeat_ngram_size=3
            )

        return tokenizer.decode(outputs[0], skip_special_tokens=True)

    def extract_tech_terms(self, text: str) -> List[str]:
        """Extract technical terms from text"""
        # Simple implementation - looks for capitalized terms and known tech keywords
        tech_keywords = ['AI', 'API', 'GPU', 'CPU', 'ML', 'AWS', 'Azure', 'Google', 'Microsoft',
                        'Apple', 'Meta', 'OpenAI', 'startup', 'cloud', 'software', 'hardware',
                        'algorithm', 'neural', 'quantum', 'blockchain', 'cryptocurrency']

        found_terms = []
        text_lower = text.lower()

        for term in tech_keywords:
            if term.lower() in text_lower:
                found_terms.append(term)

        # Also find capitalized multi-word terms (like "Google Cloud")
        import re
        capitalized = re.findall(r'[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*', text)
        found_terms.extend([term for term in capitalized if len(term) > 3])

        return list(set(found_terms))

    def rouge_l_score(self, pred, ref):
        return self.rouge_scorer.score(ref, pred)['rougeL'].fmeasure

    def evaluate_dataset(self, test_data_path: str = "data/processed/test.json"):
        """Evaluate both models on test dataset"""

        # Load test data
        with open(test_data_path, 'r', encoding='utf-8') as f:
            test_data = json.load(f)

        print(f"📊 Evaluating on {len(test_data)} test examples...")

        for item in tqdm(test_data[:50], desc="Evaluating"):  # Limit to 50 for speed
            article = item['content']
            reference_summary = item['summary']

            # Generate summaries
            finetuned_summary = self.generate_summary(
                article, self.finetuned_model, self.tokenizer
            )
            base_summary = self.generate_summary(
                article, self.base_model, self.base_tokenizer
            )

            # Calculate ROUGE scores
            finetuned_scores = self.rouge_scorer.score(reference_summary, finetuned_summary)
            base_scores = self.rouge_scorer.score(reference_summary, base_summary)

            # Store results
            for metric in ['rouge1', 'rouge2', 'rougeL']:
                self.results['finetuned'][metric].append(
                    finetuned_scores[metric].fmeasure
                )
                self.results['base'][metric].append(
                    base_scores[metric].fmeasure
                )

            # Track bad predictions
            if self.rouge_l_score(finetuned_summary, reference_summary) < 0.3:
                self.bad_predictions.append({
                    "input": article[:200] + "...",
                    "reference": reference_summary,
                    "prediction": finetuned_summary
                })

            # Check technical term preservation
            tech_terms = self.extract_tech_terms(article)
            if tech_terms:
                finetuned_preserved = sum(1 for term in tech_terms if term.lower() in finetuned_summary.lower())
                base_preserved = sum(1 for term in tech_terms if term.lower() in base_summary.lower())

                self.results['finetuned']['tech_preservation'].append(
                    finetuned_preserved / len(tech_terms)
                )
                self.results['base']['tech_preservation'].append(
                    base_preserved / len(tech_terms)
                )

        # Calculate means
        summary_results = {}
        for model_name in ['finetuned', 'base']:
            summary_results[model_name] = {}
            for metric in ['rouge1', 'rouge2', 'rougeL']:
                if self.results[model_name][metric]:
                    summary_results[model_name][metric] = float(np.mean(self.results[model_name][metric]))
                else:
                    summary_results[model_name][metric] = 0.0

            if self.results[model_name]['tech_preservation']:
                summary_results[model_name]['tech_preservation'] = float(np.mean(self.results[model_name]['tech_preservation']))
            else:
                summary_results[model_name]['tech_preservation'] = 0.0

        # Save results
        os.makedirs('evaluation', exist_ok=True)

        with open('evaluation/comparison_results.json', 'w') as f:
            json.dump(summary_results, f, indent=2)

        if self.bad_predictions:
            with open("evaluation/bad_predictions.json", "w") as f:
                json.dump(self.bad_predictions[:3], f, indent=2)
            print(f"📁 Saved top {min(3, len(self.bad_predictions))} bad predictions to evaluation/bad_predictions.json")

        # Print results
        print("\n📊 Evaluation Results:")
        print(f"\n{'Model':<15} {'ROUGE-1':<10} {'ROUGE-2':<10} {'ROUGE-L':<10} {'Tech Terms':<12}")
        print("-" * 60)

        for model_name in ['finetuned', 'base']:
            print(f"{model_name.capitalize():<15} "
                  f"{summary_results[model_name]['rouge1']:<10.4f} "
                  f"{summary_results[model_name]['rouge2']:<10.4f} "
                  f"{summary_results[model_name]['rougeL']:<10.4f} "
                  f"{summary_results[model_name]['tech_preservation']:<12.4f}")

        # Log results if logger is available
        try:
            from utils.logger import log_run_results
            log_run_results(
                filepath="results_table.csv",
                run_name=getattr(config, 'run_name', 'evaluation_run'),
                learning_rate=getattr(config, 'LEARNING_RATE', 2e-5),
                batch_size=getattr(config, 'BATCH_SIZE', 8),
                num_epochs=getattr(config, 'NUM_EPOCHS', 3),
                rouge1=summary_results['finetuned']['rouge1'],
                rouge2=summary_results['finetuned']['rouge2'],
                rougeL=summary_results['finetuned']['rougeL'],
            )
            print("\n✅ Results logged to results_table.csv")
        except ImportError:
            print("\n⚠️ Logger not available, skipping CSV logging")

        return summary_results

    def generate_comparison_plots(self):
        """Generate comparison plots between models"""
        os.makedirs('evaluation/plots', exist_ok=True)

        # ROUGE scores comparison
        metrics = ['rouge1', 'rouge2', 'rougeL']
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        for idx, metric in enumerate(metrics):
            finetuned_scores = self.results['finetuned'][metric]
            base_scores = self.results['base'][metric]

            if finetuned_scores and base_scores:
                axes[idx].boxplot([finetuned_scores, base_scores],
                                labels=['Fine-tuned', 'Base'])
                axes[idx].set_title(f'{metric.upper()} Scores')
                axes[idx].set_ylabel('Score')

        plt.tight_layout()
        plt.savefig('evaluation/plots/rouge_comparison.png', dpi=300)
        print("📊 Saved ROUGE comparison plot to evaluation/plots/rouge_comparison.png")

        # Tech term preservation comparison
        if self.results['finetuned']['tech_preservation'] and self.results['base']['tech_preservation']:
            plt.figure(figsize=(8, 6))
            data = [self.results['finetuned']['tech_preservation'],
                   self.results['base']['tech_preservation']]
            plt.boxplot(data, labels=['Fine-tuned', 'Base'])
            plt.title('Technical Term Preservation')
            plt.ylabel('Preservation Rate')
            plt.savefig('evaluation/plots/tech_preservation.png', dpi=300)
            print("📊 Saved tech preservation plot to evaluation/plots/tech_preservation.png")


def run_evaluation(model_path: str = "data/models/final_model",
                  test_data_path: str = "data/processed/test.json",
                  generate_plots: bool = True):
    """Main function to run evaluation - this is what run_training.py imports"""

    print("\n🔍 Starting model evaluation...")

    # Check if model exists
    if not os.path.exists(model_path):
        print(f"❌ Model not found at {model_path}")
        print("Please train the model first by running the training script.")
        return None

    # Check if test data exists
    if not os.path.exists(test_data_path):
        print(f"❌ Test data not found at {test_data_path}")
        print("Please create the dataset first.")
        return None

    try:
        # Create evaluator and run evaluation
        evaluator = ModelEvaluator(model_path)
        results = evaluator.evaluate_dataset(test_data_path)

        # Generate plots if requested
        if generate_plots and len(evaluator.results['finetuned']['rouge1']) > 0:
            try:
                evaluator.generate_comparison_plots()
            except Exception as e:
                print(f"⚠️ Could not generate plots: {e}")

        print("\n✅ Evaluation completed successfully!")
        return results

    except Exception as e:
        print(f"❌ Error during evaluation: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    # Run evaluation when script is called directly
    run_evaluation()