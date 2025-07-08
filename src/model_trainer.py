"""Fine-tune T5 for tech news summarization"""

import torch
from transformers import (
    T5ForConditionalGeneration,
    T5Tokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback
)
from torch.utils.data import DataLoader
import wandb
from datetime import datetime
import os
import numpy as np
from rouge_score import rouge_scorer
from src import config


# ✅ Compute metrics moved OUTSIDE the class with ROBUST error handling
def compute_metrics(eval_pred):
    import numpy as np
    from rouge_score import rouge_scorer
    import torch

    try:
        predictions, labels = eval_pred

        # Handle logits tuple
        if isinstance(predictions, tuple):
            predictions = predictions[0]

        # ✅ FIX: Handle predictions that might be logits
        if predictions.ndim == 3 and predictions.shape[-1] > 100:  # Likely vocabulary size
            # If predictions are logits, get the token ids
            predictions = np.argmax(predictions, axis=-1)

        # Decode predictions
        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)

        # Convert labels to numpy
        if isinstance(labels, torch.Tensor):
            labels = labels.numpy()

        # ✅ ROBUST FIX: Handle various label formats
        # Ensure labels is a numpy array
        labels = np.array(labels)

        # If labels is 1D, reshape to 2D
        if labels.ndim == 1:
            labels = labels.reshape(1, -1)

        # If labels has more than 2 dimensions, squeeze
        while labels.ndim > 2:
            labels = labels.squeeze(0)

        # Handle nested lists/object arrays
        if labels.dtype == object:
            # This handles cases where labels might be a list of lists
            try:
                # Try to find the maximum length
                max_len = 0
                for row in labels:
                    if isinstance(row, (list, np.ndarray)):
                        max_len = max(max_len, len(row))
                    else:
                        max_len = max(max_len, 1)

                # Create new array with proper shape
                new_labels = np.full((len(labels), max_len), tokenizer.pad_token_id, dtype=np.int32)

                for i, row in enumerate(labels):
                    if isinstance(row, (list, np.ndarray)):
                        row_array = np.array(row)
                        new_labels[i, :len(row_array)] = row_array[:max_len]
                    else:
                        new_labels[i, 0] = int(row) if row is not None else tokenizer.pad_token_id

                labels = new_labels
            except Exception as e:
                print(f"⚠️ Error reshaping labels: {e}")
                # Fallback: return dummy metrics
                return {
                    "rouge1": 0.0,
                    "rouge2": 0.0,
                    "rougeL": 0.0,
                }

        # Replace -100 with pad token id
        labels = np.where(labels == -100, tokenizer.pad_token_id, labels)

        # Ensure integer type
        labels = labels.astype(np.int32)

        # Decode labels
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        # Calculate ROUGE scores
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        scores = []

        for pred, label in zip(decoded_preds, decoded_labels):
            # Skip empty predictions/labels
            if pred and label:
                score = scorer.score(label, pred)
                scores.append(score)

        if not scores:
            print("⚠️ No valid predictions to score")
            return {
                "rouge1": 0.0,
                "rouge2": 0.0,
                "rougeL": 0.0,
            }

        rouge1 = np.mean([s['rouge1'].fmeasure for s in scores])
        rouge2 = np.mean([s['rouge2'].fmeasure for s in scores])
        rougeL = np.mean([s['rougeL'].fmeasure for s in scores])

        return {
            "rouge1": round(rouge1, 4),
            "rouge2": round(rouge2, 4),
            "rougeL": round(rougeL, 4),
        }

    except Exception as e:
        print(f"❌ Error in compute_metrics: {e}")
        import traceback
        traceback.print_exc()
        # Return dummy metrics to allow training to continue
        return {
            "rouge1": 0.0,
            "rouge2": 0.0,
            "rougeL": 0.0,
        }


# ✅ Instantiate tokenizer globally for metric function
tokenizer = T5Tokenizer.from_pretrained(config.MODEL_NAME)
tokenizer.add_special_tokens({'additional_special_tokens': ['<COMPANY>', '<PRODUCT>', '<VERSION>', '<TECH>']})


class TechSummaryTrainer:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🖥️ Using device: {self.device}")

        self.tokenizer = tokenizer  # Use shared tokenizer
        self.model = T5ForConditionalGeneration.from_pretrained(config.MODEL_NAME)
        self.model.resize_token_embeddings(len(self.tokenizer))
        self.model.to(self.device)

        try:
            wandb.init(
                project="tech-news-summarizer",
                name=f"run_{datetime.now():%Y%m%d_%H%M%S}",
                config={
                    "model": config.MODEL_NAME,
                    "batch_size": config.BATCH_SIZE,
                    "learning_rate": config.LEARNING_RATE,
                    "epochs": config.NUM_EPOCHS
                }
            )
        except:
            print("⚠️ Weights & Biases not initialized. Continuing without logging.")

    def train(self, train_dataset, val_dataset, output_dir='data/models'):
        os.makedirs(output_dir, exist_ok=True)

        data_collator = DataCollatorForSeq2Seq(
            tokenizer=self.tokenizer,
            model=self.model,
            padding=True,
            label_pad_token_id=-100
        )

        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=config.NUM_EPOCHS,
            per_device_train_batch_size=config.BATCH_SIZE,
            per_device_eval_batch_size=config.BATCH_SIZE,
            warmup_steps=config.WARMUP_STEPS,
            weight_decay=0.01,
            logging_dir=config.LOG_DIR,
            logging_steps=50,
            evaluation_strategy="steps",
            eval_steps=100,
            save_strategy="steps",
            save_steps=500,
            load_best_model_at_end=True,
            metric_for_best_model="rougeL",
            greater_is_better=True,
            push_to_hub=False,
            report_to=["wandb"] if wandb.run else [],
            run_name=f"tech_summarizer_{datetime.now():%Y%m%d_%H%M%S}",
            fp16=torch.cuda.is_available(),
            gradient_checkpointing=True,
            gradient_accumulation_steps=2 if config.BATCH_SIZE < 8 else 1,
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=self.tokenizer,
            compute_metrics=compute_metrics,  # ✅ FIXED: No self.
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
        )

        print("🚀 Starting training...")
        print(f"   - Train samples: {len(train_dataset)}")
        print(f"   - Validation samples: {len(val_dataset)}")
        print(f"   - Epochs: {config.NUM_EPOCHS}")
        print(f"   - Batch size: {config.BATCH_SIZE}")

        trainer.train()

        final_model_path = os.path.join(output_dir, 'final_model')
        trainer.save_model(final_model_path)
        self.tokenizer.save_pretrained(final_model_path)

        print(f"✅ Training complete! Model saved to {final_model_path}")
        return trainer

    def evaluate_on_test(self, trainer, test_dataset):
        print("\n📊 Evaluating on test set...")
        try:
            results = trainer.evaluate(eval_dataset=test_dataset)

            print("\n📈 Test Results:")
            for key, value in results.items():
                if key.startswith('eval_'):
                    metric_name = key.replace('eval_', '')
                    if isinstance(value, (int, float)):
                        print(f"   {metric_name}: {value:.4f}")

            os.makedirs('evaluation', exist_ok=True)
            with open('evaluation/test_results.json', 'w') as f:
                import json
                json.dump(results, f, indent=2)

            return results
        except Exception as e:
            print(f"⚠️ Error during test evaluation: {e}")
            return None

    def generate_examples(self, test_dataset, num_examples=5):
        print(f"\n📝 Generating {num_examples} example summaries...")
        self.model.eval()
        examples = []

        for i in range(min(num_examples, len(test_dataset))):
            try:
                item = test_dataset[i]

                # ✅ FIX: Handle tensor creation more carefully
                input_ids = item['input_ids']
                attention_mask = item['attention_mask']

                # Ensure they are tensors
                if not isinstance(input_ids, torch.Tensor):
                    input_ids = torch.tensor(input_ids)
                if not isinstance(attention_mask, torch.Tensor):
                    attention_mask = torch.tensor(attention_mask)

                # Ensure tensors are properly shaped
                if input_ids.dim() == 1:
                    input_ids = input_ids.unsqueeze(0)
                if attention_mask.dim() == 1:
                    attention_mask = attention_mask.unsqueeze(0)

                # Clone tensors before moving to device to avoid in-place operations
                input_ids = input_ids.clone().detach().to(self.device)
                attention_mask = attention_mask.clone().detach().to(self.device)

                with torch.no_grad():
                    outputs = self.model.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        max_length=150,
                        num_beams=4,
                        early_stopping=True,
                        no_repeat_ngram_size=3
                    )

                input_text = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
                generated_summary = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

                # ✅ FIX: Better handling of labels
                labels = item['labels']

                # Convert to tensor if needed
                if isinstance(labels, list):
                    labels = torch.tensor(labels)
                elif not isinstance(labels, torch.Tensor):
                    labels = torch.tensor(labels)

                # Ensure 1D tensor
                if labels.ndim > 1:
                    labels = labels.flatten()

                # Replace -100 with pad token id
                labels = torch.where(labels != -100, labels, self.tokenizer.pad_token_id)

                # ✅ FIX: Convert to list of ints properly
                label_list = labels.cpu().numpy().astype(int).tolist()

                # Decode reference summary
                reference_summary = self.tokenizer.decode(label_list, skip_special_tokens=True)

                examples.append({
                    'input': input_text[:200] + "...",
                    'generated': generated_summary,
                    'reference': reference_summary
                })

                print(f"\n--- Example {i + 1} ---")
                print(f"Generated: {generated_summary}")
                print(f"Reference: {reference_summary}")

            except Exception as e:
                print(f"⚠️ Error processing example {i}: {str(e)}")
                continue

        # Save examples
        if examples:
            os.makedirs('evaluation', exist_ok=True)
            with open('evaluation/generation_examples.json', 'w') as f:
                import json
                json.dump(examples, f, indent=2)
            print(f"\n📁 Saved {len(examples)} examples to evaluation/generation_examples.json")

        return examples


def train_model():
    from src.data_processor import DataProcessor

    print("📁 Loading datasets...")
    processor = DataProcessor()
    datasets = processor.load_and_prepare_data()

    if not datasets:
        print("❌ Failed to load datasets. Run dataset_creator.py first.")
        return

    trainer = TechSummaryTrainer()
    trained_model = trainer.train(datasets['train'], datasets['validation'])

    if 'test' in datasets:
        try:
            trainer.evaluate_on_test(trained_model, datasets['test'])
            trainer.generate_examples(datasets['test'])
        except Exception as e:
            print(f"⚠️ Error during evaluation: {str(e)}")
            print("Training completed successfully, but evaluation encountered issues.")

    print("\n🎉 All done!")


if __name__ == "__main__":
    train_model()