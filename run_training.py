"""Main script to run the complete training pipeline"""

import os
import sys
import time
from datetime import datetime


def print_step(step_num, description):
    """Pretty print for steps"""
    print(f"\n{'=' * 60}")
    print(f"🔹 STEP {step_num}: {description}")
    print(f"{'=' * 60}")


def main():
    start_time = time.time()

    print("\n🚀 TECH NEWS SUMMARIZER - TRAINING PIPELINE")
    print(f"Started at: {datetime.now():%Y-%m-%d %H:%M:%S}")

    # Step 1: Collect Data
    print_step(1, "Collecting Tech News Articles")
    from src.data_collector import collect_initial_data
    try:
        num_articles = collect_initial_data()
        print(f"✅ Collected {num_articles} articles")
    except Exception as e:
        print(f"❌ Error collecting data: {e}")
        return

    # Step 2: Create Dataset
    print_step(2, "Creating Training Dataset")
    from src.dataset_creator import create_dataset_from_articles
    try:
        create_dataset_from_articles()
        print("✅ Dataset created successfully")
    except Exception as e:
        print(f"❌ Error creating dataset: {e}")
        return

    # Step 3: Train Model
    print_step(3, "Training the Model")
    from src.model_trainer import train_model
    try:
        train_model()
        print("✅ Model trained successfully")
    except Exception as e:
        print(f"❌ Error training model: {e}")
        return

    # Step 4: Evaluate Model
    print_step(4, "Evaluating the Model")
    from src.evaluator import run_evaluation
    try:
        run_evaluation()
        print("✅ Evaluation completed")
    except Exception as e:
        print(f"❌ Error evaluating model: {e}")
        return

    # Final summary
    elapsed_time = time.time() - start_time
    print(f"\n{'=' * 60}")
    print(f"✨ TRAINING PIPELINE COMPLETED!")
    print(f"Total time: {elapsed_time / 60:.1f} minutes")
    print(f"{'=' * 60}")

    print("\n📋 Next steps:")
    print("1. Run the Streamlit app: streamlit run app/streamlit_app.py")
    print("2. Check evaluation results in the 'evaluation' folder")
    print("3. Model saved in 'data/models/final_model'")


if __name__ == "__main__":
    main()