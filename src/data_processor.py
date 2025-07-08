"""Process and prepare data for training"""

import json
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import T5Tokenizer
import torch
from torch.utils.data import Dataset, DataLoader
import re
from typing import List, Dict
from src import config

class TechNewsDataset(Dataset):
    def __init__(self, data, tokenizer, max_input_length=512, max_target_length=150):
        self.data = data
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.max_target_length = max_target_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # Add prefix based on summary type
        prefix_map = {
            'standard': 'summarize: ',
            'bullets': 'summarize in bullet points: ',
            'tweet': 'summarize in one tweet: ',
            'technical': 'extract technical details: '
        }

        prefix = prefix_map.get(item.get('summary_type', 'standard'), 'summarize: ')
        input_text = prefix + item['content']
        target_text = item['summary']

        # Tokenize inputs
        input_encoding = self.tokenizer(
            input_text,
            max_length=self.max_input_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        # Tokenize targets
        with self.tokenizer.as_target_tokenizer():
            target_encoding = self.tokenizer(
                target_text,
                max_length=self.max_target_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )

        labels = target_encoding.input_ids
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            'input_ids': input_encoding.input_ids.flatten(),
            'attention_mask': input_encoding.attention_mask.flatten(),
            'labels': labels.flatten()
        }


class DataProcessor:
    def __init__(self):
        self.tokenizer = T5Tokenizer.from_pretrained(config.MODEL_NAME)

        # Add special tokens for tech terms
        special_tokens = ['<COMPANY>', '<PRODUCT>', '<VERSION>', '<TECH>']
        self.tokenizer.add_special_tokens({'additional_special_tokens': special_tokens})

    def preprocess_text(self, text: str) -> str:
        """Preprocess tech articles with special handling"""

        # Preserve tech companies
        for company in config.TECH_COMPANIES:
            # Case-insensitive replacement
            pattern = re.compile(re.escape(company), re.IGNORECASE)
            text = pattern.sub(f"<COMPANY>{company}</COMPANY>", text)

        # Mark version numbers
        for pattern in config.VERSION_PATTERNS:
            text = re.sub(pattern, lambda m: f"<VERSION>{m.group()}</VERSION>", text)

        # Mark technical terms
        for term in config.TECH_TERMS:
            if term.lower() in text.lower():
                pattern = re.compile(r'\b' + re.escape(term) + r'\b', re.IGNORECASE)
                text = pattern.sub(f"<TECH>{term}</TECH>", text)

        return text

    def load_and_prepare_data(self, data_dir='data/processed'):
        """Load and prepare datasets"""

        datasets = {}

        for split in ['train', 'validation', 'test']:
            file_path = f"{data_dir}/{split}.json"
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                # Preprocess texts
                for item in data:
                    item['content'] = self.preprocess_text(item['content'])

                # Create dataset
                datasets[split] = TechNewsDataset(
                    data,
                    self.tokenizer,
                    max_input_length=config.MAX_INPUT_LENGTH,
                    max_target_length=config.MAX_TARGET_LENGTH
                )

                print(f"✅ Loaded {split} dataset: {len(data)} examples")

            except FileNotFoundError:
                print(f"❌ {split} dataset not found at {file_path}")
                return None

        return datasets

    def create_data_loaders(self, datasets, batch_size=None):
        """Create DataLoaders for training"""

        if batch_size is None:
            batch_size = config.BATCH_SIZE

        loaders = {}

        for split, dataset in datasets.items():
            shuffle = (split == 'train')
            loaders[split] = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=0,  # Set to 0 for Windows compatibility
                pin_memory=torch.cuda.is_available()
            )

        return loaders


def test_data_loading():
    """Test function to verify data loading works"""
    processor = DataProcessor()
    datasets = processor.load_and_prepare_data()

    if datasets and 'train' in datasets:
        # Test one batch
        sample = datasets['train'][0]
        print("\n📊 Sample data point:")
        print(f"Input shape: {sample['input_ids'].shape}")
        print(f"Labels shape: {sample['labels'].shape}")

        # Decode to see what it looks like
        input_text = processor.tokenizer.decode(sample['input_ids'], skip_special_tokens=False)
        target_text = processor.tokenizer.decode(sample['labels'], skip_special_tokens=True)

        print(f"\nInput preview: {input_text[:200]}...")
        print(f"Target: {target_text}")


if __name__ == "__main__":
    test_data_loading()