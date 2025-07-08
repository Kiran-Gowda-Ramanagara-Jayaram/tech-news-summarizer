"""Create training dataset with summaries"""

import json
import os
from typing import Dict, List
import random
from tqdm import tqdm


class DatasetCreator:
    def __init__(self):
        self.dataset = []

    def create_summaries(self, articles: List[Dict]) -> List[Dict]:
        """Create multiple summary types for each article"""

        print("📝 Creating summaries for articles...")

        for article in tqdm(articles, desc="Processing articles"):
            # Skip if content is too short
            if len(article['content']) < 300:
                continue

            # Extract key sentences for summary
            sentences = [s.strip() for s in article['content'].split('.') if len(s.strip()) > 20]

            if len(sentences) < 3:
                continue

            # Create different summary types
            summaries = self.generate_summary_variations(article, sentences)

            # Add to dataset
            for summary_type, summary_text in summaries.items():
                self.dataset.append({
                    'article_id': len(self.dataset),
                    'title': article['title'],
                    'content': article['content'],
                    'summary': summary_text,
                    'summary_type': summary_type,
                    'source': article['source']
                })

        return self.dataset

    def generate_summary_variations(self, article: Dict, sentences: List[str]) -> Dict[str, str]:
        """Generate different types of summaries"""

        summaries = {}

        # 1. Standard summary (2-3 sentences)
        if len(sentences) >= 3:
            standard = f"{sentences[0]}. {sentences[1]}."
            if len(standard.split()) < 40 and len(sentences) > 2:
                standard += f" {sentences[2]}."
            summaries['standard'] = self.clean_summary(standard)

        # 2. Bullet point summary
        bullet_points = []
        for i, sent in enumerate(sentences[:5]):
            if any(term in sent.lower() for term in ['announced', 'released', 'launched', 'unveiled', 'introduced']):
                bullet_points.append(f"• {sent.strip()}")
            elif i < 3:
                bullet_points.append(f"• {sent.strip()}")

        if bullet_points:
            summaries['bullets'] = '\n'.join(bullet_points[:4])

        # 3. Tweet-length summary (under 280 chars)
        tweet = sentences[0].strip()
        if len(tweet) > 250:
            tweet = tweet[:247] + "..."

        # Add hashtags based on content
        if 'apple' in article['content'].lower():
            tweet += " #Apple"
        if 'ai' in article['content'].lower() or 'gpt' in article['content'].lower():
            tweet += " #AI"
        if 'google' in article['content'].lower():
            tweet += " #Google"

        summaries['tweet'] = tweet[:280]

        # 4. Technical summary (focus on specs/numbers)
        tech_summary = []
        for sent in sentences:
            # Look for sentences with technical details
            if any(char.isdigit() for char in sent):
                tech_summary.append(sent.strip())
            elif any(term in sent.lower() for term in ['feature', 'performance', 'capability', 'technology']):
                tech_summary.append(sent.strip())

        if tech_summary:
            summaries['technical'] = '. '.join(tech_summary[:2]) + '.'
        else:
            summaries['technical'] = summaries.get('standard', sentences[0])

        return summaries

    def clean_summary(self, text: str) -> str:
        """Clean up summary text"""
        # Remove extra spaces
        text = ' '.join(text.split())
        # Ensure it ends with a period
        if text and text[-1] not in '.!?':
            text += '.'
        return text

    def split_dataset(self, train_ratio=0.7, val_ratio=0.15):
        """Split dataset into train/val/test sets"""

        random.shuffle(self.dataset)

        total = len(self.dataset)
        train_size = int(total * train_ratio)
        val_size = int(total * val_ratio)

        train_data = self.dataset[:train_size]
        val_data = self.dataset[train_size:train_size + val_size]
        test_data = self.dataset[train_size + val_size:]

        return {
            'train': train_data,
            'validation': val_data,
            'test': test_data
        }

    def save_dataset(self, output_dir='data/processed'):
        """Save the dataset splits"""

        os.makedirs(output_dir, exist_ok=True)

        # Create splits
        splits = self.split_dataset()

        # Save each split
        for split_name, split_data in splits.items():
            filename = os.path.join(output_dir, f'{split_name}.json')
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(split_data, f, indent=2, ensure_ascii=False)

            print(f"💾 Saved {split_name} set: {len(split_data)} examples to {filename}")

        # Save complete dataset
        with open(os.path.join(output_dir, 'complete_dataset.json'), 'w') as f:
            json.dump(self.dataset, f, indent=2, ensure_ascii=False)

        print(f"\n✅ Total dataset size: {len(self.dataset)} examples")
        print(f"   - Train: {len(splits['train'])}")
        print(f"   - Validation: {len(splits['validation'])}")
        print(f"   - Test: {len(splits['test'])}")


def create_dataset_from_articles():
    """Main function to create dataset"""

    # Load articles
    try:
        with open('data/raw/tech_articles.json', 'r', encoding='utf-8') as f:
            articles = json.load(f)
    except FileNotFoundError:
        print("❌ No articles found. Run data_collector.py first!")
        return

    print(f"📚 Loaded {len(articles)} articles")

    # Create dataset
    creator = DatasetCreator()
    dataset = creator.create_summaries(articles)

    # Save dataset
    creator.save_dataset()


if __name__ == "__main__":
    create_dataset_from_articles()