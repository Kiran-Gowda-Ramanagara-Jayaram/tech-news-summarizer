"""Collect tech news articles from various sources"""

import requests
from bs4 import BeautifulSoup
import json
from datetime import datetime
import time
import os
from newspaper import Article
import pandas as pd
from tqdm import tqdm


class TechNewsCollector:
    def __init__(self):
        self.articles = []

    def scrape_techcrunch(self, num_articles=100):
        """Scrape articles from TechCrunch"""
        print("🔍 Scraping TechCrunch...")
        base_url = "https://techcrunch.com/wp-json/wp/v2/posts"

        try:
            response = requests.get(base_url, params={"per_page": min(num_articles, 100)})
            posts = response.json()

            for post in tqdm(posts, desc="Collecting TechCrunch articles"):
                article_data = {
                    'source': 'TechCrunch',
                    'title': post['title']['rendered'],
                    'url': post['link'],
                    'date': post['date'],
                    'content': self.clean_html(post['content']['rendered']),
                    'excerpt': self.clean_html(post['excerpt']['rendered'])
                }

                if len(article_data['content']) > 200:  # Skip very short articles
                    self.articles.append(article_data)

                time.sleep(0.5)  # Be respectful to the API

        except Exception as e:
            print(f"Error scraping TechCrunch: {e}")

    def scrape_with_newspaper(self, urls):
        """Use newspaper3k to extract articles"""
        for url in tqdm(urls, desc="Extracting articles"):
            try:
                article = Article(url)
                article.download()
                article.parse()

                if len(article.text) > 200:
                    self.articles.append({
                        'source': 'Custom',
                        'title': article.title,
                        'url': url,
                        'date': str(datetime.now()),
                        'content': article.text,
                        'excerpt': article.text[:200] + "..."
                    })

                time.sleep(1)

            except Exception as e:
                print(f"Error extracting {url}: {e}")

    def clean_html(self, html):
        """Remove HTML tags"""
        soup = BeautifulSoup(html, 'html.parser')
        text = soup.get_text()
        # Clean up whitespace
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = ' '.join(chunk for chunk in chunks if chunk)
        return text

    def add_sample_articles(self):
        """Add high-quality sample articles for immediate testing"""
        samples = [
            {
                'source': 'Sample',
                'title': 'Apple Unveils M3 Chip with Revolutionary 3nm Technology',
                'url': 'sample_1',
                'date': str(datetime.now()),
                'content': """Apple today announced the M3 chip, featuring groundbreaking 3-nanometer process technology. The new chip delivers up to 20% faster CPU performance and 30% faster GPU performance compared to M2. With support for up to 128GB of unified memory and advanced machine learning capabilities, the M3 represents a significant leap in Apple Silicon development. 

                The M3 features an 8-core CPU with 4 performance and 4 efficiency cores, along with a 10-core GPU. Apple claims the new chip offers industry-leading performance per watt, making it ideal for both portable and desktop machines. The neural engine has been upgraded to 16 cores, providing up to 60% faster machine learning performance.

                The chip will first appear in the updated MacBook Pro lineup, starting at $1,599. Apple expects to transition its entire Mac lineup to M3 within the next year. Early benchmarks show the M3 outperforming Intel's latest mobile processors by a significant margin while using half the power.""",
                'excerpt': 'Apple announces M3 chip with 3nm technology...'
            },
            {
                'source': 'Sample',
                'title': 'OpenAI Releases GPT-4 Turbo with 128K Context Window',
                'url': 'sample_2',
                'date': str(datetime.now()),
                'content': """OpenAI has released GPT-4 Turbo, a significant upgrade to its flagship language model. The new model features a 128,000 token context window, allowing it to process entire books in a single conversation. This represents a 16x increase from the original GPT-4's 8K context window.

                GPT-4 Turbo introduces improved instruction following, better knowledge of world events up to April 2023, and reduced pricing at $0.01 per 1K input tokens and $0.03 per 1K output tokens. The model shows significant improvements in mathematical reasoning, code generation, and multimodal capabilities.

                Developers can access GPT-4 Turbo through the OpenAI API with the model name 'gpt-4-1106-preview'. Early benchmarks show the model achieving state-of-the-art performance on various NLP tasks while being 3x cheaper than the previous GPT-4 model. OpenAI also announced a new Assistants API that makes it easier to build AI-powered applications.""",
                'excerpt': 'OpenAI releases GPT-4 Turbo with massive context window...'
            },
            {
                'source': 'Sample',
                'title': 'Google Announces Gemini AI Model to Compete with GPT-4',
                'url': 'sample_3',
                'date': str(datetime.now()),
                'content': """Google has unveiled Gemini, its most capable AI model to date, designed to compete directly with OpenAI's GPT-4. Gemini comes in three sizes: Ultra, Pro, and Nano, catering to different use cases from data centers to mobile devices.

                Gemini Ultra achieves state-of-the-art performance on 30 of 32 widely-used academic benchmarks, including MMLU where it scored 90.0%, becoming the first model to outperform human experts. The model demonstrates sophisticated multimodal capabilities, seamlessly combining text, images, audio, and video understanding.

                Google plans to integrate Gemini across its product ecosystem, starting with Bard, which will use a fine-tuned version of Gemini Pro. The Pixel 8 Pro will be the first smartphone to run Gemini Nano, enabling new AI features that work entirely on-device. Gemini Ultra will be available to developers and enterprise customers in early 2024.""",
                'excerpt': 'Google unveils Gemini AI to challenge GPT-4...'
            }
        ]

        self.articles.extend(samples)
        print(f"✓ Added {len(samples)} sample articles")

    def save_articles(self, filename='data/raw/tech_articles.json'):
        """Save collected articles"""
        os.makedirs(os.path.dirname(filename), exist_ok=True)

        # Remove duplicates based on title
        seen_titles = set()
        unique_articles = []
        for article in self.articles:
            if article['title'] not in seen_titles:
                seen_titles.add(article['title'])
                unique_articles.append(article)

        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(unique_articles, f, indent=2, ensure_ascii=False)

        print(f"💾 Saved {len(unique_articles)} unique articles to {filename}")

        # Also save as CSV for easy viewing
        df = pd.DataFrame(unique_articles)
        csv_filename = filename.replace('.json', '.csv')
        df.to_csv(csv_filename, index=False)
        print(f"📊 Also saved as {csv_filename}")


def collect_initial_data():
    """Quick function to collect initial dataset"""
    collector = TechNewsCollector()

    # Add sample articles
    collector.add_sample_articles()

    # Try to collect some real articles
    print("\n🌐 Attempting to collect real articles...")
    collector.scrape_techcrunch(num_articles=20)

    # Save everything
    collector.save_articles()

    return len(collector.articles)


if __name__ == "__main__":
    num_articles = collect_initial_data()
    print(f"\n✅ Data collection complete! Total articles: {num_articles}")