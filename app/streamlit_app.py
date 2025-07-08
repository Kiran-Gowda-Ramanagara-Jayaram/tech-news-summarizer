"""Interactive Tech News Summarizer App"""

import streamlit as st
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
import re
from datetime import datetime
import json
import os
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import config
# Configure page
st.set_page_config(
    page_title="Tech News Summarizer AI",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main {
        padding: 2rem;
    }
    .stButton > button {
        background-color: #4CAF50;
        color: white;
        border-radius: 5px;
        padding: 0.5rem 1rem;
        width: 100%;
        border: none;
        transition: 0.3s;
    }
    .stButton > button:hover {
        background-color: #45a049;
        transform: translateY(-2px);
    }
    .summary-box {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 4px solid #4CAF50;
    }
    .metric-card {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
        transition: 0.3s;
    }
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
    }
    .entity-tag {
        display: inline-block;
        padding: 0.25rem 0.5rem;
        margin: 0.25rem;
        background-color: #e3f2fd;
        border-radius: 15px;
        font-size: 0.875rem;
    }
    h1 {
        color: #1a73e8;
    }
    .info-box {
        background-color: #e8f5e9;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    """Load the fine-tuned model"""
    model_path = "../data/models/final_model"

    if os.path.exists(model_path):
        try:
            model = T5ForConditionalGeneration.from_pretrained(model_path)
            tokenizer = T5Tokenizer.from_pretrained(model_path)
            return model, tokenizer, True
        except Exception as e:
            st.error(f"Error loading fine-tuned model: {e}")

    # Fallback to base model
    st.warning("Fine-tuned model not found. Using base T5 model.")
    model = T5ForConditionalGeneration.from_pretrained("t5-small")
    tokenizer = T5Tokenizer.from_pretrained("t5-small")
    return model, tokenizer, False


def generate_summary(text, model, tokenizer, summary_type="standard", max_length=150):
    """Generate summary using the model"""

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # Prepare prompt based on summary type
    prompt_map = {
        "standard": "summarize: ",
        "bullet": "summarize in bullet points: ",
        "tweet": "summarize in one tweet: ",
        "technical": "extract technical details: "
    }

    prompt = prompt_map.get(summary_type, "summarize: ") + text

    # Adjust max_length based on type
    if summary_type == "tweet":
        max_length = 60
    elif summary_type == "bullet":
        max_length = 200

    # Tokenize
    inputs = tokenizer(
        prompt,
        max_length=512,
        truncation=True,
        return_tensors="pt"
    ).to(device)

    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            min_length=20,
            num_beams=4,
            early_stopping=True,
            no_repeat_ngram_size=3,
            temperature=0.8
        )

    summary = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Post-process for bullet points
    if summary_type == "bullet" and "•" not in summary:
        sentences = summary.split(". ")
        summary = "\n".join([f"• {s.strip()}" for s in sentences if s.strip()])

    return summary


def extract_tech_entities(text):
    """Extract technical entities from text"""
    entities = {
        "companies": [],
        "versions": [],
        "products": [],
        "tech_terms": []
    }

    # Extract companies
    for company in config.TECH_COMPANIES:
        if company.lower() in text.lower():
            entities["companies"].append(company)

    # Extract version numbers
    version_pattern = r'v?\d+\.\d+(?:\.\d+)?'
    entities["versions"] = list(set(re.findall(version_pattern, text)))

    # Extract tech terms
    for term in config.TECH_TERMS:
        if term.lower() in text.lower():
            entities["tech_terms"].append(term)

    # Extract potential product names (capitalized multi-word phrases)
    product_pattern = r'\b[A-Z][a-zA-Z]+(?:\s[A-Z][a-zA-Z]+)*\b'
    potential_products = re.findall(product_pattern, text)
    entities["products"] = [p for p in potential_products
                            if len(p) > 3 and p not in entities["companies"]][:5]

    return entities


# Sample articles for demo
SAMPLE_ARTICLES = {
    "None": "",
    "Apple M3 Chip": """Apple today announced the M3 chip, featuring groundbreaking 3-nanometer process technology. The new chip delivers up to 20% faster CPU performance and 30% faster GPU performance compared to M2. With support for up to 128GB of unified memory and advanced machine learning capabilities, the M3 represents a significant leap in Apple Silicon development. The chip features an 8-core CPU with 4 performance and 4 efficiency cores, along with a 10-core GPU. The M3 will first appear in the updated MacBook Pro lineup, starting at $1,599.""",

    "GPT-4 Turbo Release": """OpenAI released GPT-4 Turbo today, introducing a 128K token context window that can process entire books in one conversation. The model features improved instruction following and updated knowledge through April 2023. Pricing has been reduced to $0.01 per 1K input tokens and $0.03 per 1K output tokens, making it 3x cheaper than GPT-4. Available via API as 'gpt-4-1106-preview', the model shows state-of-the-art performance on NLP benchmarks.""",

    "NVIDIA H100 GPU": """NVIDIA unveiled the H100 GPU, built on the Hopper architecture with 80 billion transistors. The GPU delivers up to 9x faster AI training and 30x faster AI inference compared to the A100. With 80GB of HBM3 memory and 3TB/s bandwidth, the H100 targets large language model training and high-performance computing. The chip supports new FP8 precision and includes a Transformer Engine for accelerated AI workloads. Pricing starts at $30,000 per unit."""
}


def main():
    st.title("🤖 Tech News Summarizer AI")
    st.markdown("### Fine-tuned AI model specialized for technology news summarization")

    # Load model
    with st.spinner("Loading AI model..."):
        model, tokenizer, is_finetuned = load_model()

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")

        summary_type = st.selectbox(
            "Summary Type",
            ["standard", "bullet", "tweet", "technical"],
            format_func=lambda x: x.title().replace("_", " "),
            help="Choose the format for your summary"
        )

        st.markdown("---")

        # Model info
        st.header("📊 Model Information")
        col1, col2 = st.columns(2)

        with col1:
            if is_finetuned:
                st.metric("Model Status", "Fine-tuned ✅")
            else:
                st.metric("Model Status", "Base Model ⚠️")

        with col2:
            st.metric("Model Size", "60M params")

        if is_finetuned:
            st.success("Using specialized tech news model")

            # Display training metrics if available
            if os.path.exists("evaluation/evaluation_report.json"):
                with open("evaluation/evaluation_report.json", 'r') as f:
                    metrics = json.load(f)

                st.markdown("### 📈 Performance Metrics")
                rouge_score = metrics.get('metrics', {}).get('rougeL', {}).get('finetuned', {}).get('mean', 0)
                improvement = metrics.get('metrics', {}).get('rougeL', {}).get('improvement_percent', 0)

                st.metric("ROUGE-L Score", f"{rouge_score:.3f}")
                st.metric("Improvement vs Base", f"+{improvement:.1f}%")

        st.markdown("---")

        # About section
        st.header("📖 About")
        st.info("""
        This AI model was fine-tuned specifically for summarizing technology news articles.

        **Features:**
        - Preserves technical terminology
        - Multiple summary formats
        - Entity extraction
        - Optimized for tech content
        """)

    # Main content area
    st.markdown("---")

    # Input section
    st.header("📝 Input Article")

    # Sample article selector
    col1, col2 = st.columns([3, 1])
    with col1:
        selected_sample = st.selectbox(
            "Load sample article:",
            list(SAMPLE_ARTICLES.keys()),
            help="Try a pre-loaded tech article"
        )

    article_text = st.text_area(
        "Paste your tech article here:",
        value=SAMPLE_ARTICLES[selected_sample],
        height=300,
        placeholder="Enter a technology news article to summarize..."
    )

    # Generate button
    if st.button("🚀 Generate Summary", type="primary"):
        if article_text:
            # Create columns for output
            col1, col2 = st.columns([2, 1])

            with col1:
                with st.spinner("Generating summary..."):
                    # Generate summary
                    summary = generate_summary(
                        article_text,
                        model,
                        tokenizer,
                        summary_type
                    )

                    # Display summary
                    st.header("📄 Generated Summary")
                    st.markdown(f'<div class="summary-box">{summary}</div>',
                                unsafe_allow_html=True)

                    # Metrics
                    st.markdown("### 📊 Summary Statistics")
                    col1_m, col2_m, col3_m = st.columns(3)

                    with col1_m:
                        original_words = len(article_text.split())
                        st.metric("Original", f"{original_words} words")

                    with col2_m:
                        summary_words = len(summary.split())
                        st.metric("Summary", f"{summary_words} words")

                    with col3_m:
                        compression = (1 - summary_words / original_words) * 100
                        st.metric("Compression", f"{compression:.1f}%")

            with col2:
                # Entity extraction
                st.header("🔍 Extracted Entities")
                entities = extract_tech_entities(article_text)

                if entities['companies']:
                    st.markdown("**Companies:**")
                    for company in entities['companies']:
                        st.markdown(f'<span class="entity-tag">🏢 {company}</span>',
                                    unsafe_allow_html=True)

                if entities['versions']:
                    st.markdown("**Versions:**")
                    for version in entities['versions']:
                        st.markdown(f'<span class="entity-tag">📌 {version}</span>',
                                    unsafe_allow_html=True)

                if entities['tech_terms']:
                    st.markdown("**Tech Terms:**")
                    for term in entities['tech_terms'][:5]:
                        st.markdown(f'<span class="entity-tag">💻 {term}</span>',
                                    unsafe_allow_html=True)

                # Download button
                st.markdown("---")
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                st.download_button(
                    label="💾 Download Summary",
                    data=f"Original Article:\n{article_text}\n\n---\n\nSummary ({summary_type}):\n{summary}",
                    file_name=f"tech_summary_{timestamp}.txt",
                    mime="text/plain"
                )
        else:
            st.error("⚠️ Please enter an article to summarize!")

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 2rem;'>
        <p>Tech News Summarizer v1.0 | Fine-tuned T5 Model | Built for CS7980</p>
        <p>Made with ❤️ using Streamlit and Transformers</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()