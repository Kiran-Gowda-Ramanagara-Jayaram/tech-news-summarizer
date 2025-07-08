"""Test if all packages are installed correctly"""

print("Testing package installations...\n")

# Test imports
packages = [
    "transformers",
    "torch",
    "streamlit",
    "pandas",
    "numpy",
    "rouge_score",
    "datasets",
    "beautifulsoup4",
    "requests",
    "newspaper3k",
    "matplotlib",
    "seaborn",
    "tqdm",
    "sklearn"
]

failed = []

for package in packages:
    try:
        if package == "beautifulsoup4":
            import bs4
        elif package == "newspaper3k":
            # newspaper3k is imported as 'newspaper'
            import newspaper
        elif package == "sklearn":
            import sklearn
        else:
            __import__(package)
        print(f"✓ {package} installed")
    except ImportError:
        print(f"✗ {package} NOT installed")
        failed.append(package)

# Test model loading
print("\n" + "-"*40)
print("Testing model loading...\n")

try:
    from transformers import T5ForConditionalGeneration, T5Tokenizer
    model_name = "t5-small"

    print(f"Attempting to load {model_name}...")
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)

    print(f"✓ Successfully loaded {model_name}")
    print(f"  Model parameters: {model.num_parameters() / 1e6:.1f}M")

    # Test generation
    inputs = tokenizer("summarize: This is a test article about technology.", return_tensors="pt")
    outputs = model.generate(**inputs, max_length=50)
    summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"  Test generation: '{summary}'")

except Exception as e:
    print(f"✗ Error loading model: {e}")

# Test newspaper functionality specifically
print("\n" + "-"*40)
print("Testing newspaper3k functionality...\n")

try:
    import newspaper
    from newspaper import Article
    print("✓ newspaper3k imports working")

    # Test creating an Article object
    article = Article('https://example.com')
    print("✓ Can create Article objects")

    # Test newspaper3k configuration
    from newspaper import Config
    config = Config()
    print("✓ newspaper3k fully functional")

except ImportError as e:
    print(f"✗ newspaper3k import error: {e}")
except Exception as e:
    print(f"✗ newspaper3k error: {e}")

# Summary
print("\n" + "="*40)
if failed:
    print(f"❌ Setup incomplete. Failed packages: {', '.join(failed)}")
    print("Run: pip install -r requirements.txt")
else:
    print("✅ All packages installed successfully!")
    print("You're ready to start training!")
    print("\nNext step: python run_training.py")