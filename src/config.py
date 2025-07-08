"""Configuration file for Tech News Summarizer"""

# Model Configuration
MODEL_NAME = "t5-small"
MAX_INPUT_LENGTH = 512
MAX_TARGET_LENGTH = 150
BATCH_SIZE = 8
LEARNING_RATE = 3e-4
NUM_EPOCHS = 3
WARMUP_STEPS = 500

# Data Configuration
TRAIN_SIZE = 5000  # Start small
VAL_SIZE = 500
TEST_SIZE = 500

# Tech-specific keywords to preserve
TECH_COMPANIES = [
    "Apple", "Google", "Microsoft", "Amazon", "Meta", "Tesla", "NVIDIA",
    "Intel", "AMD", "Samsung", "OpenAI", "Anthropic", "SpaceX", "IBM",
    "Oracle", "Salesforce", "Adobe", "Netflix", "Uber", "Twitter"
]

TECH_TERMS = [
    "AI", "ML", "API", "GPU", "CPU", "RAM", "SSD", "5G", "IoT",
    "blockchain", "cryptocurrency", "metaverse", "quantum", "cloud",
    "DevOps", "microservices", "Kubernetes", "Docker", "AWS", "Azure"
]

VERSION_PATTERNS = [
    r'v?\d+\.\d+(?:\.\d+)?',  # v1.2.3 or 1.2.3
    r'[A-Z]+\d+',  # M3, A17, H100
]

# Paths
DATA_DIR = "data"
MODEL_DIR = "data/models"
LOG_DIR = "logs"

run_name = "t5_lr_3e-4_bs_8_ep_3"
