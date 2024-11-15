# config.py

import os

# OpenAI API key
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')  # Reads the API key from an environment variable

# Model settings
MODEL_NAME = 'gpt-4'
MAX_INPUT_TOKENS = 4000
MAX_SUMMARY_TOKENS = 150
TEMPERATURE = 0.5
