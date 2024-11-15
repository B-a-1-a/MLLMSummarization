import os
import json
import openai
from datasets import load_dataset
from tqdm import tqdm
import evaluate
import config  # Import config.py for settings

# Set OpenAI API key from config
openai.api_key = config.OPENAI_API_KEY
print("API Key Loaded:", bool(openai.api_key))  # Confirm API key is loaded

# Configuration settings from config.py
MODEL_NAME = config.MODEL_NAME
MAX_INPUT_TOKENS = config.MAX_INPUT_TOKENS
MAX_SUMMARY_TOKENS = config.MAX_SUMMARY_TOKENS
TEMPERATURE = config.TEMPERATURE

def generate_summary(text):
    """Generate summary using OpenAI's API."""
    # Truncate text to fit token limit
    text_tokens = text.split()
    if len(text_tokens) > MAX_INPUT_TOKENS:
        text = ' '.join(text_tokens[:MAX_INPUT_TOKENS])

    try:
        response = openai.ChatCompletion.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "You are an expert summarizer."},
                {"role": "user", "content": f"Summarize the following text:\n\n{text}"}
            ],
            max_tokens=MAX_SUMMARY_TOKENS,
            temperature=TEMPERATURE,
        )
        summary = response['choices'][0]['message']['content'].strip()
        return summary
    except Exception as e:
        print(f"Error during API call: {e}")
        return ""

def load_cache(cache_file):
    """Load cache from JSON file."""
    if os.path.exists(cache_file):
        with open(cache_file, 'r') as f:
            return json.load(f)
    return {}

def save_cache(cache_file, cache_data):
    """Save cache to JSON file."""
    with open(cache_file, 'w') as f:
        json.dump(cache_data, f)

def main(sample_size=10):
    # Load dataset
    print("Loading dataset...")
    dataset = load_dataset('cnn_dailymail', '3.0.0')
    test_dataset = dataset['test'].select(range(sample_size))  # Use sample size

    articles = test_dataset['article']
    references = test_dataset['highlights']

    # Load or initialize cache
    cache_file = 'summaries_cache.json'
    summaries_cache = load_cache(cache_file)

    generated_summaries = []

    print("Generating summaries...")
    for article in tqdm(articles, desc="Processing articles"):
        if article in summaries_cache:
            summary = summaries_cache[article]
        else:
            summary = generate_summary(article)
            summaries_cache[article] = summary
            save_cache(cache_file, summaries_cache)  # Save after each generation

        generated_summaries.append(summary)

    # Evaluate summaries
    print("Evaluating summaries...")
    rouge = evaluate.load('rouge')
    results = rouge.compute(predictions=generated_summaries, references=references)

    print("\nROUGE Scores:")
    for key, value in results.items():
        print(f"{key}: {value:.2f}")

    # Display summaries
    for i in range(len(generated_summaries)):
        print(f"\nArticle {i+1}:")
        print(articles[i][:500] + '...')  # Print first 500 characters
        print("\nGenerated Summary:")
        print(generated_summaries[i])
        print("\nReference Summary:")
        print(references[i])
        print("\n" + "="*80)

if __name__ == "__main__":
    main(sample_size=3)  # Adjust sample size as needed
