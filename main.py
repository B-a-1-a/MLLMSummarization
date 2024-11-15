import os
import json
import openai
from datasets import load_dataset
from tqdm import tqdm
import evaluate

# Use environment variable for OpenAI API key
openai.api_key = os.getenv('OPENAI_API_KEY')
print("API Key Loaded:", openai.api_key)  # Confirm API key is loaded

# Set configuration settings with fallback defaults
MODEL_NAME = 'gpt-4'
MAX_INPUT_TOKENS = 4000
MAX_SUMMARY_TOKENS = 50  # Reduce token limit for testing
TEMPERATURE = 0.3  # Lower temperature for more consistent outputs

def generate_summary(text):
    # Truncate text if necessary
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
        print("API Response:", response)  # Debug statement
        summary = response['choices'][0]['message']['content'].strip()
        return summary
    except Exception as e:
        print(f"Error during API call: {e}")
        return ""



def test_generate_summary():
    # Test the generate_summary function with a sample input
    test_text = "OpenAI creates cutting-edge AI tools for research and development."
    print("Test summary:", generate_summary(test_text))

def main():
    # Test the summary generation function with a sample input
    test_generate_summary()

    # Load dataset
    print("Loading dataset...")
    dataset = load_dataset('cnn_dailymail', '3.0.0')
    test_dataset = dataset['test'].select(range(10))  # Modify the range as needed

    articles = test_dataset['article']
    references = test_dataset['highlights']

    # Initialize cache
    cache_file = 'summaries_cache.json'
    if os.path.exists(cache_file):
        with open(cache_file, 'r') as f:
            summaries_cache = json.load(f)
    else:
        summaries_cache = {}

    generated_summaries = []

    print("Generating summaries...")
    for article in tqdm(articles, desc="Processing articles"):
        if article in summaries_cache:
            summary = summaries_cache[article]
        else:
            try:
                summary = generate_summary(article)
                summaries_cache[article] = summary

                # Save to cache file
                with open(cache_file, 'w') as f:
                    json.dump(summaries_cache, f)
            except Exception as e:
                print(f"Error generating summary: {e}")
                summary = ""

        print(f"Generated summary for article: {summary}")  # Debug statement for each generated summary
        generated_summaries.append(summary)

    # Evaluate summaries
    print("Evaluating summaries...")
    rouge = evaluate.load('rouge')
    results = rouge.compute(predictions=generated_summaries, references=references)

    print("\nROUGE Scores:")
    for key in results:
        print(f"{key}: {results[key]:.2f}")

    # Optional: Display summaries
    for i in range(len(generated_summaries)):
        print(f"\nArticle {i+1}:")
        print(articles[i][:500] + '...')  # Print the first 500 characters
        print("\nGenerated Summary:")
        print(generated_summaries[i])
        print("\nReference Summary:")
        print(references[i])
        print("\n" + "="*80)

if __name__ == "__main__":
    main()
