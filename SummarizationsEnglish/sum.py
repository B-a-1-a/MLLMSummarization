import openai
import pandas as pd
from rouge_score import rouge_scorer


openai.api_key = 'YOUR_OPENAI_API_KEY'

# Function to generate summary using OpenAI API
def generate_summary(text, model="gpt-4o"):
    response = openai.ChatCompletion.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant that summarizes text."},
            {"role": "user", "content": text}
        ],
        max_tokens=100  # Adjust as needed for summary length
    )
    return response['choices'][0]['message']['content'].strip()

# Function to calculate ROUGE scores
def calculate_rouge_scores(reference, generated):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(reference, generated)
    return {
        "ROUGE-1": scores['rouge1'].fmeasure,
        "ROUGE-2": scores['rouge2'].fmeasure,
        "ROUGE-L": scores['rougeL'].fmeasure
    }


df = pd.read_csv('data/your_dataset.csv')  

# Initialize lists to store results
generated_summaries = []
rouge_scores = []

# Process each article in the dataset
for _, row in df.iterrows():
    article = row['article']
    reference_summary = row['highlights']
    
    # Generate summary using OpenAI API
    generated_summary = generate_summary(article)
    generated_summaries.append(generated_summary)
    
    # Calculate ROUGE scores
    scores = calculate_rouge_scores(reference_summary, generated_summary)
    rouge_scores.append(scores)


df['generated_summary'] = generated_summaries
df['rouge_scores'] = rouge_scores

# Display results
print(df[['article', 'highlights', 'generated_summary', 'rouge_scores']].head())
