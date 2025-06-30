import pandas as pd
import numpy as np
import random

def generate_sentiment_data(num_samples=1000, filename='sentiment_data.csv'):
    """
    Generates a synthetic dataset for sentiment analysis.

    Args:
        num_samples (int): The number of text samples to generate.
        filename (str): The name of the CSV file to save the data to.
    """
    print(f"--- Generating {num_samples} Synthetic Sentiment Data Samples ---")
    np.random.seed(42) # for reproducibility

    positive_reviews = [
        "This product is amazing! I love it so much. It's truly fantastic and works perfectly.",
        "Excellent service and a fantastic experience. Highly recommend! Very satisfied with everything.",
        "Absolutely thrilled with my purchase. It exceeded my expectations and performs wonderfully.",
        "A truly wonderful day, everything went perfectly. Feeling great and happy.",
        "Feeling incredibly happy and satisfied with the results. Couldn't ask for more.",
        "The best movie I've seen this year, a masterpiece. So engaging and well-made.",
        "Delicious food and a great atmosphere. Will visit again soon, highly recommended."
    ]

    negative_reviews = [
        "This product is terrible. Very disappointed. It broke instantly and is useless.",
        "Awful experience, customer service was unhelpful and rude. I will never return.",
        "Completely dissatisfied with the quality, a waste of money. It fell apart quickly.",
        "What a bad day, nothing went right. Everything was a mess and frustrating.",
        "Feeling very frustrated and unhappy with the outcome. This was a poor decision.",
        "The worst movie, a complete bore. I regret wasting my time watching it.",
        "Unappetizing food and a noisy environment. Never again, a terrible place."
    ]

    neutral_reviews = [
        "The product arrived on time, packaging was standard. It functions as expected.",
        "Customer service answered my question, it was an average interaction. No complaints.",
        "The item is as described, no strong feelings either way. It's neither good nor bad.",
        "Today was just another day. Nothing special happened, just routine tasks.",
        "The process was completed. It was straightforward and without issues.",
        "It's an okay film, nothing special. It didn't impress or disappoint me.",
        "The restaurant serves typical fare. The experience was neutral, not memorable."
    ]

    texts = []
    sentiments = []

    # Generate mixed data
    for _ in range(num_samples):
        choice = np.random.choice(['positive', 'negative', 'neutral'], p=[0.4, 0.3, 0.3]) # 40% pos, 30% neg, 30% neu
        if choice == 'positive':
            # Combine two positive phrases for more variety and length
            text_combo = random.choice(positive_reviews) + " " + random.choice(positive_reviews)
            texts.append(text_combo)
            sentiments.append('positive')
        elif choice == 'negative':
            # Combine two negative phrases
            text_combo = random.choice(negative_reviews) + " " + random.choice(negative_reviews)
            texts.append(text_combo)
            sentiments.append('negative')
        else:
            # Combine two neutral phrases
            text_combo = random.choice(neutral_reviews) + " " + random.choice(neutral_reviews)
            texts.append(text_combo)
            sentiments.append('neutral')

    df = pd.DataFrame({'text': texts, 'sentiment': sentiments})
    df.to_csv(filename, index=False)
    print(f"Synthetic sentiment data saved to '{filename}'")
    print("\nDataset head:\n", df.head())
    print("\nDataset info:\n")
    df.info()
    print("\nDataset description:\n", df.describe())

if __name__ == "__main__":
    generate_sentiment_data()
