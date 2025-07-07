# Sentiment Analysis with NLP

**Internship Details:**
* **Company:** CODTECH IT SOLUTIONS
* **Name:** Jayasri N
* **Intern ID:** CT04DG1603
* **Domain:** Data Analytics
* **Duration:** 4 Weeks
* **Mentor:** Neela Santosh Kumar

This project demonstrates a complete pipeline for sentiment analysis on textual data using Natural Language Processing (NLP) techniques. It covers data generation, preprocessing, model training, evaluation, and making predictions to classify text into sentiments (positive, negative, neutral).

## Features

* **Synthetic Data Generation**: Creates a dataset of simulated reviews/tweets with pre-assigned sentiments.
* **Text Preprocessing (Simplified)**: Includes tokenization, lowercasing, and removing punctuation/numbers (no NLTK).
* **Feature Extraction**: Converts text into numerical features using TF-IDF.
* **Machine Learning Model**: Trains a classification model (e.g., Naive Bayes) to predict sentiment.
* **Model Evaluation**: Assesses performance using accuracy, precision, recall, and F1-score.
* **Prediction**: Demonstrates using the trained model to predict sentiment on new text.
* **Output Reports**: Generates text files showing sample predictions and a detailed model performance report.

## Getting Started

Follow these steps to set up and run the project locally.

### Prerequisites

* Python 3.8+
* `pip` (Python package installer)

### Installation

1.  **Clone the repository (or download the files):**
    ```bash
    https://github.com/YOUR_USERNAME/sentiment-analysis-nlp.git
    ```
    *(Note: Replace `YOUR_USERNAME` with your actual GitHub username and adjust the repo name if it's different)*

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    ```

3.  **Activate the virtual environment:**
    * **On Windows:**
        ```bash
        .\venv\Scripts\activate
        ```
    * **On macOS/Linux:**
        ```bash
        source venv/bin/activate
        ```

4.  **Install the required Python packages:**
    ```bash
    pip install -r requirements.txt
    ```

### How to Run

1.  **Generate the dataset:**
    This script creates `sentiment_data.csv`.
    ```bash
    python data_generator.py
    ```

2.  **Perform Sentiment Analysis:**
    This script loads data, preprocesses, trains, evaluates, saves the model (`sentiment_model.pkl`), and generates `sample_predictions.txt` and `model_report.txt`.
    ```bash
    python sentiment_analyzer.py
    ```

## Project Structure
```
sentiment-analysis-nlp/
├── README.md
├── requirements.txt
├── data_generator.py          
├── sentiment_analyzer.py
├── sample_predictions.txt
```
## Dependencies

All dependencies are listed in `requirements.txt`.

## Future Enhancements

* **Real Dataset**: Integrate with a real-world dataset.
* **Advanced Models**: Experiment with deep learning models (e.g., LSTMs, Transformers like BERT).
* **Hyperparameter Tuning**: Optimize model performance.
* **Deployment**: Build a simple web application for predictions.
* **Visualization**: Add plots for sentiment distribution, word clouds, or model performance.
* **Aspect-Based Sentiment Analysis**: Analyze sentiment towards specific entities or aspects.

        

