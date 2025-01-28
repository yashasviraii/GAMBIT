from utils.generate_question import generate_question
from utils.generate_answer import generate_response
from utils.detection import detect_ai_text

import numpy as np
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
from scipy.spatial.distance import mahalanobis
from sklearn.covariance import EmpiricalCovariance
from scipy.stats import multivariate_normal
import nltk

import os
import pandas as pd
nltk.download('punkt')

def collect_initial_data(
    model1, tokenizer1, model2, tokenizer2, detection_model, detection_tokenizer, 
    num_samples=250, csv_path="initial_data.csv"
):
    """
    Collect initial dataset of questions and answers with classification, saving to CSV if not already saved.
    """
    # Check if the CSV file already exists
    if os.path.exists(csv_path):
        print(f"CSV file '{csv_path}' already exists. Loading data from file...")
        return pd.read_csv(csv_path)
    
    print("CSV file not found. Collecting initial training data...")
    initial_data = []

    for i in range(num_samples):
        if i % 10 == 0:
            print(f"Collecting sample {i+1}/{num_samples}")

        # Generate question
        question = generate_question(model1, tokenizer1, "Ask an interesting question:")
        print("Question asked for collecting data:", question)
        prompt = "Q: " + question + "\nA:"

        # Generate response
        responses = generate_response(model2, tokenizer2, prompt, num_variants=1)
        for response in responses:
            # Get detection confidence
            print("Response:", response)
            detection_result = detect_ai_text(detection_model, detection_tokenizer, response)
            confidence = detection_result['confidence']

            # Classify based on confidence threshold
            is_real = confidence > 0.95

            # Add to the dataset
            initial_data.append({
                'question': question,
                'answer': response,
                'confidence': confidence,
                'is_real': is_real
            })

    # Convert to DataFrame
    df = pd.DataFrame(initial_data)

    # Save the collected data to CSV
    print(f"Saving data to '{csv_path}'...")
    df.to_csv(csv_path, index=False)

    return df


import numpy as np
from gensim.models import Word2Vec
from scipy.spatial.distance import mahalanobis
from sklearn.covariance import EmpiricalCovariance
from scipy.stats import multivariate_normal
import torch
from transformers import BertModel, BertTokenizer
from nltk.tokenize import word_tokenize

# Initialize BERT model and tokenizer globally
bert_model = BertModel.from_pretrained('bert-base-uncased')
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def setup_embedding_model(questions):
    """Setup Word2Vec model for question embeddings"""
    # Tokenize questions
    tokenized_questions = [word_tokenize(question.lower()) for question in questions]
    # Train Word2Vec model using tokenized data
    model = Word2Vec(sentences=tokenized_questions, vector_size=100, window=5, min_count=1, workers=4)
    return model

def get_question_embedding(question, word2vec_model):
    """Convert question to embedding vector using BERT and Word2Vec combined."""
    # Get BERT embeddings
    inputs = bert_tokenizer(question, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = bert_model(**inputs)
    
    # Get [CLS] token embedding for sentence representation
    bert_embedding = outputs.last_hidden_state[:, 0, :].squeeze().numpy()
    
    # Tokenize and retrieve Word2Vec embeddings
    word_tokens = [word for word in word_tokenize(question.lower()) if word in word2vec_model.wv]
    word_vectors = [word2vec_model.wv[word] for word in word_tokens]
    
    # Combine BERT and Word2Vec embeddings
    # if len(word_vectors) > 0:
    if 0==1:
        word2vec_embedding = np.mean(word_vectors, axis=0)
        combined_embedding = np.concatenate((bert_embedding, word2vec_embedding))
    else:
        combined_embedding = bert_embedding
    
    return combined_embedding

def calculate_distribution_metrics(real_questions, word2vec_model):
    """Calculate distribution metrics for real questions"""
    # Get embeddings for real questions
    real_embeddings = np.array([get_question_embedding(q, word2vec_model) for q in real_questions])

    # Calculate mean
    real_mean = np.mean(real_embeddings, axis=0)

    # Calculate covariance with regularization
    cov_estimator = EmpiricalCovariance(assume_centered=False)
    cov_estimator.fit(real_embeddings)
    real_cov = cov_estimator.covariance_

    # Add regularization
    eps = 1e-6
    k = real_cov.shape[0]
    real_cov += eps * np.eye(k)
    real_cov = (real_cov + real_cov.T) / 2

    try:
        real_dist = multivariate_normal(mean=real_mean, cov=real_cov, allow_singular=True)
    except np.linalg.LinAlgError:
        real_cov = np.diag(np.diag(real_cov))
        real_dist = multivariate_normal(mean=real_mean, cov=real_cov, allow_singular=True)

    return {
        'mean': real_mean,
        'covariance': real_cov,
        'distribution': real_dist,
        'embeddings': real_embeddings
    }

def calculate_vector_reward(generated_question, real_metrics, word2vec_model):
    """Calculate reward score based on vector distances"""
    # Get embedding for generated question
    generated_embedding = get_question_embedding(generated_question, word2vec_model)
    generated_embedding = generated_embedding.reshape(1, -1)
    
    # Calculate distances
    euclidean_distance = np.abs(np.sqrt(np.sum((generated_embedding - real_metrics['mean']) ** 2)))

    try:
        inv_cov = np.linalg.inv(real_metrics['covariance'])
        mahalanobis_distance = abs(mahalanobis(
            generated_embedding.flatten(),
            real_metrics['mean'],
            inv_cov
        ))
    except np.linalg.LinAlgError:
        mahalanobis_distance = euclidean_distance

    log_prob = abs(-real_metrics['distribution'].logpdf(generated_embedding))

    # Normalize distances to [0, 1] range using reference values
    max_euclidean = 10.0  # Set based on your data distribution
    max_mahalanobis = 20.0  # Set based on your data distribution
    max_log_prob = 50.0  # Set based on your data distribution

    norm_euclidean = 1 - min(euclidean_distance / max_euclidean, 1)
    norm_mahalanobis = 1 - min(mahalanobis_distance / max_mahalanobis, 1)
    norm_log_prob = 1 - min(log_prob / max_log_prob, 1)

    # Combine metrics into final reward (higher is better)
    reward = (0.4 * norm_euclidean +
             0.3 * norm_mahalanobis +
             0.3 * norm_log_prob)
    print(generated_question,"----->", reward)
    return float(reward)
