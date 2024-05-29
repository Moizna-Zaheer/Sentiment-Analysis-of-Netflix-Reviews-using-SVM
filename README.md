
# Sentiment Analysis of Netflix Reviews using SVM

## Introduction
Sentiment analysis is a technique used in data mining to understand the sentiment or emotional tone behind a series of words. It is commonly used to gauge public opinion, monitor social media, and improve customer service. This project demonstrates how to perform sentiment analysis on Netflix reviews using a Support Vector Machine (SVM) classifier.

## Overview of the Process

### 1. Loading the Dataset
The first step involves loading a dataset that contains reviews of Netflix content. Each review is accompanied by a score that represents the viewer's rating of the content.

### 2. Data Exploration and Cleaning
#### Checking for Missing Values
Before analyzing the data, it's essential to check for any missing values. Missing data can skew results and lead to incorrect conclusions, so we identify and handle any missing entries.

#### Dropping Missing Values
If there are any missing values in the essential columns, such as the review content or score, those rows are removed from the dataset to ensure the integrity of the analysis.

### 3. Text Preprocessing
Text data often contains noise and inconsistencies that need to be addressed before analysis. Preprocessing involves cleaning the text data by converting all characters to lowercase, which helps standardize the text and improve the accuracy of the analysis.

### 4. Converting Scores to Sentiment Labels
The scores in the dataset are numerical values representing the viewer's rating. For simplicity, these scores are converted into binary sentiment labels:
- Scores of 3 or higher are considered positive (1).
- Scores below 3 are considered negative (0).

### 5. Splitting the Dataset
To evaluate the model's performance, the dataset is split into two parts:
- **Training Set**: Used to train the machine learning model.
- **Testing Set**: Used to evaluate the model's performance on unseen data.

### 6. Feature Extraction with TF-IDF
#### TF-IDF Vectorization
Text data must be converted into a numerical format to be used in machine learning models. TF-IDF (Term Frequency-Inverse Document Frequency) is a technique that transforms text into numerical vectors:
- **Term Frequency (TF)**: Measures how frequently a term appears in a document.
- **Inverse Document Frequency (IDF)**: Measures how important a term is by considering how often it appears across all documents.

TF-IDF balances the frequency of terms with their importance, providing a better representation of the text for analysis.

### 7. Training the SVM Model
A Support Vector Machine (SVM) with a linear kernel is chosen for this task. SVM is a powerful classifier that works well with high-dimensional data. It finds the hyperplane that best separates the classes (positive and negative sentiments) in the feature space.

### 8. Model Evaluation
To assess the model's performance, metrics such as accuracy and a classification report (which includes precision, recall, and F1-score) are used. These metrics provide insights into how well the model is performing in terms of correctly predicting the sentiment of the reviews.

### 9. Saving and Loading the Model
Once the model is trained, it can be saved to a file for later use. This allows the model to be reused without retraining it, saving time and computational resources. The TF-IDF vectorizer is also saved so that new reviews can be processed in the same way as the training data.

### 10. Predicting Sentiment for New Reviews
New reviews can be analyzed by loading the saved model and vectorizer, preprocessing the new text, and using the model to predict the sentiment. This step demonstrates the practical application of the model for real-world use cases.

## Data Mining Techniques Used

### Data Cleaning
Data cleaning is crucial to ensure the quality of the dataset. It involves removing or correcting incomplete, inaccurate, or irrelevant parts of the data. In this project, data cleaning includes checking for and removing missing values.

### Text Preprocessing
Text preprocessing is a form of data transformation specific to text data. It includes:
- Converting text to lowercase to ensure uniformity.
- Removing punctuation, numbers, and stop words (common words that do not carry significant meaning).

### Feature Extraction: TF-IDF
TF-IDF is a feature extraction technique that transforms text data into numerical vectors that machine learning models can process. It is a widely used method in text mining to convert textual information into a format suitable for analysis while maintaining the importance of words.

### Machine Learning: SVM
Support Vector Machines (SVM) are supervised learning models used for classification and regression tasks. SVMs are particularly effective in high-dimensional spaces and are used here for binary classification of sentiment (positive or negative).

### Model Evaluation
Evaluating the model is a critical step in data mining. It involves measuring the model's performance using various metrics to ensure it generalizes well to new, unseen data. In this project, accuracy and classification reports are used for evaluation.

### Model Persistence
Model persistence involves saving the trained model and any associated preprocessing steps (like the TF-IDF vectorizer) to files. This allows the model to be loaded and used in future without needing to retrain it, facilitating real-world application.


The document outlines a comprehensive process for performing sentiment analysis on Netflix reviews using an SVM classifier. It includes data loading, cleaning, preprocessing, feature extraction, model training, evaluation, and prediction. These steps, combined with various data mining techniques, provide a robust framework for analyzing text data and extracting meaningful insights.
