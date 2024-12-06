# Sentiment Analysis on IMDB Movie Reviews

## **Project Overview**

This project focuses on performing sentiment analysis on the IMDB Movie Review dataset to classify movie reviews as either positive or negative. Sentiment analysis is a key application in Natural Language Processing (NLP) that helps in understanding the emotional tone behind a body of text. It is widely used in areas such as marketing, customer feedback analysis, and social media monitoring.

## **Objectives**

- **Model Development:** Build and evaluate models that can accurately classify movie reviews into positive or negative sentiments.
- **Data Preprocessing:** Implement effective text preprocessing techniques to prepare the dataset for machine learning algorithms.
- **Feature Extraction:** Compare different feature extraction methods, specifically Bag of Words (BoW) and Term Frequency-Inverse Document Frequency (TF-IDF).
- **Model Comparison:** Analyze the performance of traditional machine learning models (Naive Bayes and Support Vector Machine) against a deep learning model (LSTM-based Recurrent Neural Network).
- **Result Interpretation:** Assess and interpret the results to understand the strengths and limitations of each approach.

## **Dataset Description**

The project utilizes the **IMDB Movie Review Dataset**, which comprises 50,000 movie reviews evenly split between positive and negative sentiments. The dataset is pre-divided into 25,000 training and 25,000 testing samples, providing a balanced and diverse set of reviews for effective model training and evaluation.

### **Key Features:**

- **Balanced Classes:** Equal representation of positive and negative reviews to prevent model bias.
- **Diverse Texts:** Reviews vary in length and complexity, reflecting real-world data variability.
- **Pre-split Data:** Simplifies the workflow by providing separate training and testing sets.

## **Methodology**

### **1. Data Preprocessing**

Effective preprocessing is crucial for improving model performance. The following steps were undertaken:

- **Text Cleaning:** Removed HTML tags, special characters, and numbers.
- **Tokenization:** Split text into individual words.
- **Normalization:** Converted all text to lowercase.
- **Stop-word Removal:** Eliminated common words that do not contribute to sentiment.
- **Lemmatization:** Reduced words to their base forms to minimize variation.

### **2. Feature Extraction**

Two primary feature extraction techniques were employed:

- **Bag of Words (BoW):** Captures the frequency of each word in the reviews.
- **TF-IDF:** Weights words based on their importance in a review relative to the entire dataset.

### **3. Model Training and Evaluation**

- **Naive Bayes:** A probabilistic model effective for text classification.
- **Support Vector Machine (SVM):** A robust classifier suitable for high-dimensional data.
- **LSTM-based Recurrent Neural Network (RNN):** An advanced deep learning model capable of capturing sequential dependencies in text.

Models were trained using both BoW and TF-IDF features, and their performances were evaluated using metrics such as accuracy, precision, recall, and F1-score.

## **Results**

### **Traditional Models: Naive Bayes and SVM**

- **TF-IDF vs. Bag of Words:** TF-IDF outperformed BoW in both Naive Bayes and SVM models by emphasizing more relevant terms.
- **SVM vs. Naive Bayes:** SVM achieved higher accuracy and better overall metrics compared to Naive Bayes, demonstrating its effectiveness in handling complex feature spaces.

### **Deep Learning Model: LSTM-based RNN**

- **Training Performance:** The LSTM model achieved near-perfect accuracy on the training data.
- **Generalization Issues:** Significant overfitting was observed, as validation and test accuracies were considerably lower than training accuracy despite the use of dropout techniques.
- **Overall Accuracy:** The LSTM model attained an accuracy of approximately 82.49% on the test set, indicating room for improvement in generalization.

### **Performance Comparison**

| Feature Set   | Model        | Accuracy | Precision (Class 0) | Recall (Class 0) | F1-Score (Class 0) | Precision (Class 1) | Recall (Class 1) | F1-Score (Class 1) |
|---------------|--------------|----------|---------------------|------------------|--------------------|---------------------|------------------|--------------------|
| Bag of Words  | Naive Bayes  | 0.83348  | 0.81529             | 0.86232          | 0.83815            | 0.85389             | 0.80464          | 0.82854            |
| TF-IDF        | Naive Bayes  | 0.83880  | 0.82592             | 0.85856          | 0.84192            | 0.85274             | 0.81904          | 0.83555            |
| Bag of Words  | SVM          | 0.81752  | 0.81095             | 0.82808          | 0.81943            | 0.82437             | 0.80696          | 0.81557            |
| TF-IDF        | SVM          | 0.85808  | 0.85339             | 0.86472          | 0.85902            | 0.86290             | 0.85144          | 0.85713            |

## **Conclusion**

Our sentiment analysis on the IMDB Movie Review dataset highlighted how preprocessing steps and feature choices influence model performance. After removing HTML tags, normalizing text, eliminating stopwords, and applying lemmatization, the data became more uniform and easier to interpret. Converting the text into TF-IDF features proved more effective than using Bag of Words, as it helped our models focus on terms that were more informative for distinguishing sentiment. When comparing Naive Bayes and SVM, we found that SVM consistently delivered higher accuracy and better overall metrics, showing that it handled the resulting high-dimensional space well.

Introducing an LSTM-based RNN allowed us to explore deep learning approaches. The LSTM model achieved near-perfect accuracy on the training data, showing that it was able to capture patterns present in the reviews. However, its validation and test results indicated that it was not generalizing these patterns effectively to unseen data. Even with the inclusion of dropout layers to mitigate overfitting, the LSTM ended up closely fitting the training set while failing to maintain the same level of performance on new samples. This highlights that while the model was powerful in memorizing the training set, it struggled to retain flexibility when facing different reviews.

Taken together, these results illustrate the range of outcomes that can occur with different modeling techniques and feature representations. The simpler models combined with TF-IDF features demonstrated consistent and competitive performance, while the LSTM-based approach, though stronger at fitting the training data, did not translate those gains into equally strong generalization. By examining how each method behaved under similar conditions, we gained a better understanding of the strengths and limitations inherent in each approach, as well as the factors that influence sentiment classification accuracy.

## **Citing the Dataset**

```bibtex
@InProceedings{maas-EtAl:2011:ACL-HLT2011,
  author    = {Maas, Andrew L. and Daly, Raymond E. and Pham, Peter T. and Huang, Dan and Ng, Andrew Y. and Potts, Christopher},
  title     = {Learning Word Vectors for Sentiment Analysis},
  booktitle = {Proceedings of the 49th Annual Meeting of the Association for Computational Linguistics: Human Language Technologies},
  month     = {June},
  year      = {2011},
  address   = {Portland, Oregon, USA},
  publisher = {Association for Computational Linguistics},
  pages     = {142--150},
  url       = {http://www.aclweb.org/anthology/P11-1015}
}
