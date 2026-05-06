# Fake Job Posting Detection

## Introduction

This project aims to build and evaluate machine learning models for detecting fraudulent job postings. By leveraging natural language processing (NLP) techniques, the goal is to classify job advertisements as either legitimate or fake based on their textual content, including job titles, company profiles, descriptions, and requirements.

## Dataset

The dataset used for this project is sourced from Hugging Face: [`victor/real-or-fake-fake-jobposting-prediction`](https://huggingface.co/datasets/victor/real-or-fake-fake-jobposting-prediction). It contains a collection of job postings, each labeled as `0` for real and `1` for fraudulent.

### Dataset Statistics:
*   **Total Job Postings:** 17,880
*   **Real Jobs (0):** 17,014 (95.2%)
*   **Fake Jobs (1):** 866 (4.8%)

This shows a significant class imbalance, which is an important consideration for model evaluation.

## Methodology

The project follows a standard machine learning workflow:

1.  **Data Loading:** The dataset was loaded using the `datasets` library and converted to a Pandas DataFrame.
2.  **Text Preprocessing:**
    *   Combined job `title`, `company_profile`, `description`, and `requirements` into a single `combined_text` column.
    *   Cleaned the `combined_text` by converting to lowercase, removing URLs, email addresses, numbers, and special characters.
3.  **Feature Engineering:**
    *   **TF-IDF Vectorization:** Used `TfidfVectorizer` to convert text into numerical features, considering unigrams and bigrams (`ngram_range=(1, 2)`), limiting to `max_features=500`, and removing common English stop words.
    *   **Count Vectorization:** Used `CountVectorizer` for comparison, with similar parameters to TF-IDF.
4.  **Data Splitting:** The data was split into training (80%) and testing (20%) sets using `train_test_split`, ensuring stratification to maintain the original class distribution in both sets.
5.  **Model Training:** Four different classification models were trained:
    *   Logistic Regression with TF-IDF features.
    *   Logistic Regression with Count features.
    *   Multinomial Naive Bayes with TF-IDF features.
    *   Multinomial Naive Bayes with Count features.
6.  **Model Evaluation:** Models were evaluated using Accuracy, Precision, Recall, and F1-Score. F1-Score was prioritized due to the class imbalance.
7.  **Feature Importance:** For the best-performing Logistic Regression model, the coefficients were analyzed to identify words most indicative of fraudulent and real job postings.

## Results and Best Model

The models were compared based on their performance metrics. The **Logistic Regression with Count Vectorizer** emerged as the best model, primarily due to its F1-Score which balances precision and recall for the minority 'fake' class effectively.

| Model                  | Accuracy | Precision | Recall   | F1       |
| :--------------------- | :------- | :-------- | :------- | :------- |
| Model 1: LR + TF-IDF   | 0.9650   | 0.9444    | 0.2948   | 0.4493   |
| **Model 2: LR + Count**| **0.9667**| **0.7077**| **0.5318**| **0.6073**|
| Model 3: NB + TF-IDF   | 0.9550   | 1.0000    | 0.0694   | 0.1297   |
| Model 4: NB + Count    | 0.8624   | 0.2207    | 0.7283   | 0.3387   |

**Best Model:** Model 2: Logistic Regression + Count
*   **Accuracy:** 0.9667
*   **Precision:** 0.7077
*   **Recall:** 0.5318
*   **F1-Score:** 0.6073

## Fraud Indicators

The feature importance analysis from the best model (Logistic Regression + Count) highlighted words that strongly indicate either fraud or real jobs. Some top indicators for fraud included words like 'enjoy', 'bring', 'word', 'team members', 'recruiting', 'city', 'achieve', 'exciting', 'increase', 'grow', 'free', 'email', 'internet', 'seeking', 'impact'.

## Limitations

*   **Class Imbalance:** The heavily imbalanced dataset (4.8% fake jobs) makes it challenging to achieve high recall without significantly impacting precision.
*   **Evolving Tactics:** Fraudulent job postings constantly evolve, requiring continuous model retraining and adaptation.
*   **Text-only Analysis:** The model relies solely on text, potentially missing other non-textual cues of fraud.
*   **Generalizability:** Performance may vary on different platforms or for job postings in other languages.

## Future Work

*   Implement advanced techniques to handle class imbalance, such as SMOTE or weighted loss functions.
*   Explore more sophisticated NLP models (e.g., Transformer-based models like BERT).
*   Integrate additional features beyond text, such as sender reputation or domain analysis.
*   Develop a user-friendly application or API for real-time fraud detection.

## How to Run This Project

1.  **Clone the repository:**
    ```bash
git clone https://github.com/Aman3Ojha/fake-job-posting-detection.git
cd fake-job-posting-detection
    ```
2.  **Open in Google Colab:** Upload the `fake_job_prediction.ipynb` file to Google Colab, or open it directly via a GitHub link if available.
3.  **Install Dependencies:** Ensure all necessary libraries are installed by running the first cell in the notebook:
    ```python
!pip install -q datasets pandas numpy scikit-learn matplotlib seaborn
    ```
4.  **Execute Cells:** Run all cells sequentially in the Colab notebook.

## Files in This Repository

*   `fake_job_prediction.ipynb`: The Jupyter notebook containing all the code for data loading, preprocessing, model training, evaluation, and analysis.
*   `01_class_distribution.png`: Bar chart showing the distribution of real vs. fake jobs.
*   `02_train_test_split.png`: Bar charts showing class distribution in training and testing sets.
*   `03_model_comparison.csv`: CSV file detailing the performance metrics of all trained models.
*   `04_model_comparison_chart.png`: Visual comparison of model performance metrics.
*   `05_confusion_matrices.png`: Confusion matrices for all models.
*   `06_feature_importance.png`: Bar charts illustrating words indicative of fraud and real jobs.
*   `model_lr_tfidf.pkl`, `model_lr_count.pkl`, `model_nb_tfidf.pkl`, `model_nb_count.pkl`: Saved trained model files.
*   `vectorizer_tfidf.pkl`, `vectorizer_count.pkl`: Saved vectorizer objects.
*   `BEST_MODEL.pkl`, `BEST_VECTORIZER.pkl`: The best-performing model and its corresponding vectorizer.

## Author

**Aman Ojha**
*   [GitHub Profile](https://github.com/Aman3Ojha)
