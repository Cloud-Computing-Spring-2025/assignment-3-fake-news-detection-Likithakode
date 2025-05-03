# 📰 Fake News Detection using PySpark

This project implements a modular fake news detection pipeline using Apache PySpark. The workflow is divided into five tasks:

- **Task 1**: Data Loading & Preprocessing  
- **Task 2**: Text Cleaning  
- **Task 3**: Feature Extraction  
- **Task 4**: Model Training  
- **Task 5**: Model Evaluation

---

## 🧠 Task 1: Data Loading & Preprocessing

**Objective**  
Load the raw dataset and perform initial preprocessing.

**Input**  
- `fake.csv`

**Output**  
- Cleaned DataFrame with columns: `id`, `title`, `text`, `label`

**Key Steps**  
- Read CSV with header and inferred schema  
- Remove rows with null or empty fields  
- Select relevant columns for downstream tasks  

---

## 🧼 Task 2: Text Cleaning

**Objective**  
Clean and tokenize the text data.

**Input**  
- Output from Task 1

**Output**  
- `task2_output/*.csv` with: `id`, `title`, `filtered_words`

**Key Steps**  
- Convert text to lowercase  
- Remove punctuation and digits using regex  
- Tokenize text  
- Remove stopwords  
- Output cleaned token lists  

---

## 📊 Task 3: Feature Extraction

**Objective**  
Convert cleaned tokens into numerical TF-IDF vectors.

**Input**  
- Output from Task 2

**Output**  
- `task3_output/*.csv` with: `id`, `filtered_words`, `features`, `label_index`

**Key Steps**  
- Use `CountVectorizer` for term frequency  
- Apply `IDF` to weigh terms  
- Encode labels using `StringIndexer`  
- Export features as sparse vectors for modeling  

---

## 🤖 Task 4: Model Training

**Objective**  
Train a logistic regression model using extracted features.

**Input**  
- Parsed features and labels from Task 3  
- `title` from Task 2 (joined via `id`)

**Output**  
- `task4_output/*.csv` with: `id`, `title`, `label_index`, `prediction`

**Key Steps**  
- Convert stringified vectors to `SparseVector` via UDF  
- Split data into training and test sets  
- Train `LogisticRegression` model  
- Predict on test set  
- Join results with `title` and save predictions  

---

## 📈 Task 5: Model Evaluation

**Objective**  
Evaluate the classifier using standard metrics.

**Input**  
- Predictions from Task 4

**Output**  
- `task5_output.csv` or markdown table with evaluation metrics

**Key Steps**  
- Use `MulticlassClassificationEvaluator` to compute:
  - Accuracy  
  - F1 Score  
  - Precision  
  - Recall  

**Sample Output**  
```markdown
| Metric    | Value |
|-----------|-------|
| Accuracy  | 0.89  |
| F1 Score  | 0.88  |
| Precision | 0.87  |
| Recall    | 0.89  |
```

## 💻 Requirements

Python 3.12+

PySpark

Standard Python libraries: re, os, etc.


## Install Requirements

```
pip install pyspark

```


## 🚀 How to Run

```
python Task1/data_loading.py
python Task2/text_cleaning.py
python Task3/feature_extraction.py
python Task4/model_training.py
python Task5/evaluation.py

```
## 📁 Directory Structure

```
.
├── Task1/
│   └── data_loading.py
├── Task2/
│   └── text_cleaning.py
├── Task3/
│   └── feature_extraction.py
├── Task4/
│   └── model_training.py
├── Task5/
│   └── evaluation.py
├── fake.csv
├── README.md
└── requirements.txt

```
