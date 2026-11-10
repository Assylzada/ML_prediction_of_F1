# 🏎️ F1 ML Prediction Lab

### Machine Learning-Based Formula 1 Race Prediction & Analytics Platform

<p align="center">
  <strong>Exploring Formula 1 performance through Machine Learning, predictive modeling, and data analytics.</strong>
</p>

---

## 📌 About the Project

**F1 ML Prediction Lab** is a Machine Learning-based web application developed to analyze historical Formula 1 data, identify racing performance patterns, and predict potential race outcomes.

The project combines data preprocessing, exploratory data analysis, feature engineering, machine learning algorithms, model evaluation, and web development into an interactive platform.

The primary goal is to investigate how different Machine Learning approaches can be applied to Formula 1 racing data and evaluate their effectiveness in identifying potential top-performing drivers.

## 🎯 Project Objectives

- Analyze historical Formula 1 racing datasets.
- Perform data cleaning and preprocessing.
- Explore relationships between drivers, constructors, circuits, and race results.
- Develop and compare multiple Machine Learning models.
- Evaluate classification performance using different metrics.
- Experiment with decision threshold optimization.
- Explore ensemble learning and ranking-based approaches.
- Develop an interactive web application for presenting predictions and analytical results.

## 🧠 Machine Learning Models

The project explores multiple supervised and unsupervised Machine Learning techniques.

### Classification Models

- Logistic Regression
- Logistic Regression Tuned
- K-Nearest Neighbors (KNN)
- Support Vector Machine (SVM with RBF kernel)
- Decision Tree Baseline
- Random Forest
- Random Forest (Balanced)
- Extra Trees
- AdaBoost
- Gradient Boosting
- Neural Network (MLP)
- Stacking Ensemble

### Advanced Approaches

- PCA + Logistic Regression
- EDNN-style (Two Networks)
- Pairwise Ranker
- Multilevel Rank-Logit
- Logistic Regression with Isotonic Calibration

### Unsupervised Learning

- K-Means Clustering
- DBSCAN

These approaches allow the project to explore classification, clustering, ensemble learning, dimensionality reduction, ranking, and neural network architectures.

---

## 📊 Model Performance & Evaluation

A major part of the project focuses on comparing different models using classification metrics and race-level ranking performance.

### Evaluation Metrics

| Metric | Description |
|---|---|
| Accuracy | Overall proportion of correct predictions |
| Precision | Proportion of positive predictions that are correct |
| Recall | Proportion of actual positive cases identified |
| F1-Score | Harmonic mean of Precision and Recall |
| ROC-AUC | Ability to distinguish between positive and negative classes |
| PR-AUC | Precision-Recall trade-off, particularly useful for imbalanced datasets |
| Top-3 per race | Race-level performance of identifying drivers within the predicted top three |

### Model Comparison Results

The following table presents the experimental results obtained during model development.

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC | Top-3 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Baseline (Grid ≤ 3) | 0.8947 | 0.6498 | 0.6473 | 0.6485 | 0.9082 | 0.6368 | 0.6512 |
| Logistic Regression | 0.8959 | 0.6330 | 0.7287 | 0.6775 | 0.9365 | **0.7143** | **0.6628** |
| KNN | 0.8895 | 0.6328 | 0.6279 | 0.6304 | 0.9151 | 0.6721 | 0.6318 |
| SVM (RBF) | 0.8912 | 0.6134 | 0.7442 | 0.6725 | 0.8816 | 0.6798 | 0.6395 |
| Neural Network (MLP) | 0.8842 | 0.5908 | 0.7442 | 0.6587 | 0.9295 | 0.6831 | 0.6550 |
| Random Forest | 0.8889 | 0.6070 | 0.7364 | 0.6655 | 0.9343 | 0.7037 | 0.6434 |
| Extra Trees | 0.8930 | 0.6285 | 0.7016 | 0.6630 | 0.9362 | 0.7115 | 0.6357 |
| AdaBoost | 0.8790 | 0.5679 | **0.8101** | 0.6677 | 0.9342 | 0.6977 | 0.6434 |
| Gradient Boosting | 0.8848 | 0.5909 | 0.7558 | 0.6633 | 0.9345 | 0.7054 | 0.6589 |
| Stacking Ensemble | 0.8912 | 0.6164 | 0.7287 | 0.6679 | 0.9360 | 0.7079 | 0.6473 |
| PCA + Logistic Regression | 0.8802 | 0.5788 | 0.7403 | 0.6497 | 0.9247 | 0.6854 | 0.6240 |
| EDNN-style (2 Networks) | 0.1501 | 0.1501 | 1.0000 | 0.2610 | 0.7632 | 0.4676 | 0.5349 |
| Random Forest (Balanced) | 0.8895 | 0.6083 | 0.7403 | 0.6678 | 0.9346 | 0.7058 | 0.6395 |
| Pairwise Ranker | 0.8924 | 0.6299 | 0.6860 | 0.6568 | 0.9344 | 0.6893 | 0.6434 |
| Multilevel Rank-Logit | 0.8842 | 0.5942 | 0.7209 | 0.6515 | 0.9314 | 0.7074 | 0.6395 |
| Logistic Regression Tuned | **0.8982** | 0.6426 | 0.7248 | **0.6812** | 0.9363 | 0.7141 | **0.6628** |
| Logistic Regression Tuned (Isotonic) | 0.8871 | 0.5899 | 0.8140 | — | — | — | — |

*Note: Dashes indicate metrics that were not included in the available experimental results.*

### Decision Threshold Optimization

Different classification thresholds were evaluated to investigate the trade-off between Precision and Recall.

| Model | Decision Threshold |
|---|---:|
| Logistic Regression | 0.34 |
| KNN | 0.41 |
| SVM (RBF) | 0.10 |
| Neural Network (MLP) | 0.31 |
| Random Forest | 0.31 |
| Extra Trees | 0.37 |
| AdaBoost | 0.41 |
| Gradient Boosting | 0.30 |
| Stacking Ensemble | 0.22 |
| PCA + Logistic Regression | 0.29 |
| EDNN-style (2 Networks) | 0.05 |
| Random Forest (Balanced) | 0.52 |
| Pairwise Ranker | 0.20 |
| Multilevel Rank-Logit | 0.31 |
| Logistic Regression Tuned | 0.35 |

The Logistic Regression Tuned (Isotonic) experiment achieved a Recall of 0.8140, although its remaining evaluation metrics are not included in the current results.

---

## 🏆 Best Experimental Results

Based on the available experiments:

| Performance Category | Best Model | Score |
|---|---|---:|
| Highest Accuracy | Logistic Regression Tuned | 0.8982 |
| Highest Precision | Baseline (Grid ≤ 3) | 0.6498 |
| Highest Recall (complete results) | AdaBoost | 0.8101 |
| Highest F1-Score | Logistic Regression Tuned | 0.6812 |
| Highest ROC-AUC | Logistic Regression | 0.9365 |
| Highest PR-AUC | Logistic Regression | 0.7143 |
| Highest Top-3 per race | Logistic Regression / Tuned | 0.6628 |

### Key Findings

1. **Logistic Regression Tuned** achieved the highest overall Accuracy and F1-score, demonstrating a strong balance between Precision and Recall.
2. **Logistic Regression** achieved the highest ROC-AUC and PR-AUC among the models with complete results.
3. **AdaBoost** demonstrated the highest Recall among the models with complete evaluation results, identifying a larger proportion of positive cases.
4. **Stacking Ensemble** and boosting-based models provided competitive results, although they did not outperform the tuned Logistic Regression in F1-score.
5. **EDNN-style (Two Networks)** produced an unusual classification result, with perfect Recall but very low Accuracy and Precision. This suggests that its decision threshold and prediction distribution require further investigation.
6. **Isotonic calibration** demonstrated a higher Recall, but its effect on the other metrics requires additional evaluation.

Overall, Logistic Regression Tuned represents the strongest model according to F1-score, while standard Logistic Regression demonstrates the strongest PR-AUC performance.

---

## 📂 Dataset

The project uses historical Formula 1 datasets containing information about races, drivers, constructors, qualifying sessions, and race results.

### Main Datasets

| Dataset | Description |
|---|---|
| `results.csv` | Race results and driver performance |
| `races.csv` | Race information, circuits, dates, and seasons |
| `qualifying.csv` | Qualifying session results |
| `drivers.csv` | Driver information |
| `constructors.csv` | Constructor information |

### Data Processing Pipeline

1. Loading and inspecting datasets.
2. Handling missing values.
3. Merging datasets using relational identifiers.
4. Feature engineering and selection.
5. Preparing training and testing datasets.
6. Feature scaling where required.
7. Model training and hyperparameter tuning.
8. Performance evaluation and visualization.

---

## 🖥️ Web Application

The project includes an interactive web application developed using Flask, Python, HTML, CSS, and JavaScript.

The website connects the Machine Learning pipeline with a user-friendly interface for presenting prediction results and analytical insights.

### Main Components

- Interactive prediction interface.
- Backend integration with Python ML modules.
- Data processing pipeline.
- Multiple model execution.
- Performance metrics visualization.
- Model comparison and evaluation charts.

### Application Screenshots

Screenshots of the implemented interface and analytical visualizations will be added here.

<!-- Add screenshots here -->

---

## 🛠️ Technologies & Tools

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Web Framework | Flask |
| Data Processing | Pandas, NumPy |
| Machine Learning | Scikit-learn |
| Deep Learning | TensorFlow / Keras |
| Visualization | Plotly, Matplotlib, Seaborn |
| Frontend | HTML, CSS, JavaScript |
| Development Environment | Visual Studio Code |
| Version Control | Git, GitHub |

---

## 📁 Project Structure

```text
F1_ML_Website/
│
├── data/
│   ├── constructors.csv
│   ├── drivers.csv
│   ├── qualifying.csv
│   ├── races.csv
│   └── results.csv
│
├── ml/
│   ├── __init__.py
│   ├── advanced.py
│   ├── common.py
│   └── pipeline.py
│
├── outputs/
│   └── results/
│
├── static/
│   ├── css/
│   │   ├── app.css
│   │   ├── extra.css
│   │   └── style.css
│   │
│   └── js/
│       └── app.js
│
├── templates/
│   └── index.html
│
├── app.py
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation & Setup

### 1. Clone the repository

```bash
git clone https://github.com/Assylzada/ML_prediction_of_F1.git
```

### 2. Navigate to the project directory

```bash
cd ML_prediction_of_F1
```

### 3. Create a virtual environment

```bash
python -m venv venv
```

### 4. Activate the environment

**macOS / Linux**

```bash
source venv/bin/activate
```

**Windows**

```bash
venv\Scripts\activate
```

### 5. Install dependencies

```bash
pip install -r requirements.txt
```

### 6. Run the application

```bash
python app.py
```

### 7. Open in your browser

```text
http://127.0.0.1:5000
```

---

## 🚀 Future Improvements

- Further optimize model performance and generalization.
- Investigate probability calibration.
- Improve race-level ranking performance.
- Experiment with additional ensemble architectures.
- Perform more extensive hyperparameter optimization.
- Expand feature engineering.
- Add automated model testing and validation.
- Improve interactive data visualizations.
- Deploy the application to a production environment.

---

## 👩‍💻 Author

**Assylzada Nurbayeva**

Software Engineering Student | Python Developer | Machine Learning Enthusiast

Interested in Backend Development, Artificial Intelligence, Machine Learning, and building practical software solutions.

**GitHub:** [@Assylzada](https://github.com/Assylzada)

---

<p align="center">
  <strong>Exploring the intersection of Formula 1 and Artificial Intelligence.</strong>
  <br>
  Built with Python, Machine Learning, and curiosity. 🏁
</p>
