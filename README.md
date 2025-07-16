# 🎙️ Parkinson's Voice Predictor

A machine learning-powered voice-based screening tool to detect the presence of Parkinson’s Disease using vocal biomarkers.

---

## 📊 Project Overview

This project aims to detect Parkinson’s Disease from voice samples using classical machine learning and signal processing. A trained model, based on the Oxford Parkinson's Disease Detection Dataset, evaluates the user's vocal patterns and predicts the likelihood of Parkinson's.

---

## 📁 Dataset

- **Source**: Oxford Parkinson’s Disease Detection Dataset (UCI)
- **Rows**: 195 samples
- **Columns**: 24 total (22 biomedical voice features, 1 target label, 1 identifier)
- **Target Variable**: `status` (0 = healthy, 1 = Parkinson’s)
- Attached some Voice Recordings Dataset for testing purpose.

---

## 🛠️ Model Training (`parkinsons_model.ipynb`)

### 🔹 Libraries Used

- `scikit-learn`
- `xgboost`
- `imbalanced-learn`
- `matplotlib`, `seaborn`, `pandas`, `numpy`

### 🔹 Preprocessing Pipeline

1. **Feature Selection**: Dropped `name`, retained voice-based features
2. **PowerTransformer**: Normalize feature distributions
3. **MinMaxScaler**: Scaled features to (-1, 1)
4. **SMOTETomek**: Handled class imbalance (oversampling + undersampling)
5. **Stratified Train-Test Split**: 70% training, 30% testing

### 🔹 Model: `XGBClassifier`

- Integrated into a `Pipeline`
- Hyperparameter tuning using `GridSearchCV` with 5-fold `StratifiedKFold`
- Tuned Params:
  - `max_depth`: [3, 5, 7, 11]
  - `learning_rate`: [0.01, 0.1, 0.2]
  - `n_estimators`: [50, 100]
  - `subsample`: [0.8, 1]

### ✅ Best Cross-Validation Score (ROC AUC)

- 0.9121 (calculated via 5-fold Stratified Cross-Validation on the training set)

### 🔧 Best Hyperparameters (GridSearchCV)

- `learning_rate`: 0.2  
- `max_depth`: 5  
- `n_estimators`: 100  
- `subsample`: 0.8

### 📈 Final Evaluation Metrics (on Test Set)

| Metric     | Class 0 (Healthy) | Class 1 (Parkinson's) |
|------------|------------------|------------------------|
| Precision  | 0.92             | 0.93                   |
| Recall     | 0.80             | 0.98                   |
| F1-score   | 0.86             | 0.96                   |
| Accuracy   | 93%              |                        |
| ROC AUC    | ~0.94            |                        |

---

## 🎙️ Prediction Interface (`main.py`)

### 🔹 Built With:
- `customtkinter` (Dark-themed polished GUI)
- `parselmouth` (Praat integration)
- `sounddevice` (for audio capture)

### 🔹 Pipeline Flow:

1. User clicks **“Record Voice”** to capture a `.wav` file (5 seconds).
2. Features are extracted using **Praat Parselmouth**, matching the 22 features used during training.
3. Any feature not extracted is filled with a **default (neutral) value** (like 0.0).
4. The trained pipeline from `parkinsons_model.pkl` is loaded using `joblib`.
5. Extracted features are passed through the model for prediction.
6. The GUI displays whether **Parkinson’s is likely present or not**, along with a **confidence score**.

---

## 🧪 Voice Sample Testing

You can test `.wav` files captured in similar conditions by using the **GUI**. The model expects clean, voiced speech.

---

## 📦 Installation & Setup

### 🔹 Requirements

All required packages are listed in [`requirements.txt`]. Key ones include:
- `scikit-learn==1.6.1`
- `xgboost==2.1.4`
- `praat-parselmouth==0.4.3`
- `customtkinter==5.2.1`

## 🔹 Steps

- Clone the repo or download the ZIP
cd ParkinsonsVoicePredictor

- (Optional) Create a virtual environment
python -m venv venv
venv\Scripts\activate    # On Windows

- Install dependencies
pip install -r requirements.txt

- Run the GUI application
python main.py

## 👩‍💻 Developed By

**Shivam Verma**  
🎓 Final Year Student | 💻 Aspiring Data Scientist | Tech Enthusiast  
🌐 [LinkedIn](https://www.linkedin.com/in/shivam-verma-6267a0203) 

---

## 📌 Notes

- This is a research/educational project. Not a medical diagnostic tool.
- Trained using balanced data (SMOTETomek), so model generalizes well to imbalanced real-world samples.
- The `.pkl` model must be loaded with the **same library versions** used during training.

---

## 📸 Screenshots

> <img width="596" height="515" alt="Screenshot 2025-07-17 002807" src="https://github.com/user-attachments/assets/2a080ac7-2e00-46cf-8a8b-1a034c815cf5" />


---

## ⭐ Star the Repo

If you found this helpful, give the project a ⭐ on GitHub and share it with friends!
