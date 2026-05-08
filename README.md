# CS301 Machine Learning Project — Loan Prediction App

A full ML pipeline project built for CS301 at NJIT. Covers data preprocessing, model training and optimization, and a deployed interactive web application for real-time loan predictions.

---

## Purpose

The goal is to analyze a real-world loan applicant dataset and answer two questions:

- **Will this loan be approved?** (Classification — predicts `Loan_Status`: Y or N)
- **How much will be loaned?** (Regression — predicts `LoanAmount` in thousands)

The project follows a three-stage structure: preprocessing and initial modeling → model optimization → interactive web app with deployment.

---

## Dataset

**File:** `loan_data_set.csv`  
**Source:** Loan applicant records  
**Size:** 614 rows × 13 columns

| Column | Type | Description |
|--------|------|-------------|
| Loan_ID | Categorical (ID) | Unique identifier — excluded from modeling |
| Gender | Categorical | Male / Female |
| Married | Categorical | Yes / No |
| Dependents | Categorical | 0, 1, 2, 3+ |
| Education | Categorical | Graduate / Not Graduate |
| Self_Employed | Categorical | Yes / No |
| ApplicantIncome | Numerical | Monthly income of applicant |
| CoapplicantIncome | Numerical | Monthly income of co-applicant |
| LoanAmount | Numerical | Loan amount requested (thousands) |
| Loan_Amount_Term | Numerical | Term in months |
| Credit_History | Numerical | 1.0 = good history, 0.0 = bad |
| Property_Area | Categorical | Urban / Semiurban / Rural |
| Loan_Status | Categorical | Y = approved, N = denied (classification target) |

---

## Project Structure

```
CS301-MachineLearningProject/
├── main.ipynb          # Stages 1 & 2 — full ML pipeline notebook
├── app.py              # Stage 3 — Dash web application
├── loan_data_set.csv   # Raw dataset
├── requirements.txt    # Python dependencies
├── Procfile            # AWS Elastic Beanstalk process declaration
└── .ebignore           # Files excluded from EB deployment bundle
```

---

## Stage 1 — Data Preprocessing & Initial Modeling (`main.ipynb`)

### Preprocessing
- Drop `Loan_ID` (unique identifier, no predictive value)
- Remove duplicate rows
- Fill missing values: **mode** for categorical columns, **mean** for numerical columns
- Map `Loan_Status` from Y/N → 1/0 for modeling (original labels kept for visualization)

### Exploratory Data Analysis
- Box plots: numerical features vs `Loan_Status`
- Scatter plots: numerical features vs `LoanAmount`
- Histograms: categorical features grouped by `Loan_Status`
- Correlation heatmap of all numerical features

### Feature Selection
Features chosen for modeling:

| Feature | Reason for inclusion |
|---------|----------------------|
| `ApplicantIncome` | Highest correlation with `LoanAmount` (0.57) |
| `CoapplicantIncome` | Additional financial signal |
| `Credit_History` | Strongest predictor of `Loan_Status` (correlation 0.54) |
| `Married` | Visual pattern — married applicants show higher approval rates |
| `Property_Area` | Visual pattern — Semiurban areas have higher approval rates |

Features excluded: `Loan_Amount_Term`, `Gender`, `Dependents`, `Education`, `Self_Employed` — weak correlation and minimal visual impact.

### Initial Models
- **Classification:** Logistic Regression → Accuracy: 0.7886, F1: 0.8587
- **Regression:** Linear Regression → R²: 0.537, RMSE: 50.21

---

## Stage 2 — Model Optimization (`main.ipynb`, continued)

Two additional classifiers introduced alongside the Logistic Regression baseline:

| Model | Accuracy | F1 Score |
|-------|----------|----------|
| Logistic Regression (Baseline) | 0.7886 | 0.8587 |
| Decision Tree (Basic) | 0.7154 | 0.7904 |
| Random Forest (Basic) | 0.7317 | 0.8070 |
| Decision Tree (Tuned) | 0.7886 | 0.8587 |
| Random Forest (Tuned) | 0.7886 | 0.8587 |

**Tuning method:** `GridSearchCV` with 5-fold cross-validation, scored by F1.

**Best model:** Logistic Regression — same F1 as the tuned tree models, but simpler, faster, and easier to interpret. F1 Score was chosen as the primary metric because the approved class (Y) is more frequent (~69%), making accuracy alone misleading.

---

## Stage 3 — Web Application (`app.py`)

A fully interactive Dash app that generalizes the pipeline to work with any uploaded CSV.

### How It Works

The app walks the user through 5 sequential steps:

**Step 1 — Upload Dataset**  
Drag and drop any `.csv` file. The app parses it and shows row/column counts. ID-like columns (where >50% of values are unique) are automatically excluded from feature selection to prevent one-hot encoding explosions.

**Step 2 — Select Target Column**  
Choose the column to predict from a dropdown. The app auto-detects the task type:
- Object dtype or ≤10 unique values → **Classification** → shows class distribution bar chart
- Otherwise → **Regression** → shows distribution histogram with mean

**Step 3 — Select Features**  
Multi-select dropdown (target and ID-like columns already excluded). Selecting features renders a horizontal **correlation bar chart** showing each feature's absolute correlation with the target.

**Step 4 — Train Model**  
Clicking Train builds and fits a `sklearn.Pipeline` on an 80/20 train/test split:

```
SimpleImputer (mean)        ← fills missing numerical values
StandardScaler              ← normalizes numerical features
SimpleImputer (most_freq)   ← fills missing categorical values
OneHotEncoder               ← encodes categorical features
LogisticRegression / LinearRegression
```

Classification shows Accuracy + F1 Score. Regression shows R² + RMSE.

**Step 5 — Predict**  
A dynamic input form is generated — one field per selected feature. Numerical features show the column median as a placeholder. Categorical features show a dropdown of all unique values. Clicking Predict returns:
- Classification: predicted class + probability bar chart per class
- Regression: predicted numeric value

### Pipeline Design Notes
- `StandardScaler` is applied only to numerical features — tree models don't need it but it's critical for Logistic Regression convergence
- `OneHotEncoder(handle_unknown="ignore")` — safely handles values at prediction time that weren't in the training set
- The trained pipeline is serialized with `pickle` + `base64` and stored in a `dcc.Store` component, keeping the app stateless between callbacks
- Changing the target or features automatically resets and hides downstream sections to prevent stale predictions

---

## Running Locally

### Install dependencies
```bash
pip install -r requirements.txt
```

### Run the notebook (Stages 1 & 2)
```bash
jupyter notebook main.ipynb
```
Run all cells top-to-bottom — cells are stateful and Stage 2 reuses variables from Stage 1.

### Run the web app (Stage 3)
```bash
python app.py
```
Opens at `http://127.0.0.1:8050`

---

## Testing the App

### Golden path (classification)
1. Upload `loan_data_set.csv`
2. Select target: `Loan_Status` → verify **Classification** badge + class distribution chart appears
3. Select features: `ApplicantIncome`, `CoapplicantIncome`, `Credit_History`, `Married`, `Property_Area`
4. Verify correlation chart shows `Credit_History` with the strongest bar
5. Click **Train Model** → expect ~0.79 Accuracy, ~0.77 F1
6. Fill prediction form:
   - ApplicantIncome: `5000`, CoapplicantIncome: `2000`, Credit_History: `1.0`
   - Married: `Yes`, Property_Area: `Semiurban`
7. Click **Predict** → expect **Y** (Approved) with high confidence

### Edge cases to verify
| Scenario | Expected behavior |
|----------|-------------------|
| `Credit_History: 0.0`, all else same | Flips prediction to **N** (Denied) |
| Select `LoanAmount` as target | Detects **Regression**, shows histogram |
| Upload a non-CSV file | Shows red error message, does not crash |
| Change target after training | Train and Predict sections reset automatically |
| Change features after training | Train and Predict sections reset automatically |

### Regression path
1. Upload `loan_data_set.csv`
2. Select target: `LoanAmount` → verify **Regression** badge + histogram with mean
3. Select features: `ApplicantIncome`, `CoapplicantIncome`, `Married`, `Property_Area`
4. Train → expect R² ~0.35–0.55, RMSE ~50–60
5. Fill inputs and predict → returns a numeric loan amount

---

## Deployment (Render)

1. Push this repo to GitHub
2. Go to [render.com](https://render.com) and sign up (no credit card required)
3. New → **Web Service** → connect your GitHub repo
4. Configure:
   - **Runtime:** Python
   - **Build command:** `pip install -r requirements.txt`
   - **Start command:** `gunicorn app:server`
5. Click **Deploy**

Render assigns a permanent public HTTPS URL (e.g. `https://cs301-loan-app.onrender.com`) that can be included in the project report.

The app is served by **Gunicorn** via the `Procfile`:
```
web: gunicorn app:server
```

`app:server` points to the Flask server instance exposed by `server = app.server` in `app.py`.

---

## Dependencies

| Package | Purpose |
|---------|---------|
| `dash` | Web framework and reactive UI components |
| `plotly` | Interactive charts |
| `pandas` | Data loading and manipulation |
| `scikit-learn` | ML Pipeline, models, and metrics |
| `numpy` | Numerical operations |
| `gunicorn` | Production WSGI server for deployment |
