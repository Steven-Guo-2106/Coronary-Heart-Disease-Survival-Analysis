# Heart Survival Analysis Dashboard

This project uses the Framingham Heart Study dataset to model time-to-event outcomes for cardiovascular disease using survival analysis. The project includes a training notebook that builds the model locally and a Plotly Dash dashboard for visualizing predictions, survival curves, and feature impact.

---

# Important Note

The trained model file is not included in this repository because it is too large for normal GitHub storage.

To use the dashboard, you must first run the notebook locally to generate the model file.

---

# Setup Instructions

## Clone the Repository

```bash
git clone https://github.com/Steven-Guo-2106/Coronary-Heart-Disease-Survival-Analysis.git
cd Coronary-Heart-Disease-Survival-Analysis
```

---

# Running the Project

## 1. Run the Training Notebook

Open the notebook:

```bash
jupyter notebook notebooks/heart_survival_analysis.ipynb
```

Run all notebook cells in order.

The notebook will:

- load and clean the Framingham dataset
- preprocess survival-analysis variables
- train the survival model
- evaluate model performance
- save the trained model locally

The model should be saved to:

```text
model.pkl
```

---

## 3. Run the Dashboard

After the model file has been generated, run:

```bash
python app.py
```

Then open the Dash app in your browser, usually at:

```text
http://127.0.0.1:8050/
```

---

# Model Artifacts

Model files are ignored by Git and must be generated locally.

The following files are excluded from version control:

```gitignore
models/
*.pkl
*.joblib
```

This keeps the repository lightweight and avoids pushing large binary files to GitHub.

---

# Dashboard Features

The dashboard includes:

- survival curve visualization
- predicted cardiovascular disease risk
- patient-level input form
- model summary metrics

---

# Reproducing Results

To reproduce the full project:

1. Install dependencies
2. Run the notebook from start to finish
3. Confirm the model file is created
4. Launch the dashboard
