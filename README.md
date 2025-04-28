
Predictive Maintenance using Machine Learning

 Overview
This project builds a **Predictive Maintenance System** using machine learning models (LightGBM, Random Forest) to predict machine failures based on real-world sensor data.  
It provides:
- A **Flask API** backend for prediction
- A **Streamlit dashboard** frontend for visualization and user interaction

The system is fully integrated for real-time prediction and monitoring.

---

## Tech Stack
- Python 3.8+
- Flask (Backend API)
- Streamlit (Frontend Dashboard)
- LightGBM (ML Model)
- Random Forest (ML Model)
- SMOTE (Data Balancing Technique)
- Scikit-learn
- Pandas, NumPy
- Matplotlib, Seaborn

---

## Project Structure
```
predictive-maintenance/
├── app/
│   ├── api/
│   │   └── api.py              # Flask API for model inference
│   ├── dashboard/
│   │   └── dashboard.py         # Streamlit dashboard
│   ├── model/
│   │   └── lightgbm_best_model.pkl  # Trained ML model
│   ├── training/
│   │   └── PMM.py               # Model training code
│   └── utils/
│       └── predict.py           # Utility functions for predictions
├── data/
│   └── predictive_maintenance.csv   # Dataset (if included)
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Features
- Machine Learning pipeline for predictive maintenance
- SMOTE applied to balance the dataset
- Hyperparameter tuning using GridSearchCV
- Model performance evaluation with accuracy, precision, recall, and F1-score
- Flask API to serve real-time predictions
- Streamlit dashboard for easy interaction
- Visualization of input metrics and prediction results

---

## Installation Instructions

1. Clone the repository:
   ```
   git clone https://github.com/YourUsername/predictive-maintenance.git
   cd predictive-maintenance
   ```

2. Create and activate a virtual environment (optional but recommended):
   ```
   python -m venv venv
   venv\Scripts\activate     # On Windows
   source venv/bin/activate  # On Mac/Linux
   ```

3. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Run the backend (Flask API):
   ```
   cd app/api
   python api.py
   ```
   This will start the server at `http://127.0.0.1:5000`.

5. Run the frontend (Streamlit dashboard):
   Open a new terminal and execute:
   ```
   cd app/dashboard
   streamlit run dashboard.py
   ```
   This will open the dashboard in your browser.

---

## How It Works

- **Training**:  
  The model is trained using LightGBM and Random Forest on a resampled dataset (SMOTE applied).  
  After training, the best-performing model is saved as a `.pkl` file.

- **Serving**:  
  Flask API loads the trained model and exposes a `/predict` endpoint for real-time inference.

- **Frontend**:  
  The Streamlit dashboard sends feature data to the Flask API and displays the prediction results interactively.

---

## Future Improvements
- Move to a FastAPI + React.js architecture for production-grade scalability
- Implement model monitoring and automatic retraining
- Add database integration for storing prediction history
- Dockerize the application for easier deployment
