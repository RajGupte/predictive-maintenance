# Predictive Maintenance Failure Prediction 

This project predicts machine failure using ML (LightGBM with SMOTE balancing).

## Setup

```bash
pip install -r requirements.txt

#Run API 
python app/api.py

#Run Streamlit Dashboard 
streamlit run app/dashboard.py


---

#  Run Your API Locally 


```bash
cd app
python api.py

# POST sample data using cURL:

curl --header "Content-Type: application/json" \
  --request POST \
  --data '{"features": [0.5, 0.5, 0.1, 0.3, 0.2, 0.6, 0.0, 0.0, 0.1]}' \
  http://127.0.0.1:5000/predict_failure

Step | Command
Install Libraries | pip install -r requirements.txt
Train Model (optional) | python app/training/PMM.py
Run API Server | set PYTHONPATH=.  python app/api/api.py
Run Dashboard | streamlit run app/dashboard/dashboard.py
