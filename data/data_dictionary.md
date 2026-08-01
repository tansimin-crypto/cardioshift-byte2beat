# Data dictionary

| Column | Role | Definition |
|---|---|---|
| patient_id | identifier | Stable local key: normalized site plus one-based source row |
| site | validation-only | Cleveland, Hungary, Switzerland, or VA Long Beach |
| age | predictor | Age in years |
| sex | predictor | 1 male, 0 female in the source documentation |
| cp | predictor | Chest-pain type, source values 1–4 |
| trestbps | predictor | Resting blood pressure on admission, mm Hg |
| chol | predictor | Serum cholesterol, mg/dL |
| fbs | predictor | Fasting blood sugar >120 mg/dL indicator |
| restecg | predictor | Resting ECG category |
| thalach | predictor | Maximum heart rate achieved |
| exang | predictor | Exercise-induced angina indicator |
| oldpeak | predictor | Exercise-induced ST depression relative to rest |
| slope | predictor | Peak exercise ST-segment slope |
| ca | predictor | Number of major vessels colored by fluoroscopy |
| thal | predictor | Thallium test category |
| num | source outcome | UCI angiographic disease-status value 0–4 |
| target | analysis outcome | 0 when num=0, otherwise 1 |

`site` and `patient_id` are forbidden disease-prediction features. Missing values
are preserved as `NaN` until a training-fold pipeline handles them.
