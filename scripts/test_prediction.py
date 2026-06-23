import pandas as pd
import joblib
from tensorflow.keras.models import load_model

model = load_model("diabetes_ann_model.keras")
scaler = joblib.load("robust_scaler.pkl")

new_patient = pd.DataFrame({
    "HighBP": [1],
    "HighChol": [1],
    "CholCheck": [1],
    "BMI": [35],
    "Smoker": [0],
    "Stroke": [0],
    "HeartDiseaseorAttack": [0],
    "PhysActivity": [0],
    "Fruits": [0],
    "Veggies": [1],
    "HvyAlcoholConsump": [0],
    "AnyHealthcare": [1],
    "NoDocbcCost": [0],
    "GenHlth": [4],
    "MentHlth": [5],
    "PhysHlth": [10],
    "DiffWalk": [1],
    "Sex": [1],
    "Age": [10],
    "Education": [5],
    "Income": [4]
})

new_patient_scaled = scaler.transform(new_patient)

probability = model.predict(new_patient_scaled)[0][0]
prediction = int(probability >= 0.5)

print(f"Probabilitas Diabetes: {probability:.4f}")

if prediction == 1:
    print("Hasil Prediksi: Diabetes")
else:
    print("Hasil Prediksi: Tidak Diabetes")