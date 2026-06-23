import pandas as pd
import joblib
from tensorflow.keras.models import load_model

MODEL_PATH = "models/diabetes_ann_model.keras"
SCALER_PATH = "models/robust_scaler.pkl"

model = load_model(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)


def predict_diabetes(input_data, threshold=0.5):
    """
    Melakukan prediksi diabetes menggunakan model ANN.

    Returns:
        prediction: 0 = No Diabetes, 1 = Diabetes
        probability: probabilitas diabetes
    """

    input_df = pd.DataFrame([input_data])

    input_scaled = scaler.transform(input_df)

    probability = float(
        model.predict(input_scaled, verbose=0)[0][0]
    )

    prediction = int(probability >= threshold)

    return prediction, probability