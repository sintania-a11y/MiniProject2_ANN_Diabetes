import joblib
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Dropout

# Bangun ulang arsitektur model
model = Sequential([
    Input(shape=(21,), name="input_layer"),

    Dense(
        64,
        activation="relu",
        name="hidden_layer_1"
    ),

    Dropout(
        0.2,
        name="dropout_layer"
    ),

    Dense(
        32,
        activation="relu",
        name="hidden_layer_2"
    ),

    Dense(
        1,
        activation="sigmoid",
        name="output_layer"
    )
])

# Load weights
model.load_weights("diabetes_ann_model.weights.h5")

# Load scaler
scaler = joblib.load("robust_scaler.pkl")

print("Model berhasil di-load")
print("Scaler berhasil di-load")

model.summary()