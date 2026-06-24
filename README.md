# 🩺 Diabetes Prediction App using Artificial Neural Network

![Python](https://img.shields.io/badge/Python-3.11-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-Deployed-red)
![Accuracy](https://img.shields.io/badge/Accuracy-75.07%25-success)
![Recall](https://img.shields.io/badge/Recall-85%25-brightgreen)

## 🔍 Key Findings

Berdasarkan hasil Exploratory Data Analysis (EDA) dan Feature Importance:

- High Blood Pressure (HighBP) merupakan faktor paling berpengaruh terhadap prediksi diabetes dengan kontribusi sekitar 58.75%.
- Individu dengan tekanan darah tinggi memiliki persentase diabetes yang jauh lebih tinggi dibandingkan individu tanpa tekanan darah tinggi.
- Risiko diabetes cenderung meningkat seiring bertambahnya usia.
- Kelompok diabetes memiliki distribusi BMI yang lebih tinggi dibandingkan kelompok non-diabetes.
- Individu yang aktif secara fisik menunjukkan persentase diabetes yang lebih rendah.
- General Health dan High Cholesterol juga menjadi faktor penting dalam proses prediksi.

## 🚀 Live Demo

[Try the App Here](https://miniproject2anndiabetes-jqcfkzztacdobkwlfix2wx.streamlit.app/)

## 📌 Project Overview

Diabetes Prediction App adalah aplikasi berbasis Streamlit yang digunakan untuk memprediksi kemungkinan seseorang mengalami diabetes berdasarkan indikator kesehatan, gaya hidup, dan faktor demografis.

Project ini dibangun sebagai implementasi Artificial Neural Network (ANN) untuk klasifikasi diabetes dan dilengkapi dengan visualisasi data interaktif serta analisis feature importance.

---

## 🎯 Objectives

* Menganalisis faktor-faktor yang berhubungan dengan diabetes.
* Membangun model Artificial Neural Network (ANN) untuk prediksi diabetes.
* Menyediakan aplikasi interaktif yang mudah digunakan oleh pengguna non-teknis.
* Menampilkan insight data dan feature importance secara visual.

---

## 📂 Dataset

Dataset yang digunakan:

**Diabetes Health Indicators Dataset (BRFSS 2015)**

Source:

* CDC Behavioral Risk Factor Surveillance System (BRFSS) 2015
* Kaggle Diabetes Health Indicators Dataset

File yang digunakan:

`diabetes_binary_5050split_health_indicators_BRFSS2015.csv`

Dataset terdiri dari:

* 70,692 records
* 21 fitur kesehatan
* Target: Diabetes_binary
* Distribusi kelas seimbang (50% Diabetes, 50% No Diabetes)

---

## 🤖 Model Architecture

Artificial Neural Network (ANN)

Architecture:

Input Layer (21 Features)
↓
Dense(64, ReLU)
↓
Dropout(0.2)
↓
Dense(32, ReLU)
↓
Dense(1, Sigmoid)

Training Configuration:

* Optimizer: Adam
* Loss Function: Binary Crossentropy
* Scaler: RobustScaler
* Threshold: 0.50

---

## 📊 Model Performance

| Metric    | Score  |
| --------- | ------ |
| Accuracy  | 75.07% |
| Precision | 71%    |
| Recall    | 85%    |
| F1 Score  | 77%    |

ANN menunjukkan kemampuan yang baik dalam mendeteksi pasien diabetes dengan nilai recall yang tinggi.

---

## 📈 Application Features

### Home Page

* Informasi diabetes
![Home Page](assets/home_page(1).png)

* Ringkasan model
![Home Page](assets/home_page(2).png)

* Informasi dataset
![Home Page](assets/home_page(3).png)

### Visualization Page

* Exploratory Data Analysis (EDA)
![Visualization Page](assets/visualization_page(1).png)
![Visualization Page](assets/visualization_page(2).png)

* Feature Importance
![Feature Importance](assets/feature_importance(1).png)
![Feature Importance](assets/feature_importance(2).png)

* Model Summary
![Model Summary](assets/model_summary(1).png)
![Model Summary](assets/model_summary(2).png)
![Model Summary](assets/model_summary(3).png)


### Prediction Page

* Interactive diabetes prediction
* Probability output
* User-friendly feature explanation
![Prediction Page](assets/predict(1).png)
![Prediction Page](assets/predict(2).png)
![Prediction Page](assets/predict(3).png)

---

## 🛠️ Technologies Used

* Python
* Streamlit
* TensorFlow / Keras
* Scikit-Learn
* Plotly
* Pandas

---

## 🚀 How to Run

git clone ...
cd MiniProject2_ANN_Diabetes
pip install -r requirements.txt
streamlit run app.py

---

## 👨‍💻 Author

Developed as a Data Science Mini Project focusing on healthcare analytics and diabetes prediction using Artificial Neural Networks.
