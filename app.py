import streamlit as st

from utils.visualization import show_visualization_page
from utils.predictor import predict_diabetes


st.set_page_config(
    page_title="Diabetes Prediction App",
    page_icon="🩺",
    layout="wide"
)

# =========================
# SIDEBAR NAVIGATION
# =========================

st.sidebar.title("🩺 Diabetes App")
st.sidebar.write("ANN Prediction System")

if "menu" not in st.session_state:
    st.session_state.menu = "Home"

if st.sidebar.button("🏠 Home", use_container_width=True):
    st.session_state.menu = "Home"

if st.sidebar.button("📊 Visualization", use_container_width=True):
    st.session_state.menu = "Visualization"

if st.sidebar.button("🔍 Predict", use_container_width=True):
    st.session_state.menu = "Predict"

menu = st.session_state.menu


# =========================
# HOME PAGE
# =========================

if menu == "Home":

    st.title("🩺 Diabetes Prediction App")

    st.write("""
    Aplikasi ini menggunakan Artificial Neural Network (ANN)
    untuk memprediksi kemungkinan diabetes berdasarkan kondisi kesehatan pasien.
    """)

    st.subheader("Model Information")

    st.markdown("""
    - **Model**: Artificial Neural Network (ANN)
    - **Architecture**: 64 → 32 → 1
    - **Hidden Activation**: ReLU
    - **Output Activation**: Sigmoid
    - **Regularization**: Dropout (0.2)
    - **Optimizer**: Adam
    - **Scaler**: RobustScaler
    - **Threshold**: 0.50
    """)

    st.info("""
    Catatan: Aplikasi ini dibuat untuk kebutuhan pembelajaran mini project
    dan bukan untuk diagnosis medis resmi.
    """)


# =========================
# VISUALIZATION PAGE
# =========================

elif menu == "Visualization":

    show_visualization_page()


# =========================
# PREDICT PAGE
# =========================

elif menu == "Predict":

    st.title("🔍 Diabetes Prediction")

    st.write(
        "Masukkan data pasien untuk mendapatkan hasil prediksi diabetes."
    )

    st.info("""
    **Panduan pengisian:**
    - Pilih **0** jika kondisi tidak ada / tidak dialami.
    - Pilih **1** jika kondisi ada / dialami.
    - Untuk fitur seperti **Age, Education, Income, dan General Health**, gunakan kategori angka sesuai skala dataset.
    - Hasil prediksi ini hanya untuk pembelajaran dan bukan diagnosis medis resmi.
    """)
    age_options = {
        "18–24 tahun": 1,
        "25–29 tahun": 2,
        "30–34 tahun": 3,
        "35–39 tahun": 4,
        "40–44 tahun": 5,
        "45–49 tahun": 6,
        "50–54 tahun": 7,
        "55–59 tahun": 8,
        "60–64 tahun": 9,
        "65–69 tahun": 10,
        "70–74 tahun": 11,
        "75–79 tahun": 12,
        "80 tahun atau lebih": 13
        }
    
    education_options = {
        "Tidak pernah sekolah / TK": 1,
        "SD (kelas 1–8)": 2,
        "SMP / SMA belum lulus": 3,
        "Lulus SMA / GED": 4,
        "Kuliah 1–3 tahun": 5,
        "Lulusan S1 atau lebih": 6
        }
    income_options = {
        "< $10.000": 1,
        "$10.000–15.000": 2,
        "$15.000–20.000": 3,
        "$20.000–25.000": 4,
        "$25.000–35.000": 5,
        "$35.000–50.000": 6,
        "$50.000–75.000": 7,
        "> $75.000": 8
        }
    
    with st.expander("🩺 Health Condition", expanded=True):
        col1, col2 = st.columns(2)

    with col1:
        HighBP = st.selectbox("High Blood Pressure", [0, 1])
        HighChol = st.selectbox("High Cholesterol", [0, 1])
        CholCheck = st.selectbox("Cholesterol Check", [0, 1])
        BMI = st.number_input("BMI", min_value=10.0, max_value=100.0, value=25.0)

    with col2:
        Stroke = st.selectbox("Stroke", [0, 1])
        HeartDiseaseorAttack = st.selectbox("Heart Disease or Attack", [0, 1])
        GenHlth = st.slider("General Health", 1, 5, 3)
        DiffWalk = st.selectbox("Difficulty Walking", [0, 1])


    with st.expander("🏃 Lifestyle", expanded=False):
     col1, col2 = st.columns(2)

    with col1:
        Smoker = st.selectbox("Smoker", [0, 1])
        PhysActivity = st.selectbox("Physical Activity", [0, 1])
        Fruits = st.selectbox("Fruits Consumption", [0, 1])

    with col2:
        Veggies = st.selectbox("Vegetables Consumption", [0, 1])
        HvyAlcoholConsump = st.selectbox("Heavy Alcohol Consumption", [0, 1])


    with st.expander("🏥 Healthcare Access", expanded=False):
        col1, col2 = st.columns(2)

    with col1:
        AnyHealthcare = st.selectbox("Any Healthcare", [0, 1])

    with col2:
        NoDocbcCost = st.selectbox("No Doctor Because of Cost", [0, 1])


    with st.expander("👤 Demographic & Wellbeing", expanded=False):
        col1, col2 = st.columns(2)

    with col1:
        MentHlth = st.slider("Mental Health Days", 0, 30, 0)
        PhysHlth = st.slider("Physical Health Days", 0, 30, 0)

        sex_options = {
            "Female": 0,
            "Male": 1
        }

        selected_sex = st.selectbox("Sex", list(sex_options.keys()))
        Sex = sex_options[selected_sex]

    with col2:
        selected_age = st.selectbox("Age Range", list(age_options.keys()))
        Age = age_options[selected_age]

        selected_education = st.selectbox("Education Level", list(education_options.keys()))
        Education = education_options[selected_education]

        selected_income = st.selectbox("Income Level", list(income_options.keys()))
        Income = income_options[selected_income]
       

    # =========================
    # PREDICTION BUTTON
    # =========================

    if st.button("Predict", use_container_width=True):

        input_data = {
            "HighBP": HighBP,
            "HighChol": HighChol,
            "CholCheck": CholCheck,
            "BMI": BMI,
            "Smoker": Smoker,
            "Stroke": Stroke,
            "HeartDiseaseorAttack": HeartDiseaseorAttack,
            "PhysActivity": PhysActivity,
            "Fruits": Fruits,
            "Veggies": Veggies,
            "HvyAlcoholConsump": HvyAlcoholConsump,
            "AnyHealthcare": AnyHealthcare,
            "NoDocbcCost": NoDocbcCost,
            "GenHlth": GenHlth,
            "MentHlth": MentHlth,
            "PhysHlth": PhysHlth,
            "DiffWalk": DiffWalk,
            "Sex": Sex,
            "Age": Age,
            "Education": Education,
            "Income": Income
        }

        prediction, probability = predict_diabetes(input_data)

        prob_diabetes = probability * 100
        prob_no_diabetes = (1 - probability) * 100

        st.divider()

        st.subheader("Prediction Result")

        col_result1, col_result2 = st.columns(2)

        with col_result1:
            st.metric(
                "Probability Diabetes",
                f"{prob_diabetes:.2f}%"
            )

        with col_result2:
            st.metric(
                "Probability No Diabetes",
                f"{prob_no_diabetes:.2f}%"
            )

        st.write("Model Confidence Toward Diabetes")
        st.progress(float(probability))

        if prediction == 1:
            st.error("⚠️ Diabetes Detected")
        else:
            st.success("✅ No Diabetes Detected")

        st.caption("""
        Catatan: Hasil prediksi ini hanya digunakan untuk tujuan edukasi
        dan screening awal. Hasil prediksi tidak dapat menggantikan diagnosis
        dokter atau tenaga medis profesional.
        """)