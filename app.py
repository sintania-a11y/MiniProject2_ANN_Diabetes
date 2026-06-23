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
    Aplikasi ini menggunakan **Artificial Neural Network (ANN)** untuk
    memprediksi kemungkinan seseorang memiliki diabetes berdasarkan indikator
    kesehatan, gaya hidup, dan faktor demografis.
    """)

    st.warning("""
    **Disclaimer:** Aplikasi ini dibuat untuk tujuan pembelajaran dan screening awal.
    Hasil prediksi tidak dapat menggantikan diagnosis dokter atau tenaga medis profesional.
    """)

    st.divider()

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Model", "ANN")

    with col2:
        st.metric("Accuracy", "75.07%")

    with col3:
        st.metric("Recall", "85%")

    st.divider()

    st.subheader("📌 About Diabetes")

    st.write("""
    Diabetes adalah kondisi kronis ketika tubuh tidak dapat mengatur kadar gula darah
    dengan baik. Faktor seperti tekanan darah tinggi, kolesterol tinggi, BMI, usia,
    aktivitas fisik, dan kondisi kesehatan umum dapat berkaitan dengan risiko diabetes.
    """)

    st.subheader("🎯 Application Purpose")

    st.markdown("""
    Aplikasi ini bertujuan untuk:
    - Membantu memahami faktor-faktor yang berkaitan dengan diabetes.
    - Menampilkan insight dari data kesehatan.
    - Memprediksi kemungkinan diabetes menggunakan model ANN.
    - Menyediakan tampilan interaktif berbasis Streamlit.
    """)

    st.subheader("🤖 Model Information")

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

    st.subheader("📂 Dataset Source")

    st.write("""
    Dataset yang digunakan adalah **diabetes_binary_5050split_health_indicators_BRFSS2015.csv**,
    yaitu dataset seimbang yang berasal dari data survei **CDC Behavioral Risk Factor
    Surveillance System (BRFSS) 2015**. Dataset ini tersedia melalui Kaggle dalam
    kumpulan **Diabetes Health Indicators Dataset**.
    """)

    st.markdown("""
    **Dataset details:**
    - Source: CDC BRFSS 2015
    - Kaggle dataset: Diabetes Health Indicators Dataset
    - File used: diabetes_binary_5050split_health_indicators_BRFSS2015.csv
    - Features: 21 health indicator features
    - Target: Diabetes_binary
    - Class distribution: 50% Diabetes and 50% No Diabetes
    """)

    st.info("""
    Dataset ini digunakan karena memiliki indikator kesehatan yang relevan untuk
    analisis dan prediksi diabetes, seperti HighBP, HighChol, BMI, GenHlth, Age,
    Physical Activity, dan fitur lainnya.
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
        HighBP = st.selectbox(
            "History of High Blood Pressure",
            [0, 1],
            help="""
            Menunjukkan apakah seseorang memiliki tekanan darah tinggi (hipertensi). \n
            0 = Tidak memiliki tekanan darah tinggi \n
            1 = Memiliki tekanan darah tinggi atau pernah didiagnosis hipertensi oleh dokter \n
            Contoh: \n
            ✓ Tekanan darah normal → 0 \n
            ✓ Pernah didiagnosis hipertensi → 1 \n
            ✓ Mengonsumsi obat hipertensi secara rutin → 1 \n

            Catatan: \n
            Tekanan darah tinggi merupakan salah satu faktor risiko utama diabetes dan penyakit kardiovaskular.""")
        HighChol = st.selectbox(
            "History of High Cholesterol",
            [0, 1],
            help="""
            Menunjukkan apakah seseorang memiliki kadar kolesterol tinggi. \n
            0 = Tidak memiliki kolesterol tinggi \n
            1 = Memiliki kolesterol tinggi atau pernah didiagnosis kolesterol tinggi oleh dokter \n

            Contoh: \n
            ✓ Hasil pemeriksaan kolesterol normal → 0 \n
            ✓ Pernah didiagnosis kolesterol tinggi → 1 \n
            ✓ Mengonsumsi obat penurun kolesterol → 1 \n

            Catatan: \n
            Kolesterol tinggi dapat meningkatkan risiko penyakit jantung dan sering berkaitan dengan diabetes tipe 2.""")        
        CholCheck = st.selectbox(
            "Recent Cholesterol Screening", 
            [0, 1],
            help="""
            Menunjukkan apakah Anda pernah melakukan
            pemeriksaan kolesterol dalam 5 tahun terakhir. \n

            0 = Tidak pernah melakukan pemeriksaan kolesterol
            dalam 5 tahun terakhir \n

            1 = Pernah melakukan pemeriksaan kolesterol
            dalam 5 tahun terakhir \n

            Contoh: \n
            ✓ Cek kolesterol saat medical check-up → 1 \n
            ✓ Cek kolesterol di klinik atau rumah sakit → 1 \n
            ✗ Tidak pernah cek kolesterol dalam 5 tahun terakhir → 0 \n

            Catatan: \n
            Fitur ini tidak menunjukkan apakah kolesterol Anda tinggi atau rendah.
            Hanya menunjukkan apakah pernah diperiksa atau tidak.""")
        BMI = st.number_input(
            "BMI",
            min_value=10.0,
            max_value=100.0,
            value=25.0,
            help="""
            BMI (Body Mass Index) \n
            Rumus: BMI = Berat Badan (kg) / (Tinggi Badan (m)²)\n
            Kategori BMI:
            < 18.5 = Underweight
            18.5 - 24.9 = Normal
            25.0 - 29.9 = Overweight
            ≥ 30 = Obese \n
            Contoh:
            Tinggi 170 cm, Berat 72 kg → BMI ≈ 24.9
            """)
        if BMI < 18.5:
            st.info("BMI Category: Underweight")

        elif BMI < 25:
            st.success("BMI Category: Normal Weight")

        elif BMI < 30:
            st.warning("BMI Category: Overweight")

        else:
            st.error("BMI Category: Obese")

    with col2:
        Stroke = st.selectbox(
            "History of Stroke", 
            [0, 1],
            help="""
            Menunjukkan apakah Anda pernah mengalami stroke. \n
            0 = Tidak pernah mengalami stroke \n
            1 = Pernah mengalami stroke \n

            Contoh: \n 
            ✓ Tidak pernah didiagnosis stroke → 0 \n
            ✓ Pernah didiagnosis stroke oleh dokter → 1 \n

            Catatan: \n
            Yang dimaksud adalah riwayat stroke pada diri sendiri,
            bukan anggota keluarga. """)
        HeartDiseaseorAttack = st.selectbox(
            "History of Heart Disease or Attack", 
            [0, 1],
            help="""
            Menunjukkan apakah Anda pernah didiagnosis
            penyakit jantung atau serangan jantung. \n

            0 = Tidak pernah memiliki penyakit jantung
            atau serangan jantung \n
            1 = Pernah didiagnosis penyakit jantung
            atau mengalami serangan jantung \n      

            Contoh: \n
            ✓ Tidak ada riwayat penyakit jantung → 0 \n        
            ✓ Pernah mengalami serangan jantung → 1 \n
            ✓ Pernah didiagnosis penyakit jantung koroner → 1 \n

            Catatan:\n
            Yang dimaksud adalah riwayat pada diri sendiri,
            bukan riwayat keluarga.""")
        GenHlth = st.slider(
            "General Health",
            1, 5, 3,
            help="""
            1 = Excellent
            2 = Very Good
            3 = Good
            4 = Fair
            5 = Poor

            Pilih sesuai kondisi kesehatan secara umum.
            """)
        DiffWalk = st.selectbox(
            "Difficulty Walking", 
            [0, 1],
            help="""
            0 = Tidak mengalami kesulitan berjalan atau menaiki tangga \n
            1 = Mengalami kesulitan berjalan atau menaiki tangga \n

            Contoh: \n
            ✓ Mudah berjalan normal → 0 \n
            ✓ Cepat lelah atau kesulitan bergerak → 1""")


    with st.expander("🏃 Lifestyle", expanded=False):
     col1, col2 = st.columns(2)

    with col1:
        Smoker = st.selectbox(
            "History of Smoking", 
            [0, 1],
            help="""
            Menunjukkan apakah seseorang memiliki riwayat merokok. \n
            0 = Tidak memiliki riwayat merokok yang signifikan \n
            1 = Memiliki riwayat merokok \n
            
            Definisi BRFSS: \n
            • Pernah merokok setidaknya 100 batang rokok sepanjang hidup → 1 \n
            • Belum pernah mencapai 100 batang rokok sepanjang hidup → 0 \n
            
            Contoh: \n
            ✓ Tidak pernah merokok → 0 \n
            ✓ Hanya mencoba beberapa batang rokok → 0 \n
            ✓ Pernah menjadi perokok aktif dalam jangka waktu lama → 1 \n
            ✓ Mantan perokok → 1 \n

            Catatan: \n
            Fitur ini mengukur riwayat merokok, bukan hanya apakah sedang merokok saat ini.""")

        PhysActivity = st.selectbox(
            "Physical Activity",
            [0, 1],
            help="""
            0 = Tidak melakukan aktivitas fisik rutin
            1 = Melakukan aktivitas fisik rutin
            dalam 30 hari terakhir
            """)
        Fruits = st.selectbox(
            "Fruits Consumption",
            [0, 1],
            help="""
            0 = Jarang/tidak mengonsumsi buah
            1 = Mengonsumsi buah secara rutin
            """)
    with col2:
        Veggies = st.selectbox(
            "Vegetables Consumption",
            [0, 1],
            help="""
            0 = Jarang/tidak mengonsumsi sayuran
            1 = Mengonsumsi sayuran secara rutin
            """)    
        HvyAlcoholConsump = st.selectbox(
            "Heavy Alcohol Consumption (Excessive Drinking)", 
            [0, 1],
            help="""
            Menunjukkan apakah seseorang mengonsumsi alkohol
            dalam jumlah berlebihan. \n
            0 = Tidak mengonsumsi alkohol berlebihan \n
            1 = Mengonsumsi alkohol berlebihan \n
            Definisi BRFSS: \n
            • Pria: lebih dari 14 minuman beralkohol per minggu \n
            • Wanita: lebih dari 7 minuman beralkohol per minggu \n

            Contoh: \n
            ✓ Sesekali minum alkohol → 0 \n
            ✓ Tidak minum alkohol sama sekali → 0 \n 
            ✓ Mengonsumsi alkohol dalam jumlah besar secara rutin → 1 \n

            Catatan: \n
            Fitur ini mengukur konsumsi alkohol berlebihan, \n
            bukan sekadar pernah atau tidak pernah minum alkohol.""")


    with st.expander("🏥 Healthcare Access", expanded=False):
        col1, col2 = st.columns(2)

    with col1:
        AnyHealthcare = st.selectbox(
            "Access to Healthcare",
            [0, 1],
            help="""
            0 = Tidak memiliki akses atau perlindungan layanan kesehatan

            1 = Memiliki akses atau perlindungan layanan kesehatan
            (asuransi kesehatan, BPJS, fasilitas kesehatan, dll)

            Contoh: \n
            ✓ Memiliki BPJS → pilih 1 \n
            ✓ Memiliki asuransi kesehatan → pilih 1 \n
            ✗ Tidak memiliki perlindungan kesehatan → pilih 0""")
    with col2:
        NoDocbcCost = st.selectbox(
            "Unable to Visit Doctor Due to Cost",
            [0, 1],
            help="""
            0 = Tidak pernah menunda atau membatalkan
            kunjungan ke dokter karena biaya

            1 = Pernah tidak pergi ke dokter
            karena biaya terlalu mahal

            Contoh: \n
            ✓ Ingin periksa tetapi biaya menjadi hambatan → pilih 1 \n
            ✓ Tidak ada kendala biaya saat berobat → pilih 0""")

    with st.expander("👤 Demographic & Wellbeing", expanded=False):
        col1, col2 = st.columns(2)

    with col1:
        MentHlth = st.slider(
            "Mental Health Days",
            0, 30, 0,
            help="""
            Jumlah hari dalam 30 hari terakhir
            ketika kondisi mental kurang baik. \n

            Contoh: \n
            - Stres \n
            - Cemas \n
            - Depresi \n
            - Sulit tidur karena masalah psikologis \n
            - Burnout \n

            Nilai: \n
            0  = Tidak ada gangguan mental \n
            15 = Sekitar setengah bulan mengalami gangguan \n
            30 = Hampir setiap hari mengalami gangguan""")
        PhysHlth = st.slider(
            "Physical Health Days", 
            0, 30, 0,
            help="""
            Jumlah hari dalam 30 hari terakhir
            ketika kondisi fisik kurang baik. \n

            Contoh: \n
            - Sakit demam \n
            - Flu \n
            - Nyeri tubuh \n
            - Cedera \n
            - Penyakit kronis yang mengganggu aktivitas \n

            Nilai: \n
            0  = Tidak ada hari sakit \n
            15 = Sekitar setengah bulan merasa tidak sehat \n
            30 = Seluruh bulan merasa kondisi fisik kurang baik""")

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