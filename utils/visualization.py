import pandas as pd
import plotly.express as px
import streamlit as st


def load_dataset():
    df = pd.read_csv(
        "data/diabetes_binary_5050split_health_indicators_BRFSS2015.csv"
    )

    df["Diabetes Label"] = df["Diabetes_binary"].map({
        0.0: "No Diabetes",
        1.0: "Diabetes"
    })

    return df


def bmi_boxplot(df):
    fig = px.box(
        df,
        x="Diabetes Label",
        y="BMI",
        color="Diabetes Label",
        title="BMI Distribution by Diabetes Status"
    )
    fig.update_layout(showlegend=False)
    return fig


def diabetes_by_highbp(df):
    highbp_rate = df.groupby("HighBP")["Diabetes_binary"].mean().reset_index()
    highbp_rate["Diabetes Rate (%)"] = highbp_rate["Diabetes_binary"] * 100
    highbp_rate["HighBP Label"] = highbp_rate["HighBP"].map({
        0.0: "No High Blood Pressure",
        1.0: "High Blood Pressure"
    })

    fig = px.bar(
        highbp_rate,
        x="HighBP Label",
        y="Diabetes Rate (%)",
        text="Diabetes Rate (%)",
        title="Diabetes Rate by Blood Pressure Status"
    )
    fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    fig.update_layout(yaxis_range=[0, 100])
    return fig


def diabetes_by_age(df):
    age_rate = df.groupby("Age")["Diabetes_binary"].mean().reset_index()
    age_rate["Diabetes Rate (%)"] = age_rate["Diabetes_binary"] * 100

    fig = px.line(
        age_rate,
        x="Age",
        y="Diabetes Rate (%)",
        markers=True,
        title="Diabetes Rate by Age Category"
    )
    return fig


def diabetes_by_activity(df):
    activity_rate = df.groupby("PhysActivity")["Diabetes_binary"].mean().reset_index()
    activity_rate["Diabetes Rate (%)"] = activity_rate["Diabetes_binary"] * 100
    activity_rate["Activity Label"] = activity_rate["PhysActivity"].map({
        0.0: "Not Physically Active",
        1.0: "Physically Active"
    })

    fig = px.bar(
        activity_rate,
        x="Activity Label",
        y="Diabetes Rate (%)",
        text="Diabetes Rate (%)",
        title="Diabetes Rate by Physical Activity"
    )
    fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    fig.update_layout(yaxis_range=[0, 100])
    return fig


def get_feature_importance_df():
    df_importance = pd.DataFrame({
        "Feature": [
            "HighBP", "GenHlth", "HighChol", "Age",
            "HeartDiseaseorAttack", "DiffWalk", "BMI",
            "CholCheck", "HvyAlcoholConsump", "Income",
            "Sex", "Stroke", "Education"
        ],
        "Importance": [
            0.5714, 0.1031, 0.0887, 0.0377,
            0.0354, 0.0337, 0.0259,
            0.0235, 0.0207, 0.0108,
            0.0107, 0.0060, 0.0050
        ]
    })

    df_importance["Percentage"] = (
        df_importance["Importance"] / df_importance["Importance"].sum() * 100
    )

    return df_importance


def feature_importance_bar(df_importance, top_n=10):
    top_df = df_importance.head(top_n).sort_values("Importance")

    fig = px.bar(
        top_df,
        x="Importance",
        y="Feature",
        orientation="h",
        text="Importance",
        title=f"Top {top_n} XGBoost Feature Importance"
    )

    fig.update_traces(texttemplate="%{text:.4f}", textposition="outside")
    fig.update_layout(height=500)
    return fig


def feature_importance_pie(df_importance, top_n=10):
    top_df = df_importance.head(top_n)
    others_value = df_importance.iloc[top_n:]["Importance"].sum()

    pie_df = top_df[["Feature", "Importance"]].copy()

    if others_value > 0:
        pie_df.loc[len(pie_df)] = ["Others", others_value]

    fig = px.pie(
        pie_df,
        names="Feature",
        values="Importance",
        hole=0.45,
        title=f"Feature Importance Contribution Top {top_n} vs Others"
    )

    return fig


def show_model_summary():
    st.subheader("📌 Model Summary & Architecture Decision")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Accuracy", "75.07%")
    col2.metric("Precision", "71%")
    col3.metric("Recall", "85%")
    col4.metric("F1 Score", "77%")

    st.divider()

    st.subheader("ANN Architecture Explanation")

    architecture_df = pd.DataFrame({
        "Layer": [
            "Input Layer",
            "Hidden Layer 1",
            "Dropout",
            "Hidden Layer 2",
            "Output Layer"
        ],
        "Configuration": [
            "21 input features",
            "Dense(64), ReLU",
            "Dropout(0.2)",
            "Dense(32), ReLU",
            "Dense(1), Sigmoid"
        ],
        "Reason": [
            "Dataset memiliki 21 fitur kesehatan sebagai input model.",
            "64 neuron digunakan untuk menangkap pola kompleks antar fitur.",
            "Dropout membantu mengurangi risiko overfitting.",
            "32 neuron menyederhanakan representasi sebelum output.",
            "Sigmoid digunakan karena target bersifat biner."
        ]
    })

    st.dataframe(architecture_df, use_container_width=True)

    st.info("""
    Arsitektur **64 → 32 → 1** dipilih karena cukup kuat untuk mempelajari
    pola non-linear, tetapi tetap sederhana agar tidak menambah kompleksitas
    model secara berlebihan.
    """)

    st.divider()

    comparison_df = pd.DataFrame({
        "Model": [
            "Logistic Regression",
            "Decision Tree Tuned",
            "Random Forest Tuned",
            "XGBoost Tuned",
            "ANN"
        ],
        "Accuracy": [0.7431, 0.7396, 0.7457, 0.7494, 0.7507],
        "Precision": [0.7393, 0.7263, 0.7260, 0.7293, 0.7100],
        "Recall": [0.7640, 0.7828, 0.8027, 0.8061, 0.8500],
        "F1 Score": [0.7514, 0.7535, 0.7624, 0.7658, 0.7700]
    })

    st.subheader("ANN vs Machine Learning Model Comparison")

    metric_option = st.selectbox(
        "Select metric to compare",
        ["Accuracy", "Precision", "Recall", "F1 Score"]
    )

    fig = px.bar(
        comparison_df,
        x="Model",
        y=metric_option,
        color="Model",
        text=metric_option,
        title=f"Model Comparison by {metric_option}"
    )

    fig.update_traces(
        texttemplate="%{text:.4f}",
        textposition="outside"
    )

    fig.update_layout(
        yaxis_range=[0, 1],
        showlegend=False
    )

    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(comparison_df, use_container_width=True)

    st.success("""
    **Analysis:** ANN memiliki recall tertinggi dibanding model Machine Learning
    dari Mini Project 1. Ini berarti ANN lebih baik dalam mendeteksi pasien yang
    benar-benar diabetes. Namun, XGBoost masih lebih mudah diinterpretasikan karena
    menyediakan feature importance yang jelas.
    """)


def show_visualization_page():
    st.title("📊 Diabetes Data Insights")

    st.write("""
    Halaman ini menampilkan insight dari data diabetes, feature importance,
    serta ringkasan keputusan arsitektur model ANN.
    """)

    df = load_dataset()

    tab1, tab2, tab3 = st.tabs([
        "EDA Insights",
        "Feature Importance",
        "Model Summary"
    ])

    with tab1:
        st.subheader("Exploratory Data Analysis Insights")

        col1, col2 = st.columns(2)

        with col1:
            st.plotly_chart(bmi_boxplot(df), use_container_width=True)
            st.info("Kelompok diabetes cenderung memiliki nilai BMI yang lebih tinggi.")

        with col2:
            st.plotly_chart(diabetes_by_highbp(df), use_container_width=True)
            st.info("Tekanan darah tinggi berkaitan dengan persentase diabetes yang lebih besar.")

        col3, col4 = st.columns(2)

        with col3:
            st.plotly_chart(diabetes_by_age(df), use_container_width=True)
            st.info("Persentase diabetes meningkat pada kelompok usia yang lebih tua.")

        with col4:
            st.plotly_chart(diabetes_by_activity(df), use_container_width=True)
            st.info("Aktivitas fisik berkaitan dengan persentase diabetes yang lebih rendah.")

        st.success("""
        **Key Insight:** BMI, tekanan darah tinggi, usia, dan aktivitas fisik
        merupakan faktor yang terlihat berkaitan dengan status diabetes.
        """)

    with tab2:
        st.subheader("XGBoost Feature Importance")

        df_importance = get_feature_importance_df()

        top_n = st.slider(
            "Pilih jumlah fitur yang ditampilkan",
            min_value=5,
            max_value=len(df_importance),
            value=10
        )

        col1, col2 = st.columns(2)

        with col1:
            st.plotly_chart(
                feature_importance_bar(df_importance, top_n),
                use_container_width=True
            )

        with col2:
            st.plotly_chart(
                feature_importance_pie(df_importance, top_n),
                use_container_width=True
            )

        st.subheader("Feature Importance Table")

        st.dataframe(
            df_importance.style.format({
                "Importance": "{:.4f}",
                "Percentage": "{:.2f}%"
            }),
            use_container_width=True
        )

        st.success("""
        **Key Insight:** High Blood Pressure menjadi fitur paling dominan,
        diikuti oleh General Health dan High Cholesterol.
        """)

    with tab3:
        show_model_summary()