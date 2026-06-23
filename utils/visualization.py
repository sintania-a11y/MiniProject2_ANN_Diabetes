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
        title="BMI Distribution by Diabetes Status",
        labels={
            "Diabetes Label": "Diabetes Status",
            "BMI": "BMI"
        }
    )

    fig.update_layout(showlegend=False)

    return fig


def diabetes_by_highbp(df):
    highbp_rate = (
        df.groupby("HighBP")["Diabetes_binary"]
        .mean()
        .reset_index()
    )

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
        title="Diabetes Rate by Blood Pressure Status",
        labels={
            "HighBP Label": "Blood Pressure Status",
            "Diabetes Rate (%)": "Diabetes Rate (%)"
        }
    )

    fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    fig.update_layout(yaxis_range=[0, 100])

    return fig


def diabetes_by_age(df):
    age_rate = (
        df.groupby("Age")["Diabetes_binary"]
        .mean()
        .reset_index()
    )

    age_rate["Diabetes Rate (%)"] = age_rate["Diabetes_binary"] * 100

    fig = px.line(
        age_rate,
        x="Age",
        y="Diabetes Rate (%)",
        markers=True,
        title="Diabetes Rate by Age Category",
        labels={
            "Age": "Age Category",
            "Diabetes Rate (%)": "Diabetes Rate (%)"
        }
    )

    return fig


def diabetes_by_activity(df):
    activity_rate = (
        df.groupby("PhysActivity")["Diabetes_binary"]
        .mean()
        .reset_index()
    )

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
        title="Diabetes Rate by Physical Activity",
        labels={
            "Activity Label": "Physical Activity Status",
            "Diabetes Rate (%)": "Diabetes Rate (%)"
        }
    )

    fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    fig.update_layout(yaxis_range=[0, 100])

    return fig


def get_feature_importance_df():
    data = {
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
    }

    df_importance = pd.DataFrame(data)
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
        title=f"Top {top_n} XGBoost Feature Importance",
        labels={
            "Importance": "Importance Score",
            "Feature": "Feature"
        }
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


def show_visualization_page():
    st.title("📊 Diabetes Data Insights")

    st.write("""
    Halaman ini menampilkan insight dari data diabetes dan feature importance
    dari model XGBoost untuk memahami faktor-faktor yang paling berpengaruh
    terhadap prediksi diabetes.
    """)

    df = load_dataset()

    tab1, tab2 = st.tabs([
        "EDA Insights",
        "Feature Importance"
    ])

    with tab1:
        st.subheader("Exploratory Data Analysis Insights")

        col1, col2 = st.columns(2)

        with col1:
            st.plotly_chart(
                bmi_boxplot(df),
                use_container_width=True
            )

            st.info("""
            Kelompok diabetes cenderung memiliki nilai BMI yang lebih tinggi
            dibanding kelompok non-diabetes.
            """)

        with col2:
            st.plotly_chart(
                diabetes_by_highbp(df),
                use_container_width=True
            )

            st.info("""
            Individu dengan tekanan darah tinggi memiliki persentase diabetes
            yang lebih besar dibanding individu tanpa tekanan darah tinggi.
            """)

        col3, col4 = st.columns(2)

        with col3:
            st.plotly_chart(
                diabetes_by_age(df),
                use_container_width=True
            )

            st.info("""
            Persentase diabetes cenderung meningkat pada kategori usia yang lebih tua.
            """)

        with col4:
            st.plotly_chart(
                diabetes_by_activity(df),
                use_container_width=True
            )

            st.info("""
            Individu yang aktif secara fisik cenderung memiliki persentase diabetes
            yang lebih rendah.
            """)

        st.success("""
        **Key Insight:** Faktor kesehatan seperti BMI, tekanan darah tinggi,
        usia, dan aktivitas fisik memiliki hubungan yang kuat dengan status diabetes.
        """)

    with tab2:
        st.subheader("XGBoost Feature Importance")

        st.write("""
        Feature importance menunjukkan fitur mana yang paling besar kontribusinya
        dalam membantu model XGBoost membedakan pasien diabetes dan non-diabetes.
        """)

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
        diikuti oleh General Health dan High Cholesterol. Hal ini menunjukkan bahwa
        faktor kesehatan klinis memiliki kontribusi lebih besar dibanding faktor demografis.
        """)