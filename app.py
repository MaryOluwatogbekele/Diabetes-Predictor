#### Creating a Simple Website Interface Using Streamlit
##### Customizing Streamlit UI using markdown, sidebar, themes, images, etc.

import streamlit as st
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# --------------------------------------------------
# Page configuration (MUST be first Streamlit call)
# --------------------------------------------------
st.set_page_config(
    page_title="Diabetes Prediction System",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --------------------------------------------------
# Load trained model
# --------------------------------------------------
model = joblib.load("Best_Bagging_Model_For_Women_Diabetes_Risk_Prediction.pkl")

# --------------------------------------------------
# Custom CSS
# --------------------------------------------------
st.markdown(
    """
    <style>
        .main { background-color: #f0f2f6; }
        .stApp { max-width: 1200px; margin: 0 auto; }
        .header {
            color: #1e3a8a;
            text-align: center;
            padding: 1rem;
            border-bottom: 2px solid #3b82f6;
            margin-bottom: 2rem;
        }
        .feature-card {
            background-color: white;
            border-radius: 10px;
            padding: 1rem;
            margin-bottom: 1rem;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        .prediction-card {
            background-color: #dbeafe;
            border-radius: 10px;
            padding: 2rem;
            text-align: center;
            margin-top: 1rem;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        .negative { color: #10b981; font-weight: bold; }
        .positive { color: #ef4444; font-weight: bold; }
        .footer {
            text-align: center;
            margin-top: 3rem;
            padding-top: 1rem;
            border-top: 1px solid #e5e7eb;
            color: #6b7280;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------
# Header & intro
# --------------------------------------------------
st.markdown(
    '<div class="header"><h1>🩺 Diabetes Prediction System</h1></div>',
    unsafe_allow_html=True,
)

st.markdown(
    """
    This application uses machine learning to predict diabetes risk based on
    health metrics. Enter your health information to receive a prediction.
    """
)

# --------------------------------------------------
# Layout
# --------------------------------------------------
col1, col2 = st.columns(2, gap="large")

# --------------------------------------------------
# Input form
# --------------------------------------------------
with col1:
    st.markdown("### Patient Information")

    with st.form("prediction_form"):
        st.markdown('<div class="feature-card">', unsafe_allow_html=True)
        pregnancies = st.slider("Pregnancies", 0, 17, 1)
        glucose = st.slider("Glucose Level (mg/dL)", 0, 200, 100)
        blood_pressure = st.slider("Blood Pressure (mm Hg)", 0, 130, 70)
        skin_thickness = st.slider("Skin Thickness (mm)", 0, 100, 20)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="feature-card">', unsafe_allow_html=True)
        insulin = st.slider("Insulin Level (μU/mL)", 0, 900, 80)
        bmi = st.slider("BMI", 10.0, 70.0, 25.0, 0.1)
        pedigree = st.slider("Diabetes Pedigree Function", 0.0, 3.0, 0.5, 0.01)
        age = st.slider("Age", 20, 100, 35)
        st.markdown("</div>", unsafe_allow_html=True)

        submit_button = st.form_submit_button("Predict Diabetes Risk")

# --------------------------------------------------
# Prediction results
# --------------------------------------------------
with col2:
    st.markdown("### Prediction Results")

    if submit_button:
        input_data = np.array(
            [[pregnancies, glucose, blood_pressure,
              skin_thickness, insulin, bmi, pedigree, age]]
        )

        prediction = model.predict(input_data)[0]
        probabilities = model.predict_proba(input_data)[0]

        st.markdown('<div class="prediction-card">', unsafe_allow_html=True)

        if prediction == 0:
            probability = probabilities[0] * 100
            label = "Low Diabetes Risk"
            color = "#10b981"

            st.markdown(
                f'<h2 class="negative">Prediction: {label}</h2>',
                unsafe_allow_html=True,
            )
            st.markdown(
                f"<h3>Probability: {probability:.1f}%</h3>",
                unsafe_allow_html=True,
            )
            st.success("Based on your inputs, you have a low risk of diabetes.")
        else:
            probability = probabilities[1] * 100
            label = "High Diabetes Risk"
            color = "#ef4444"

            st.markdown(
                f'<h2 class="positive">Prediction: {label}</h2>',
                unsafe_allow_html=True,
            )
            st.markdown(
                f"<h3>Probability: {probability:.1f}%</h3>",
                unsafe_allow_html=True,
            )
            st.error(
                "Based on your inputs, you may be at risk of diabetes. "
                "Please consult a healthcare professional."
            )

        # Risk probability bar (single, consistent)
        fig, ax = plt.subplots(figsize=(8, 2))
        ax.barh([label], [probability], color=color)
        ax.set_xlim(0, 100)
        ax.set_xlabel("Probability (%)")
        ax.set_title("Predicted Class Probability")
        ax.grid(axis="x", linestyle="--", alpha=0.6)
        st.pyplot(fig)

        st.markdown("</div>", unsafe_allow_html=True)

        # Feature importance (if supported)
        if hasattr(model, "feature_importances_"):
            st.markdown("### Feature Importance")
            features = [
                "Pregnancies", "Glucose", "Blood Pressure",
                "Skin Thickness", "Insulin", "BMI", "Pedigree", "Age"
            ]
            importance = model.feature_importances_

            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(x=importance, y=features, ax=ax)
            ax.set_title("Feature Importance in Prediction")
            ax.set_xlabel("Importance Score")
            st.pyplot(fig)

    else:
        st.info(
            "Please enter your health information and click "
            "'Predict Diabetes Risk' to see results."
        )
        st.image(
            "https://images.unsplash.com/photo-1532938911079-1b06ac7ceec7"
            "?auto=format&fit=crop&w=600&h=400",
            caption="Diabetes Risk Assessment"
        )

# --------------------------------------------------
# Footer
# --------------------------------------------------
st.markdown('<div class="footer">', unsafe_allow_html=True)
st.markdown(
    """
    **Note:** This prediction is based on machine learning algorithms and does not
    replace professional medical advice. Always consult a healthcare provider.
    """
)
st.markdown("© 2025 Diabetes Prediction System | Developed with Streamlit")
st.markdown("</div>", unsafe_allow_html=True)
