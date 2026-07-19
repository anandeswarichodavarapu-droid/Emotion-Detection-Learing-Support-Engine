import streamlit as st

from config.config import Config
from src.preprocessing import preprocess_text
from src.bilstm_predictor import BiLSTMPredictor

# ----------------------------
# Load Predictor
# ----------------------------
predictor = BiLSTMPredictor()

# ----------------------------
# Page Configuration
# ----------------------------
st.set_page_config(
    page_title=Config.APP_NAME,
    page_icon="🧠",
    layout="wide"
)

# ----------------------------
# Session State
# ----------------------------
if "history" not in st.session_state:
    st.session_state.history = []

# ----------------------------
# Sidebar
# ----------------------------
st.sidebar.title("🧠 Emotion Detection")
st.sidebar.success("System Ready")

st.sidebar.markdown("---")
st.sidebar.subheader("Model Status")

st.sidebar.success("✅ BiLSTM Ready")
st.sidebar.info("🤖 BERT (Coming Soon)")
st.sidebar.info("✨ Gemini (Coming Soon)")

st.sidebar.markdown("---")
st.sidebar.metric("Interactions", len(st.session_state.history))

# ----------------------------
# Main Title
# ----------------------------
st.title("🎓 Emotion Detection & Learning Support Engine")

st.write("""
Welcome to the AI Learning Assistant.

This application detects student emotions and provides learning support.
""")

# ----------------------------
# Input
# ----------------------------
field = st.selectbox(
    "Select Academic Field",
    [
        "Computer Science",
        "Mathematics",
        "Physics",
        "Chemistry",
        "Biology",
        "Electronics",
        "Mechanical",
        "Civil",
        "English",
        "Business",
        "Other"
    ]
)

problem = st.text_area(
    "Describe your learning problem",
    height=180,
    placeholder="Example: I understand loops but recursion is confusing..."
)

col1, col2 = st.columns(2)

with col1:
    analyze = st.button("🔍 Analyze Emotion")

with col2:
    clear = st.button("🗑 Clear")

# ----------------------------
# Analyze
# ----------------------------
if analyze:

    if problem.strip() == "":
        st.warning("Please enter your learning problem.")
        st.stop()

    result = preprocess_text(problem)

    st.subheader("📝 Preprocessed Text")

    st.code(result["cleaned_text"])

    st.write(result["tokens"])

    prediction = predictor.predict(result["cleaned_text"])

    st.subheader("🧠 Emotion Prediction")

    st.success(f"Detected Emotion: {prediction['emotion']}")

    st.metric(
        "Confidence",
        f"{prediction['confidence']:.2%}"
    )

    st.json(prediction["probabilities"])

    st.session_state.history.append(
        {
            "Field": field,
            "Problem": problem,
            "Emotion": prediction["emotion"]
        }
    )

# ----------------------------
# Session History
# ----------------------------
if st.session_state.history:

    st.divider()

    st.subheader("📜 Session History")

    st.dataframe(st.session_state.history)

# ----------------------------
# Clear
# ----------------------------
if clear:
    st.session_state.history = []
    st.rerun()