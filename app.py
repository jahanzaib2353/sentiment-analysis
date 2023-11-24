import streamlit as st
from transformers import pipeline

# Load pre-trained text classification model
classifier = pipeline("sentiment-analysis")

# Streamlit app
def main():
    st.title("Text Classification App")

    # Input text from user
    user_text = st.text_area("Enter text for classification:", "")

    if st.button("Classify"):
        # Perform classification when the user clicks the "Classify" button
        result = classifier(user_text)[0]
        st.write(f"Prediction: {result['label']} (Confidence: {result['score']:.2f})")

if __name__ == "__main__":
    main()
