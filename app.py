import streamlit as st
from transformers import BertTokenizer, AutoModelForSequenceClassification
import torch

# 1. Load the Model and Tokenizer we just saved
@st.cache_resource # This keeps the model in memory so it doesn't reload every time
def load_my_model():
    model_path = "./news_classifier_bert"
    tokenizer = BertTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    return tokenizer, model

tokenizer, model = load_my_model()

# 2. Define the Categories (Must match the AG News order)
categories = ["World", "Sports", "Business", "Sci/Tech"]

# 3. Streamlit UI
st.title("🗞️ AI News Topic Classifier")
st.subheader("Phase 2: Task 1 - Fine-tuned BERT")

user_input = st.text_area("Paste a news headline here:", placeholder="e.g., Apple stock rises as new iPhone is announced...")

if st.button("Predict Topic"):
    if user_input.strip() != "":
        # Process the input
        inputs = tokenizer(user_input, return_tensors="pt", truncation=True, padding=True, max_length=64)
        
        # Get Prediction
        with torch.no_grad():
            outputs = model(**inputs)
            prediction = torch.argmax(outputs.logits, dim=-1).item()
        
        # Display Result
        result = categories[prediction]
        st.success(f"The AI classifies this as: **{result}**")
    else:
        st.warning("Please enter some text first!")