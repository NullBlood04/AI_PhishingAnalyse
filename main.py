import streamlit as st
import torch
from PIL import Image
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain.schema import SystemMessage, HumanMessage, Document
from transformers import CLIPProcessor, CLIPModel
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
import json
import numpy as np

# Load from secrets
AZURE_OPENAI_API_KEY = st.secrets["AZURE_OPENAI_API_KEY"]
AZURE_DEPLOYMENT_NAME = st.secrets["AZURE_DEPLOYMENT_NAME"]
AZURE_RESOURCE_NAME = st.secrets["AZURE_RESOURCE_NAME"]
AZURE_API_VERSION = st.secrets["AZURE_API_VERSION"]
AZURE_EMBEDDING_MODEL_NAME = st.secrets["AZURE_EMBEDDING_MODEL_NAME"]

# Initialize LangChain Azure OpenAI
chat = AzureChatOpenAI(
    azure_deployment=AZURE_DEPLOYMENT_NAME,
    azure_endpoint=f"https://{AZURE_RESOURCE_NAME}.openai.azure.com/",
    api_key=AZURE_OPENAI_API_KEY,
    api_version=AZURE_API_VERSION,
    temperature=0.2
)

# Function to handle LangChain response parsing
def parse_ai_response(content):
    try:
        result = json.loads(content)
        st.subheader("AI Analysis Result")

        if result["Result"].lower() == "phishing":
            st.markdown(f"**Result:** :red-badge[{result['Result']}]")
        else:
            st.markdown(f"**Result:** :green-badge[{result['Result']}]")

        st.markdown(f"**Confidence:** {result['Confidence']}")
        st.markdown(f"**Reason:** {result['Reason']}")
    except Exception as e:
        st.error(f"Failed to parse LLM response: {e}")
        st.text("Raw LLM output:")
        st.text(content or "No content.")

query = f"""
Analyze the following email and determine if it is a phishing attempt.

Respond in plain text format, however, it should look like a Python dictionary like this:

{{
"Result": "Phishing or Not Phishing",
"Confidence": "high/medium/low",
"Reason": "brief explanation"
}}

I need only the answer and don't add any prefixes.
"""

# Function to analyze the email content using Azure OpenAI
def analyze_email(email_content):
    try:
        messages = [
            SystemMessage(content="You are a cybersecurity expert trained to detect phishing emails."),
            HumanMessage(content= query + f"Email:\n{email_content}")
        ]
        response = chat(messages)
        content = response.content
        parse_ai_response(content)
    except Exception as e:
        st.error(f"Something went wrong: {e}")

# Streamlit GUI
st.set_page_config(page_title="Phishing Email Detector", layout="centered")
st.title("Phishing Email Detector")
st.markdown("Paste the email content below and get a classification.")

tab_textInput, tab_imageInput = st.tabs(["Enter Text", "Input Image"])

# TEXT ANALYSIS TAB
with tab_textInput:
    email_content = st.text_area("Email Content", height=300, placeholder="Paste email body here...")

    if st.button("Analyze Email"):
        if not email_content.strip():
            st.warning("Please paste email content first.")
        else:
            with st.spinner("Analyzing..."):
                analyze_email(email_content)

# IMAGE ANALYSIS TAB
with tab_imageInput:
    imageUpload = st.file_uploader("Choose image", type=["jpg", "png", "jpeg"])

    if imageUpload:
        image = Image.open(imageUpload)
        st.image(image, caption="Uploaded Image", use_container_width=True)

        if st.button("Analyze Image"):
            with st.spinner("Analyzing..."):
                try:
                    # Load CLIP model and processor
                    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
                    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")

                    # Preprocess image
                    inputs = processor(images=image, return_tensors="pt", padding=True) # type: ignore

                    # Extract image features
                    with torch.no_grad():
                        image_features = model.get_image_features(**inputs) # type: ignore

                    # Convert to NumPy
                    image_embedding = image_features.detach().cpu().numpy().astype(np.float32)

                    # Turn image content into a dummy document
                    docs = [Document(page_content="This is an image", metadata={})]

                    # Initialize embeddings with Azure OpenAI
                    openai_embeddings = AzureOpenAIEmbeddings(
                        model=AZURE_EMBEDDING_MODEL_NAME,
                        api_key=AZURE_OPENAI_API_KEY,
                        azure_deployment=AZURE_DEPLOYMENT_NAME,
                        azure_endpoint=f"https://{AZURE_RESOURCE_NAME}.openai.azure.com/",
                    )

                    # Create FAISS vectorstore from dummy document
                    vector_store = FAISS.from_documents(docs, openai_embeddings)

                    # Build conversational chain
                    chain = ConversationalRetrievalChain.from_llm(
                        llm=chat,
                        retriever=vector_store.as_retriever()
                    )

                    # Question input
                    user_input = query
                    if user_input:
                        response = chain.run(question=user_input, chat_history=[])
                        parse_ai_response(response)

                except Exception as e:
                    st.error(f"Failed to analyze the image: {e}")
