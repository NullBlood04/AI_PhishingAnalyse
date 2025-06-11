from transformers import BlipProcessor, BlipForConditionalGeneration
from langchain.schema import SystemMessage, HumanMessage
from langchain_openai import AzureChatOpenAI
import streamlit as st
from PIL import Image
import numpy as np
import easyocr
import torch
import json

# Load from secrets
AZURE_OPENAI_API_KEY = st.secrets["AZURE_OPENAI_API_KEY"]
AZURE_DEPLOYMENT_NAME = st.secrets["AZURE_DEPLOYMENT_NAME"]
AZURE_RESOURCE_NAME = st.secrets["AZURE_RESOURCE_NAME"]
AZURE_API_VERSION = st.secrets["AZURE_API_VERSION"]

# Initialize LangChain Azure OpenAI
chat = AzureChatOpenAI(
    azure_deployment=AZURE_DEPLOYMENT_NAME,  # type: ignore
    azure_endpoint=f"https://{AZURE_RESOURCE_NAME}.openai.azure.com/",
    api_key=AZURE_OPENAI_API_KEY,
    api_version=AZURE_API_VERSION,
    temperature=0.2
)

# Prompt
query = """
Analyze the following email and determine if it is a phishing attempt.
Take into consideration there my be spelling mistakes made by the text extractor, so give your output accordingly
Respond in plain text format, however, it should look like a Python dictionary like this:

{
"Result": "Phishing or Not Phishing",
"Confidence": "high/medium/low",
"Reason": "brief explanation"
}

I need only the answer and don't add any prefixes.
"""

system_message = """
You are a cybersecurity expert. Your only task is to evaluate emails and determine if they are phishing attempts.
Respond ONLY with a valid JSON object with these keys:
- "Result": "Phishing" or "Not Phishing"
- "Confidence": "high", "medium", or "low"
- "Reason": A short sentence explaining your decision

Do not add any explanations, formatting, or extra text outside the JSON.
"""


# Function to parse and display result
def parse_ai_response(content):
    try:
        result = json.loads(content)
        st.subheader("AI Analysis Result")

        if result["Result"].lower() == "phishing":
            st.markdown(f"**Result:** :red-badge[{result['Result']}]")
        else:
            st.markdown(f"**Result:** :green-badge[{result['Result']}]")

        confidence = result["Confidence"].lower()
        if confidence == "high":
            st.markdown("**Confidence:** :green-badge[High]")
        elif confidence == "medium":
            st.markdown("**Confidence:** :orange-badge[Medium]")
        else:
            st.markdown("**Confidence:** :red-badge[Low]")

        st.markdown(f"**Reason:** {result['Reason']}")

    except Exception as e:
        st.error(f"Failed to parse LLM response: {e}")
        st.text("Raw LLM output:")
        st.text(content or "No content.")

# Function to analyze plain text
def analyze_email(email_content):
    try:
        messages = [
            SystemMessage(content=system_message),
            HumanMessage(content=query + f"Email:\n{email_content}")
        ]
        response = chat(messages)
        content = response.content
        parse_ai_response(content)
    except Exception as e:
        st.error(f"Something went wrong: {e}")

# OCR Function
reader = easyocr.Reader(['en'])

def extract_text_ai_ocr(image: Image.Image) -> str:
    image_np = np.array(image)

    results = reader.readtext(image_np, detail=0)  # detail=0 returns just the text
    return "\n".join(results) # type: ignore

# BLIP Captioning (To be used along with OCR)
@st.cache_resource
def load_blip_model():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    return processor, model

def generate_caption(image):
    processor, model = load_blip_model()
    inputs = processor(images=image, return_tensors="pt")  # type: ignore
    with torch.no_grad():
        out = model.generate(**inputs)  # type: ignore
    caption = processor.decode(out[0], skip_special_tokens=True)  # type: ignore
    return caption

# Streamlit GUI
st.set_page_config(page_title="Phishing Email Detector", layout="centered")
st.title("Phishing Email Detector")
st.markdown("Paste the email content below or upload a screenshot image of the email.")

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
        image = Image.open(imageUpload).convert("RGB")
        st.image(image, caption="Uploaded Image", use_container_width=True)

        if st.button("Analyze Image"):
            with st.spinner("Extracting text and analyzing..."):
                try:
                    # Step 1: OCR
                    ocr_text = extract_text_ai_ocr(image)
                    st.markdown("**Extracted Text from Image:**")
                    st.code(ocr_text, language="text")

                    # Step 2: Add BLIP description
                    caption = generate_caption(image)
                    combined_input = f"Visual Description: {caption}\nExtracted Email Text:\n{ocr_text}"

                    # Step 3: Analyze
                    messages = [
                        SystemMessage(content=system_message),
                        HumanMessage(content=query + combined_input)
                    ]
                    response = chat(messages)
                    parse_ai_response(response.content)

                except Exception as e:
                    st.error(f"Failed to analyze the image: {e}")
