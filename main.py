from langchain_openai import AzureChatOpenAI
from langchain.schema import SystemMessage, HumanMessage
import streamlit as st
from PIL import Image
import pytesseract
import json
import torch

# Optional: For BLIP captioning (if you want both OCR + visual context)
from transformers import BlipProcessor, BlipForConditionalGeneration

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

Respond in plain text format, however, it should look like a Python dictionary like this:

{
"Result": "Phishing or Not Phishing",
"Confidence": "high/medium/low",
"Reason": "brief explanation"
}

I need only the answer and don't add any prefixes.
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

        st.markdown(f"**Confidence:** {result['Confidence']}")
        st.markdown(f"**Reason:** {result['Reason']}")

    except Exception as e:
        st.error(f"Failed to parse LLM response: {e}")
        st.text("Raw LLM output:")
        st.text(content or "No content.")

# Function to analyze plain text
def analyze_email(email_content):
    try:
        messages = [
            SystemMessage(content="You are a cybersecurity expert trained to detect phishing emails."),
            HumanMessage(content=query + f"Email:\n{email_content}")
        ]
        response = chat(messages)
        content = response.content
        parse_ai_response(content)
    except Exception as e:
        st.error(f"Something went wrong: {e}")

# OCR Function
def extract_text_from_image(image: Image.Image) -> str:
    return pytesseract.image_to_string(image)

# Optional: BLIP Captioning (can be used along with OCR)
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
    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    imageUpload = st.file_uploader("Choose image", type=["jpg", "png", "jpeg"])

    if imageUpload:
        image = Image.open(imageUpload).convert("RGB")
        st.image(image, caption="Uploaded Image", use_container_width=True)

        if st.button("Analyze Image"):
            with st.spinner("Extracting text and analyzing..."):
                try:
                    # Step 1: OCR
                    ocr_text = extract_text_from_image(image)
                    st.markdown("**Extracted Text from Image:**")
                    st.code(ocr_text, language="text")

                    # Step 2: (Optional) Add BLIP description
                    # caption = generate_caption(image)
                    # combined_input = f"Visual Description: {caption}\nExtracted Email Text:\n{ocr_text}"

                    # Step 3: Analyze
                    messages = [
                        SystemMessage(content="You are a cybersecurity expert trained to detect phishing emails."),
                        HumanMessage(content=query + f"\nExtracted Email Text:\n{ocr_text}")
                        # OR if combining: HumanMessage(content=query + combined_input)
                    ]
                    response = chat(messages)
                    parse_ai_response(response.content)

                except Exception as e:
                    st.error(f"Failed to analyze the image: {e}")
