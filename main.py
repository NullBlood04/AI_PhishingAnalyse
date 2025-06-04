import streamlit as st
from langchain_openai import AzureChatOpenAI
from langchain.schema import SystemMessage, HumanMessage
import json

# Load from secrets
AZURE_OPENAI_API_KEY = st.secrets["AZURE_OPENAI_API_KEY"]
AZURE_DEPLOYMENT_NAME = st.secrets["AZURE_DEPLOYMENT_NAME"]
AZURE_RESOURCE_NAME = st.secrets["AZURE_RESOURCE_NAME"]
AZURE_API_VERSION = st.secrets["AZURE_API_VERSION"]

# Initialize LangChain Azure OpenAI
chat = AzureChatOpenAI(
    azure_deployment=AZURE_DEPLOYMENT_NAME,
    azure_endpoint=f"https://{AZURE_RESOURCE_NAME}.openai.azure.com/",
    api_key=AZURE_OPENAI_API_KEY,
    api_version=AZURE_API_VERSION,
    temperature=0.2
)

# Streamlit GUI
st.set_page_config(page_title="Phishing Email Detector", layout="centered")
st.title("📧 Phishing Email Detector")
st.markdown("Paste the email content below and get a plain-text classification.")

email_content = st.text_area("Email Content", height=300, placeholder="Paste email body here...")

if st.button("Analyze Email"):
    if not email_content.strip():
        st.warning("Please paste email content first.")
    else:
        with st.spinner("Analyzing..."):
            try:
                messages = [
                    SystemMessage(content="You are a cybersecurity expert trained to detect phishing emails."),
                    HumanMessage(content=f"""
Analyze the following email and determine if it is a phishing attempt.

Respond in plane text format however it should look python dictionary like this:
                                 
{{
"Result": Phishing or Not Phishing  
"Confidence": high/medium/low  
"Reason": brief explanation
}}
                                 
Email:
{email_content}
""")
                ]

                response = chat(messages)
                content = response.content

                st.subheader("🧠 AI Analysis Result")
                try:
                    result = json.loads(str(content))
                    if result["Result"].lower() == "phishing":
                        st.badge(result["Result"], color="red")
                    else:
                        st.badge(result["Result"])

                    st.badge(result["Confidence"])
                    st.markdown(result["Reason"])

                except Exception as e:
                    st.error(f"Failed to parse LLM response: {e}")
                    st.text("Raw LLM output:")
                    st.text(content or "No content.")

            except Exception as e:
                st.error(f"Something went wrong: {e}")


