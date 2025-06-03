import streamlit as st
import requests
import json
from langchain_openai import AzureChatOpenAI
from langchain.schema import SystemMessage, HumanMessage

# ------------------ Configuration ------------------

# API keys and deployment settings

NEWS_API_KEY = st.secrets["NEWS_API_KEY"]
AZURE_OPENAI_API_KEY = st.secrets["AZURE_OPENAI_API_KEY"]
AZURE_DEPLOYMENT_NAME = st.secrets["AZURE_DEPLOYMENT_NAME"]
AZURE_RESOURCE_NAME = st.secrets["AZURE_RESOURCE_NAME"]
AZURE_API_VERSION = st.secrets["AZURE_API_VERSION"]


# ------------------ News Fetching ------------------

def fetch_news(topic, max_articles=20):
    url = "https://newsapi.org/v2/everything"
    params = {
        "q": topic,
        "apiKey": NEWS_API_KEY,
        "pageSize": max_articles,
        "sortBy": "publishedAt",
        "language": "en"
    }
    response = requests.get(url, params=params)
    response.raise_for_status()
    return response.json().get("articles", [])

# ------------------ Summarization with LangChain ------------------

def summarize_articles(topic, articles):
    basic_articles = [
        {"title": a["title"], "description": a["description"], "url": a["url"]}
        for a in articles
    ]

    prompt = f"""
You are an AI assistant. A user is interested in news about "{topic}".

Here are some articles. Rank them by relevance to the topic and provide a summary of about 200 words for each of the top 5 most relevant ones.

Respond with JSON format:
[
  {{
    "title": "...",
    "url": "...",
    "summary": "200-word summary..."
  }},
  ...
]

Articles:
{json.dumps(basic_articles, indent=2)}
"""

    chat = AzureChatOpenAI(
        azure_deployment = AZURE_DEPLOYMENT_NAME, 
        azure_endpoint = f"https://{AZURE_RESOURCE_NAME}.openai.azure.com/",
        api_key = AZURE_OPENAI_API_KEY,
        api_version = AZURE_API_VERSION,
        temperature = 0.5
    )

    messages = [
        SystemMessage(content="You help filter and summarize news articles."),
        HumanMessage(content=prompt)
    ]

    response = chat(messages)

    """ try:
        content = response.content
    except AttributeError:
        # If it's a ChatResult with messages[0]
        content = response.content if hasattr(response, "messages") else None """
    
    content = response.content

    try:
        return json.loads(str(content))
    except Exception as e:
        st.error(f"Failed to parse LLM response: {e}")
        st.text("Raw LLM output:")
        st.text(content or "No content.")
        return []


# ------------------ Streamlit UI ------------------

st.set_page_config(page_title="LangChain News Summarizer", layout="centered")

st.title("🧠 AI-Powered News Summarizer (LangChain)")
st.markdown("Get the latest and most relevant news articles on any topic — summarized intelligently using LangChain and Azure OpenAI.")

topic = st.text_input("Enter a topic", placeholder="e.g. climate change, AI, Ukraine war")

if st.button("Fetch & Summarize News"):
    if not topic.strip():
        st.warning("Please enter a valid topic.")
    else:
        with st.spinner("Fetching and summarizing news..."):
            try:
                articles = fetch_news(topic)
                if not articles:
                    st.info("No news articles found.")
                else:
                    summaries = summarize_articles(topic, articles)
                    for article in summaries:
                        st.subheader(article["title"])
                        st.write(article["summary"])
                        st.markdown(f"[Read full article]({article['url']})")
                        st.markdown("---")
            except Exception as e:
                st.error(f"Error: {e}")
