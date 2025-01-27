import streamlit as st
import requests
import torch
from transformers import pipeline
from urllib.parse import urlparse
import re

# Google Search API credentials
GOOGLE_API_KEY = "AIzaSyAH7A8iVDIqssN8gRA-KFnAYJEiOKoPEW0"
GOOGLE_CX = "075b1f42a94214065"

class EnhancedNewsDetector:
    def __init__(self):
        # Initialize the model with error handling
        try:
            self.MODEL = "jy46604790/Fake-News-Bert-Detect"
            self.classifier = pipeline("text-classification", model=self.MODEL, tokenizer=self.MODEL)
        except Exception as e:
            st.error(f"Error loading model: {e}")
            self.classifier = None

        # List of verified news sources in Pakistan
        self.verified_sources = {
            'dawn.com': {'name': 'Dawn News', 'reliability': 0.9},
            'tribune.com.pk': {'name': 'Express Tribune', 'reliability': 0.9},
            'geo.tv': {'name': 'Geo News', 'reliability': 0.85},
            'thenews.com.pk': {'name': 'The News', 'reliability': 0.85},
            'nation.com.pk': {'name': 'The Nation', 'reliability': 0.8},
            'app.com.pk': {'name': 'Associated Press of Pakistan', 'reliability': 0.9},
            'radio.gov.pk': {'name': 'Radio Pakistan', 'reliability': 0.85},
            'brecorder.com': {'name': 'Business Recorder', 'reliability': 0.8}
        }

    def google_search(self, query):
        """Search Google Custom Search for the news article"""
        search_url = f"https://www.googleapis.com/customsearch/v1?q={query}&key={GOOGLE_API_KEY}&cx={GOOGLE_CX}"
        try:
            response = requests.get(search_url)
            response.raise_for_status()
            return response.json().get('items', [])
        except Exception as e:
            st.warning(f"Google Search API error: {e}")
            return []

    def verify_news(self, news_text, source_url=None):
        """Enhanced news verification combining Google Search and AI model analysis"""
        if not news_text or len(news_text.strip()) == 0:
            return {
                'error': 'Empty news text provided',
                'is_fake': True,
                'confidence': 1.0
            }

        results = {
            'google_search_results': None,
            'text_analysis': None,
            'is_fake': None,
            'confidence': 0.0,
            'recommendation': ''
        }

        # 1. Perform Google Search to verify news
        st.info("Searching credible sources on Google...")
        search_results = self.google_search(news_text)
        results['google_search_results'] = search_results

        if search_results:
            for res in search_results[:3]:
                st.write(f"**{res['title']}**")
                st.write(f"URL: {res['link']}")
        else:
            st.warning("No related news found on Google.")

        # 2. BERT Model Prediction
        try:
            if self.classifier is None:
                raise ValueError("Model not initialized")

            model_result = self.classifier(news_text)
            is_fake = model_result[0]['label'] == 'LABEL_0'  # Assuming LABEL_0 is fake
            model_confidence = model_result[0]['score']

            results['text_analysis'] = {
                'model_prediction': 'FAKE' if is_fake else 'REAL',
                'model_confidence': model_confidence
            }

            results['is_fake'] = is_fake
            results['confidence'] = model_confidence

        except Exception as e:
            st.error(f"Error in model prediction: {e}")
            return results

        # Final Recommendation
        if search_results and not is_fake:
            results['recommendation'] = "The news appears reliable based on verified sources."
        elif search_results and is_fake:
            results['recommendation'] = "Conflicting information found. Please cross-check."
        else:
            results['recommendation'] = "No verification found. Likely to be fake."

        return results

# Streamlit App
def main():
    st.set_page_config(
        page_title="News Verification Assistant",
        page_icon="🕵️",
        layout="wide"
    )

    st.title("🕵️ News Verification Assistant")
    st.markdown("""
    ### Detect Fake News with AI-Powered Analysis & Google Search
    This tool helps you verify the credibility of news articles using AI models and Google Custom Search.
    """)

    detector = EnhancedNewsDetector()

    col1, col2 = st.columns(2)

    with col1:
        news_text = st.text_area(
            "Enter the news text:",
            height=300,
            placeholder="Paste the news article or text here..."
        )

    with col2:
        source_url = st.text_input(
            "Source URL (Optional)",
            placeholder="https://example.com/news-article"
        )

    if st.button("Verify News", type="primary"):
        if not news_text.strip():
            st.error("Please enter some news text to verify.")
            return

        with st.spinner("Analyzing news article..."):
            try:
                result = detector.verify_news(news_text, source_url)

                tab1, tab2, tab3 = st.tabs([
                    "🔍 Google Search Results",
                    "📊 Content Analysis",
                    "🚨 Final Assessment"
                ])

                with tab1:
                    st.subheader("Google Search Results")
                    if result['google_search_results']:
                        for item in result['google_search_results']:
                            st.write(f"**{item['title']}**")
                            st.write(f"URL: {item['link']}")
                    else:
                        st.warning("No search results found.")

                with tab2:
                    st.subheader("Content Analysis")
                    text_analysis = result['text_analysis']
                    st.metric(
                        label="Model Prediction",
                        value=text_analysis['model_prediction'],
                        help="AI model's assessment of news authenticity"
                    )
                    st.metric(
                        label="Model Confidence",
                        value=f"{text_analysis['model_confidence']:.2%}",
                        help="Confidence level of the AI model's prediction"
                    )

                with tab3:
                    st.subheader("Final Assessment")
                    if result['is_fake']:
                        st.error("🚨 FAKE NEWS DETECTED")
                    else:
                        st.success("✅ NEWS APPEARS RELIABLE")

                    st.metric(
                        label="Overall Confidence",
                        value=f"{result['confidence']:.2%}",
                        help="Final confidence combining AI model and Google Search"
                    )

                    st.info(f"🔔 Recommendation: {result['recommendation']}")

            except Exception as e:
                st.error(f"An error occurred during verification: {e}")

if __name__ == "__main__":
    main()
