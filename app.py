import streamlit as st
import requests
from bs4 import BeautifulSoup
from transformers import pipeline
from urllib.parse import urlparse
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Google Search API credentials (replace with your own)
GOOGLE_API_KEY = "YOUR_GOOGLE_API_KEY"
GOOGLE_CX = "YOUR_CSE_ID"

class EnhancedNewsDetector:
    def __init__(self):
        # Initialize the fake news detection model
        try:
            self.MODEL = "jy46604790/Fake-News-Bert-Detect"
            self.classifier = pipeline("text-classification", model=self.MODEL, tokenizer=self.MODEL)
        except Exception as e:
            st.error(f"Error loading model: {e}")
            self.classifier = None

    def google_search(self, query):
        """Perform Google search and fetch credible news articles."""
        search_url = f"https://www.googleapis.com/customsearch/v1?q={query}&key={GOOGLE_API_KEY}&cx={GOOGLE_CX}"
        try:
            response = requests.get(search_url)
            response.raise_for_status()
            search_results = response.json().get('items', [])

            verified_results = []
            for item in search_results[:5]:  # Analyze top 5 results
                page_content = self.extract_page_content(item['link'])
                similarity_score = self.calculate_similarity(query, page_content)

                if similarity_score > 0.6:  # Similarity threshold
                    verified_results.append({
                        'title': item['title'],
                        'url': item['link'],
                        'similarity_score': similarity_score
                    })

            return verified_results
        except Exception as e:
            st.warning(f"Google Search API error: {e}")
            return []

    def extract_page_content(self, url):
        """Extract text content from a news article."""
        try:
            headers = {'User-Agent': 'Mozilla/5.0'}
            response = requests.get(url, headers=headers)
            soup = BeautifulSoup(response.text, 'html.parser')

            paragraphs = soup.find_all('p')
            text_content = ' '.join([para.get_text() for para in paragraphs])
            return text_content
        except Exception as e:
            st.warning(f"Error extracting content from {url}: {e}")
            return ""

    def calculate_similarity(self, input_text, source_text):
        """Calculate cosine similarity between input news and extracted content."""
        if not source_text:
            return 0.0

        vectorizer = TfidfVectorizer().fit_transform([input_text, source_text])
        vectors = vectorizer.toarray()
        similarity = cosine_similarity([vectors[0]], [vectors[1]])[0][0]

        return similarity

    def verify_news(self, news_text):
        """Enhanced verification combining Google Search and AI model analysis."""
        if not news_text.strip():
            return {
                'error': 'Empty news text provided',
                'is_fake': True,
                'confidence': 0.0
            }

        st.info("🔍 Searching credible sources on Google...")
        search_results = self.google_search(news_text)
        verified_sources = [res for res in search_results if res['similarity_score'] > 0.6]

        if verified_sources:
            result_status = "Likely Real"
            confidence = max(res['similarity_score'] for res in verified_sources)
            recommendation = "This news matches credible sources."
        else:
            result_status = "Likely Fake"
            confidence = 0.3
            recommendation = "No verification found. Likely to be fake."

        # AI Model Prediction
        if self.classifier:
            model_result = self.classifier(news_text)
            ai_prediction = 'FAKE' if model_result[0]['label'] == 'LABEL_0' else 'REAL'
            ai_confidence = model_result[0]['score']
        else:
            ai_prediction = 'Unknown'
            ai_confidence = 0.0

        return {
            'result_status': result_status,
            'confidence': confidence,
            'google_verified_links': verified_sources,
            'ai_prediction': ai_prediction,
            'ai_confidence': ai_confidence,
            'recommendation': recommendation
        }

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
            result = detector.verify_news(news_text)

            tab1, tab2, tab3 = st.tabs([
                "🔍 Google Search Results",
                "📊 Content Analysis",
                "🚨 Final Assessment"
            ])

            with tab1:
                st.subheader("Google Search Results")
                if result['google_verified_links']:
                    for item in result['google_verified_links']:
                        st.write(f"**{item['title']}**")
                        st.write(f"URL: {item['url']}")
                        st.write(f"Similarity Score: {item['similarity_score']:.2%}")
                else:
                    st.warning("No search results found.")

            with tab2:
                st.subheader("Content Analysis")
                st.metric(
                    label="AI Model Prediction",
                    value=result['ai_prediction'],
                    help="AI model's assessment of news authenticity"
                )
                st.metric(
                    label="Model Confidence",
                    value=f"{result['ai_confidence']:.2%}",
                    help="Confidence level of the AI model's prediction"
                )

            with tab3:
                st.subheader("Final Assessment")
                if result['result_status'] == "Likely Fake":
                    st.error("🚨 FAKE NEWS DETECTED")
                else:
                    st.success("✅ NEWS APPEARS RELIABLE")

                st.metric(
                    label="Overall Confidence",
                    value=f"{result['confidence']:.2%}",
                    help="Final confidence combining AI model and Google Search"
                )

                st.info(f"🔔 Recommendation: {result['recommendation']}")

if __name__ == "__main__":
    main()
