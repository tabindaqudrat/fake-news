import streamlit as st
import requests
from bs4 import BeautifulSoup
from transformers import pipeline
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Google Search API credentials (replace with your own)
GOOGLE_API_KEY = "AIzaSyAH7A8iVDIqssN8gRA-KFnAYJEiOKoPEW0"
GOOGLE_CX = "075b1f42a94214065"

class EnhancedNewsDetector:
    def __init__(self):
        try:
            # Initialize BERT model for fake news classification
            self.BERT_MODEL = "jy46604790/Fake-News-Bert-Detect"
            self.classifier = pipeline("text-classification", model=self.BERT_MODEL, tokenizer=self.BERT_MODEL)
            
            # Initialize SentenceTransformer for better similarity calculation
            self.sentence_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
        except Exception as e:
            st.error(f"Error loading models: {e}")
            self.classifier = None
            self.sentence_model = None

    def google_search(self, query):
        """Perform Google search and fetch credible news articles."""
        search_url = f"https://www.googleapis.com/customsearch/v1?q={query}&key={GOOGLE_API_KEY}&cx={GOOGLE_CX}"
        try:
            response = requests.get(search_url)
            response.raise_for_status()
            search_results = response.json().get('items', [])

            verified_results = []
            # Get query embedding
            query_embedding = self.sentence_model.encode([query])[0]

            for item in search_results[:5]:  # Analyze top 5 results
                page_content = self.extract_page_content(item['link'])
                if not page_content:
                    continue

                similarity_score = self.calculate_semantic_similarity(query_embedding, page_content)

                if similarity_score > 0.7:  # Increased threshold for stricter matching
                    verified_results.append({
                        'title': item['title'],
                        'url': item['link'],
                        'similarity_score': similarity_score,
                        'content': page_content[:500]  # Store first 500 chars for display
                    })

            return verified_results
        except Exception as e:
            st.warning(f"Google Search API error: {e}")
            return []

    def extract_page_content(self, url):
        """Extract text content from a news article."""
        try:
            headers = {'User-Agent': 'Mozilla/5.0'}
            response = requests.get(url, headers=headers, timeout=10)
            soup = BeautifulSoup(response.text, 'html.parser')

            # Remove script and style elements
            for script in soup(['script', 'style']):
                script.decompose()

            # Get text from article tags first, if present
            article_text = ''
            article = soup.find('article')
            if article:
                article_text = article.get_text(separator=' ', strip=True)

            # If no article tag, get text from p tags
            if not article_text:
                paragraphs = soup.find_all('p')
                article_text = ' '.join([para.get_text(strip=True) for para in paragraphs])

            return article_text
        except Exception as e:
            st.warning(f"Error extracting content from {url}: {e}")
            return ""

    def calculate_semantic_similarity(self, query_embedding, source_text):
        """Calculate semantic similarity using SentenceTransformer."""
        try:
            # Get embedding for source text
            source_embedding = self.sentence_model.encode([source_text])[0]
            
            # Calculate cosine similarity
            similarity = cosine_similarity(
                query_embedding.reshape(1, -1), 
                source_embedding.reshape(1, -1)
            )[0][0]
            
            return float(similarity)
        except Exception as e:
            st.warning(f"Error calculating similarity: {e}")
            return 0.0

    def verify_news(self, news_text):
        """Enhanced verification prioritizing Google Search results."""
        if not news_text.strip():
            return {
                'error': 'Empty news text provided',
                'is_fake': True,
                'confidence': 0.0
            }

        st.info("🔍 Searching credible sources...")
        search_results = self.google_search(news_text)
        
        # Priority 1: Check exact matches in credible sources
        exact_matches = [res for res in search_results if res['similarity_score'] > 0.85]
        
        # Priority 2: Check high similarity matches
        high_similarity = [res for res in search_results if 0.7 <= res['similarity_score'] <= 0.85]

        # Determine result based on matches
        if exact_matches:
            result_status = "Verified Real"
            confidence = max(res['similarity_score'] for res in exact_matches)
            recommendation = "This news exactly matches credible sources."
        elif high_similarity:
            result_status = "Likely Real"
            confidence = max(res['similarity_score'] for res in high_similarity)
            recommendation = "This news is similar to content from credible sources."
        else:
            result_status = "Potentially Fake"
            confidence = 0.3
            recommendation = "No matching content found in credible sources."

        # Use BERT model as secondary verification
        if self.classifier:
            model_result = self.classifier(news_text)
            ai_prediction = 'FAKE' if model_result[0]['label'] == 'LABEL_0' else 'REAL'
            ai_confidence = model_result[0]['score']
            
            # Adjust final confidence based on AI prediction
            if (result_status == "Potentially Fake" and ai_prediction == 'FAKE') or \
               (result_status in ["Verified Real", "Likely Real"] and ai_prediction == 'REAL'):
                confidence = (confidence + ai_confidence) / 2
        else:
            ai_prediction = 'Unknown'
            ai_confidence = 0.0

        return {
            'result_status': result_status,
            'confidence': confidence,
            'google_verified_links': search_results,
            'ai_prediction': ai_prediction,
            'ai_confidence': ai_confidence,
            'recommendation': recommendation
        }

# Streamlit App
def main():
    st.set_page_config(
        page_title="Enhanced News Verification Assistant",
        page_icon="🕵️",
        layout="wide"
    )

    st.title("🕵️ Enhanced News Verification Assistant")
    st.markdown("""
    ### Advanced Fake News Detection with Semantic Analysis
    This tool verifies news credibility using semantic similarity matching and AI analysis.
    """)

    detector = EnhancedNewsDetector()

    col1, col2 = st.columns([2, 1])

    with col1:
        news_text = st.text_area(
            "Enter the news text:",
            height=300,
            placeholder="Paste the news article or text here..."
        )

    with col2:
        st.info("""
        ### How it works:
        1. Searches credible news sources
        2. Performs semantic similarity analysis
        3. Verifies content authenticity
        4. Uses AI to double-check results
        """)

    if st.button("Verify News", type="primary"):
        if not news_text.strip():
            st.error("Please enter some news text to verify.")
            return

        with st.spinner("Analyzing news content..."):
            result = detector.verify_news(news_text)

            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Verification Results")
                if result['result_status'] == "Verified Real":
                    st.success(f"✅ {result['result_status']}")
                elif result['result_status'] == "Likely Real":
                    st.info(f"ℹ️ {result['result_status']}")
                else:
                    st.error(f"🚨 {result['result_status']}")

                st.metric(
                    label="Overall Confidence",
                    value=f"{result['confidence']:.2%}",
                    help="Combined confidence score"
                )
                st.info(f"🔔 {result['recommendation']}")

            with col2:
                st.subheader("AI Model Analysis")
                st.metric(
                    label="AI Prediction",
                    value=result['ai_prediction'],
                    delta=f"{result['ai_confidence']:.2%}"
                )

            st.subheader("Matching Sources")
            if result['google_verified_links']:
                for idx, item in enumerate(result['google_verified_links'], 1):
                    with st.expander(f"Source {idx}: {item['title']}"):
                        st.write(f"**URL:** {item['url']}")
                        st.write(f"**Similarity Score:** {item['similarity_score']:.2%}")
                        st.write("**Preview:**")
                        st.write(item['content'])
            else:
                st.warning("No matching sources found in credible news outlets.")

if __name__ == "__main__":
    main()
