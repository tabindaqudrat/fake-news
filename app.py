import streamlit as st
import torch
from transformers import pipeline
from urllib.parse import urlparse, quote_plus
import requests
import time
import os

class EnhancedNewsDetector:
    def __init__(self):
        try:
            self.MODEL = "jy46604790/Fake-News-Bert-Detect"
            self.classifier = pipeline("text-classification", model=self.MODEL, tokenizer=self.MODEL)
        except Exception as e:
            st.error(f"Error loading model: {e}")
            self.classifier = None

        self.API_KEY = st.secrets["GOOGLE_API_KEY"]
        self.SEARCH_ENGINE_ID = st.secrets["SEARCH_ENGINE_ID"]

        if not self.API_KEY or not self.SEARCH_ENGINE_ID:
            raise ValueError("Environment variables GOOGLE_API_KEY and SEARCH_ENGINE_ID must be set.")

        self.verified_sources = {
            'dawn.com': {'name': 'Dawn News', 'reliability': 0.9},
            'tribune.com.pk': {'name': 'Express Tribune', 'reliability': 0.9},
            'geo.tv': {'name': 'Geo News', 'reliability': 0.85},
            'thenews.com.pk': {'name': 'The News', 'reliability': 0.85},
            'nation.com.pk': {'name': 'The Nation', 'reliability': 0.8},
        }
    def predict_news(self, news_text):  # This function MUST be in your class
        if self.classifier is None:
            return None

        try:
            result = self.classifier(news_text)[0]
            label = result['label']
            score = result['score']
            return {'label': label, 'score': score}
        except Exception as e:
            st.error(f"Error during prediction: {e}")
            return None
            
    def custom_search_verification(self, news_text):
        max_retries = 3
        backoff_time = 2
        search_query = quote_plus(news_text[:100])

        for attempt in range(max_retries):
            try:
                url = (
                    f"https://www.googleapis.com/customsearch/v1"
                    f"?key={self.API_KEY}"
                    f"&cx={self.SEARCH_ENGINE_ID}"
                    f"&q={search_query}"
                    f"&alt=json"
                )

                response = requests.get(url)
                response.raise_for_status()

                search_data = response.json()
                results = []
                trusted_sources = []
                searched_domains = set()

                if 'items' in search_data:
                    for item in search_data['items'][:5]:
                        result_domain = urlparse(item['link']).netloc.lower()
                        if result_domain.startswith('www.'):
                            result_domain = result_domain[4:]
                        searched_domains.add(result_domain)
                        is_trusted = result_domain in self.verified_sources
                        result_info = {
                            'title': item.get('title', 'No Title'),
                            'link': item.get('link', ''),
                            'snippet': item.get('snippet', 'No Snippet'),
                            'source': result_domain,
                            'is_trusted': is_trusted
                        }
                        results.append(result_info)
                        if is_trusted and result_domain not in [s['domain'] for s in trusted_sources]:
                            trusted_sources.append({
                                'domain': result_domain,
                                'name': self.verified_sources[result_domain]['name'],
                                'reliability': self.verified_sources[result_domain]['reliability']
                            })
                return {
                    'verified': len(trusted_sources) > 0,
                    'trusted_sources': trusted_sources,
                    'search_results': results,
                    'searched_domains': list(searched_domains)
                }

            except requests.exceptions.RequestException as e:
                if response.status_code == 429:
                    time.sleep(backoff_time)
                    backoff_time *= 2
                    print("Rate limiting error")
                else:
                    return {'error': str(e)}
            except Exception as e:
                return {'error': str(e)}

        return {'error': "Max retries reached. API may be unavailable."}


def main():
    st.set_page_config(page_title="AI-Powered Fake News Detection", page_icon="🕵️", layout="wide")
    st.title("🕵️ AI-Powered Fake News Detection System")

    news_text = st.text_area("Enter the news article text to verify:", height=200)
    add_source = st.checkbox("Add News Source (Optional)")

    news_source = ""
    if add_source:
        news_source = st.text_input("Enter the news source:")

    detector = EnhancedNewsDetector()

    if detector.classifier is None:
        st.error("Model loading failed. Please check your model path and dependencies.")
        return

    if st.button("Verify News", type="primary"):
        if not news_text.strip():
            st.error("Please enter some news text to verify.")
            return

        with st.spinner("Analyzing news article..."):
            try:
                model_prediction = detector.predict_news(news_text)
                search_result = detector.custom_search_verification(news_text)

                if model_prediction:
                    st.markdown("## 🤖 AI Model Prediction")
                    label = model_prediction['label']
                    score = model_prediction['score']
                    st.write(f"**Prediction:** {label}")
                    st.write(f"**Confidence:** {score:.2f}")

                    if label == "FAKE":
                        st.error("This news is classified as FAKE by the AI model.")
                    else:
                        st.success("This news is classified as REAL by the AI model.")

                if 'error' in search_result:
                    st.error(f"Search API Error: {search_result['error']}")
                else:
                    st.markdown("---")
                    st.markdown("## 🌐 Sources Checked During Verification")
                    searched_domains = search_result.get('searched_domains', [])
                    if searched_domains:
                        for domain in searched_domains:
                            if domain in detector.verified_sources:
                                st.write(f"✅ **{detector.verified_sources[domain]['name']}** ({domain})")
                            else:
                                st.write(f"⚠️ {domain} (Not in verified list)")
                    else:
                        st.warning("No sources found in the search results.")

                    if 'trusted_sources' in search_result and search_result['trusted_sources']:
                        st.success("✅ Verified by Trusted Sources")
                        for source in search_result['trusted_sources']:
                            st.write(f"- **{source['name']}** ({source['domain']}), Reliability: {source['reliability']:.2f}")
                    else:
                        st.warning("⚠️ No trusted sources verified this news.")

                    if 'search_results' in search_result:
                        st.markdown("## 🔍 Top Search Results")
                        for idx, res in enumerate(search_result['search_results']):
                            with st.expander(f"{idx+1}. {res['title']}"):
                                st.write(f"**Source**: {res['source']}")
                                st.write(f"**Snippet**: {res['snippet']}")
                                st.write(f"**Link**: {res['link']}")

                    # Optional News Source Tab
                    if add_source and news_source:
                        st.markdown("---")
                        st.markdown("## 📰 Provided News Source")
                        st.write(news_source)

            except Exception as e:
                st.exception(e)


if __name__ == "__main__":
    main()
