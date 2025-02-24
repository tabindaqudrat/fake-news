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

        self.verified_sources = {  # Replace with your actual verified sources
            "bbc.com": {"name": "BBC News", "reliability": 0.95},
            "cnn.com": {"name": "CNN", "reliability": 0.90},
            # ... add more verified sources
        }

    def custom_search_verification(self, news_text):
        max_retries = 3
        backoff_time = 2
        search_query = quote_plus(news_text[:100])  # Encode the query

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
                response.raise_for_status()  # Check for HTTP errors

                search_data = response.json()
                results = []
                trusted_sources = []
                searched_domains = set()

                if 'items' in search_data:
                    for item in search_data['items'][:5]:  # Limit to top 5 results
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
                if response.status_code == 429:  # Rate limiting
                    time.sleep(backoff_time)
                    backoff_time *= 2
                    print("Rate limiting error")
                else:
                    return {'error': str(e)}  # Other request error
            except Exception as e:
                return {'error': str(e)}  # Other error (JSON, etc.)

        return {'error': "Max retries reached. API may be unavailable."}  # Max retries reached


def main():
    st.set_page_config(page_title="AI-Powered Fake News Detection", page_icon="🕵️", layout="wide")
    st.title("🕵️ AI-Powered Fake News Detection System")

    news_text = st.text_area("Enter the news article text to verify:", height=200)

    detector = EnhancedNewsDetector()

    if detector.classifier is None:  # Check if model loaded successfully
        st.error("Model loading failed. Please check your model path and dependencies.")
        return  # Stop execution if model loading failed

    if st.button("Verify News", type="primary"):
        if not news_text.strip():
            st.error("Please enter some news text to verify.")
            return

        with st.spinner("Analyzing news article..."):
            try:
                result = detector.custom_search_verification(news_text)
                if 'error' in result:
                    st.error(f"Search API Error: {result['error']}")
                else:
                    # ... (Display results in Streamlit)
                    st.markdown("---")
                    st.markdown("## 🌐 Sources Checked During Verification")
                    searched_domains = result.get('searched_domains', [])
                    if searched_domains:
                        for domain in searched_domains:
                            if domain in detector.verified_sources:
                                st.write(f"✅ **{detector.verified_sources[domain]['name']}** ({domain})")
                            else:
                                st.write(f"⚠️ {domain} (Not in verified list)")
                    else:
                        st.warning("No sources found in the search results.")

                    if 'trusted_sources' in result and result['trusted_sources']:
                        st.success("✅ Verified by Trusted Sources")
                        for source in result['trusted_sources']:
                            st.write(f"- **{source['name']}** ({source['domain']}), Reliability: {source['reliability']:.2f}")
                    else:
                        st.warning("⚠️ No trusted sources verified this news.")

                    if 'search_results' in result:
                        st.markdown("## 🔍 Top Search Results")
                        for idx, res in enumerate(result['search_results']):
                            with st.expander(f"{idx+1}. {res['title']}"):
                                st.write(f"**Source**: {res['source']}")
                                st.write(f"**Snippet**: {res['snippet']}")
                                st.write(f"**Link**: {res['link']}")

            except Exception as e:
                st.exception(e)  # Show detailed error for debugging


if __name__ == "__main__":
    main()
