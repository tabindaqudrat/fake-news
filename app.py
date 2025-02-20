import streamlit as st
import torch
from transformers import pipeline
from urllib.parse import urlparse, quote_plus
import re
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

        # API keys for Google Custom Search (use environment variables for security)
        self.API_KEY = os.getenv("GOOGLE_API_KEY")
        self.SEARCH_ENGINE_ID = os.getenv("SEARCH_ENGINE_ID")
        
        # Verified news sources in Pakistan
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
                    f"&sort=date"
                )
                
                response = requests.get(url)
                if response.status_code == 200:
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
                elif response.status_code == 429:
                    time.sleep(backoff_time)
                    backoff_time *= 2
                else:
                    return {'error': f"API Error: {response.status_code}"}
            except requests.RequestException as e:
                return {'error': str(e)}
        return {'error': "Max retries reached. Google API may be unavailable."}

# Streamlit App
def main():
    st.set_page_config(page_title="AI-Powered Fake News Detection", page_icon="🕵️", layout="wide")
    st.title("🕵️ AI-Powered Fake News Detection System")
    st.markdown("""
    ### Detect Fake News with AI-Powered Analysis
    This tool verifies news credibility using AI and trusted news sources.
    """)
    
    detector = EnhancedNewsDetector()
    col1, col2 = st.columns(2)
    with col1:
        news_text = st.text_area("Enter the news text:", height=300, placeholder="Paste the news article here...")
    with col2:
        source_url = st.text_input("Source URL (Optional)", placeholder="https://example.com/news-article")
    
    if st.button("Verify News", type="primary"):
        if not news_text.strip():
            st.error("Please enter some news text to verify.")
            return
        with st.spinner("Analyzing news article..."):
            result = detector.custom_search_verification(news_text)
            st.markdown("---")
            col_verdict, col_confidence = st.columns(2)
            with col_verdict:
                if result.get('verified', False):
                    st.success("✅ NEWS APPEARS RELIABLE")
                else:
                    st.error("🚨 FAKE NEWS DETECTED")
            
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

if __name__ == "__main__":
    main()
