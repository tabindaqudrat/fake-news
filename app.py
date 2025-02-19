import streamlit as st
import torch
from transformers import pipeline
from urllib.parse import urlparse
import re
import requests

class EnhancedNewsDetector:
    def __init__(self):
        # Initialize the model with error handling
        try:
            self.MODEL = "jy46604790/Fake-News-Bert-Detect"
            self.classifier = pipeline("text-classification", model=self.MODEL, tokenizer=self.MODEL)
        except Exception as e:
            st.error(f"Error loading model: {e}")
            self.classifier = None

        # API keys for Google Custom Search
        self.API_KEY = "AIzaSyAH7A8iVDIqssN8gRA-KFnAYJEiOKoPEW0"  
        self.SEARCH_ENGINE_ID = "075b1f42a94214065"  

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

        # Known fake news or unreliable sources
        self.unreliable_sources = [
            'facebook.com', 'whatsapp.com', 'telegram.org',
            'wordpress.com', 'blogspot.com', 'medium.com'
        ]

    def extract_source_from_text(self, text):
        """Extract potential source URLs from the text"""
        urls = re.findall(r'https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+[^\s]*', text)
        if not urls:
            # Look for website mentions without http/https
            domain_patterns = re.findall(r'(?:[\w-]+\.)+(?:com|pk|org|net|gov)', text)
            urls.extend([f"http://{domain}" for domain in domain_patterns])
        return urls

    def custom_search_verification(self, news_text):
        """Use Google Custom Search API to check news against trusted sources"""
        try:
            # Extract key terms from news text (first 100 characters)
            search_query = news_text[:100] if len(news_text) > 100 else news_text
            
            # Make request to Google Custom Search API
            url = f"https://www.googleapis.com/customsearch/v1?key={self.API_KEY}&cx={self.SEARCH_ENGINE_ID}&q={search_query}"
            response = requests.get(url)
            
            if response.status_code != 200:
                return {
                    'verified': False,
                    'error': f"API Error: {response.status_code}",
                    'trusted_sources': [],
                    'search_results': []
                }
            
            # Process search results
            search_data = response.json()
            results = []
            trusted_sources = []
            
            # Check if we have search results
            if 'items' in search_data and len(search_data['items']) > 0:
                for item in search_data['items'][:5]:  # Limit to top 5 results
                    # Extract domain from result URL
                    result_domain = urlparse(item['link']).netloc.lower()
                    if result_domain.startswith('www.'):
                        result_domain = result_domain[4:]
                    
                    # Check if result is from a trusted source
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
                    'search_results': results
                }
            else:
                return {
                    'verified': False,
                    'trusted_sources': [],
                    'search_results': [],
                    'message': "No search results found"
                }
                
        except Exception as e:
            st.warning(f"Error in custom search verification: {str(e)}")
            return {
                'verified': False,
                'error': str(e),
                'trusted_sources': [],
                'search_results': []
            }

    def analyze_source(self, text, urls=None):
        """Analyze the credibility of the news source"""
        if urls is None:
            urls = self.extract_source_from_text(text)

        source_analysis = {
            'is_verified': False,
            'source_name': 'Unknown',
            'source_url': '',
            'reliability_score': 0.0,
            'warning_flags': []
        }

        if not urls:
            source_analysis['warning_flags'].append("No source URL found in the text")
            return source_analysis

        for url in urls:
            try:
                domain = urlparse(url).netloc.lower()
                if domain.startswith('www.'):
                    domain = domain[4:]

                # Check if it's a verified source
                if domain in self.verified_sources:
                    return {
                        'is_verified': True,
                        'source_name': self.verified_sources[domain]['name'],
                        'source_url': url,
                        'reliability_score': self.verified_sources[domain]['reliability'],
                        'warning_flags': []
                    }

                # Check if it's a known unreliable source
                if any(unreliable in domain for unreliable in self.unreliable_sources):
                    return {
                        'is_verified': False,
                        'source_name': domain,
                        'source_url': url,
                        'reliability_score': 0.2,
                        'warning_flags': ["Source is from a potentially unreliable platform"]
                    }

            except Exception as e:
                st.warning(f"Error analyzing URL {url}: {e}")
                continue

        source_analysis['warning_flags'].append("Source not found in verified news sources")
        return source_analysis

    def verify_news(self, news_text, source_url=None):
        """Enhanced news verification with source analysis and custom search verification"""
        if not news_text or len(news_text.strip()) == 0:
            return {
                'error': 'Empty news text provided',
                'is_fake': True,
                'confidence': 1.0
            }

        results = {
            'text_analysis': None,
            'source_analysis': None,
            'custom_search_results': None,
            'is_fake': None,
            'confidence': 0.0,
            'warning_flags': [],
            'recommendation': ''
        }

        # 1. Source Analysis
        source_results = self.analyze_source(news_text, [source_url] if source_url else None)
        results['source_analysis'] = source_results
        
        # 2. Google Custom Search Verification
        custom_search_results = self.custom_search_verification(news_text)
        results['custom_search_results'] = custom_search_results

        # 3. BERT Model Prediction
        try:
            if self.classifier is None:
                raise ValueError("Model not initialized")

            model_result = self.classifier(news_text)
            # Convert BERT output to clear fake/real classification
            is_fake = model_result[0]['label'] == 'LABEL_0'  # Assuming LABEL_0 is fake
            model_confidence = model_result[0]['score']

            results['text_analysis'] = {
                'model_prediction': 'FAKE' if is_fake else 'REAL',
                'model_confidence': model_confidence
            }

        except Exception as e:
            st.error(f"Error in model prediction: {e}")
            results['warning_flags'].append("Error in model prediction")
            return results

        # 4. Combined Analysis
        source_reliability = source_results['reliability_score']
        
        # Factor in Custom Search results
        custom_search_factor = 0
        if custom_search_results.get('verified', False):
            # If content is verified by trusted sources, increase reliability
            trusted_count = len(custom_search_results.get('trusted_sources', []))
            custom_search_factor = min(0.3, 0.1 * trusted_count)
            
            # Add verification info to results
            results['warning_flags'].append(f"Content verified by {trusted_count} trusted news source(s)")
        else:
            # If no verification, slightly decrease reliability
            results['warning_flags'].append("Content not found in trusted news sources")

        # Adjust the final prediction based on all factors
        if source_reliability > 0.8 and not is_fake:
            results['is_fake'] = False
            results['confidence'] = (model_confidence + source_reliability + custom_search_factor) / 3
        elif source_reliability < 0.3 and is_fake:
            results['is_fake'] = True
            results['confidence'] = max(model_confidence, 0.8)
        else:
            results['is_fake'] = is_fake
            results['confidence'] = model_confidence * 0.6 + (1 - source_reliability) * 0.2 + custom_search_factor * 0.2

        # Add warning flags and recommendations
        if not source_results['is_verified']:
            results['warning_flags'].append("Unverified news source")
        if source_reliability < 0.5:
            results['warning_flags'].append("Low reliability source")

        # Generate recommendation
        if results['is_fake']:
            results['recommendation'] = "This news appears to be fake. Please verify with multiple reliable sources."
        elif results['confidence'] < 0.7:
            results['recommendation'] = "This news requires additional verification. Check multiple reliable sources."
        else:
            results['recommendation'] = "This news appears to be reliable, but it's always good to verify with multiple sources."

        return results

# Streamlit App
def main():
    # Set page configuration
    st.set_page_config(
        page_title="AI-Powered Fake News Detection System",
        page_icon="🕵️",
        layout="wide"
    )

    # Title and description
    st.title("🕵️ AI-Powered Fake News Detection System")
    st.markdown("""
    ### Detect Fake News with AI-Powered Analysis
    This tool helps you verify the credibility of news articles using advanced AI techniques and cross-references with trusted news sources.
    """)

    # Initialize the detector
    detector = EnhancedNewsDetector()

    # Input sections
    col1, col2 = st.columns(2)

    with col1:
        # News Text Input
        news_text = st.text_area(
            "Enter the news text:",
            height=300,
            placeholder="Paste the news article or text here..."
        )

    with col2:
        # Source URL Input
        source_url = st.text_input(
            "Source URL (Optional)",
            placeholder="https://example.com/news-article"
        )

    # Verification Button
    if st.button("Verify News", type="primary"):
        # Validate input
        if not news_text.strip():
            st.error("Please enter some news text to verify.")
            return

        # Show loading spinner
        with st.spinner("Analyzing news article..."):
            # Verify the news
            try:
                result = detector.verify_news(news_text, source_url)
                
                # Display all verification details on a single page
                st.markdown("---")
                
                # Main verdict section
                col_verdict, col_confidence = st.columns(2)
                with col_verdict:
                    if result['is_fake']:
                        st.error("### 🚨 FAKE NEWS DETECTED")
                    else:
                        st.success("### ✅ NEWS APPEARS RELIABLE")
                
                with col_confidence:
                    st.metric(
                        label="Overall Confidence",
                        value=f"{result['confidence']:.2%}",
                        help="Combined confidence from source, content analysis, and cross-verification"
                    )
                
                # Recommendation
                st.info(f"### Recommendation\n{result['recommendation']}")
                
                # Source Analysis Section
                st.markdown("## 1. Source Analysis")
                source_info = result['source_analysis']
                
                col_source1, col_source2 = st.columns(2)
                with col_source1:
                    # Verification Status
                    if source_info['is_verified']:
                        st.success(f"✅ Verified Source: {source_info['source_name']}")
                    else:
                        st.warning("⚠️ Unverified Source")
                
                with col_source2:
                    # Source Reliability Visualization
                    st.metric(
                        label="Source Reliability",
                        value=f"{source_info['reliability_score']:.2f}/1.00",
                        help="Reliability score based on known news sources"
                    )
                
                # Warning Flags for Source
                if source_info['warning_flags']:
                    st.error("Source Warning Flags:")
                    for flag in source_info['warning_flags']:
                        st.write(f"- {flag}")
                
                # Content Analysis Section
                st.markdown("## 2. Content Analysis")
                text_analysis = result['text_analysis']
                
                col_content1, col_content2 = st.columns(2)
                with col_content1:
                    # Model Prediction Visualization
                    st.metric(
                        label="Model Prediction",
                        value=text_analysis['model_prediction'],
                        help="AI model's assessment of news authenticity"
                    )
                
                with col_content2:
                    # Confidence Visualization
                    st.metric(
                        label="Model Confidence",
                        value=f"{text_analysis['model_confidence']:.2%}",
                        help="Confidence level of the AI model's prediction"
                    )
                
                # Custom Search Verification Section
                st.markdown("## 3. Cross-Verification with Trusted News Sources")
                search_results = result['custom_search_results']
                
                if 'error' in search_results:
                    st.error(f"Search Verification Error: {search_results['error']}")
                else:
                    # Display trusted sources that verified the content
                    trusted_sources = search_results.get('trusted_sources', [])
                    if trusted_sources:
                        st.success(f"✅ Content verified by {len(trusted_sources)} trusted news source(s)")
                        st.write("### Trusted Sources That Verified This Content:")
                        
                        for idx, source in enumerate(trusted_sources):
                            st.write(f"{idx+1}. **{source['name']}** (Reliability Score: {source['reliability']:.2f})")
                    else:
                        st.warning("⚠️ Content not found in trusted news sources")
                    
                    # Display search results
                    all_results = search_results.get('search_results', [])
                    if all_results:
                        st.write("### Top Search Results:")
                        for idx, result in enumerate(all_results):
                            with st.expander(f"{idx+1}. {result['title']} {'✅' if result['is_trusted'] else ''}"):
                                st.write(f"**Source**: {result['source']}")
                                st.write(f"**Snippet**: {result['snippet']}")
                                st.write(f"**Link**: {result['link']}")
                    else:
                        st.info("No search results found for this content")
                
                # Combined Warning Flags
                if result['warning_flags']:
                    st.markdown("## 4. Additional Flags and Notes")
                    for flag in result['warning_flags']:
                        st.write(f"- {flag}")

            except Exception as e:
                st.error(f"An error occurred during verification: {e}")

    # Display information about verified news sources
    with st.expander("ℹ️ Information About Trusted News Sources"):
        st.write("### Trusted News Sources Used for Verification")
        st.write("This system verifies news against the following trusted sources:")
        
        # Create a table of trusted sources
        trusted_sources_data = []
        for domain, info in detector.verified_sources.items():
            trusted_sources_data.append([info['name'], domain, f"{info['reliability']:.2f}/1.00"])
        
        # Display as a table
        st.table({"News Source": [s[0] for s in trusted_sources_data],
                  "Website": [s[1] for s in trusted_sources_data],
                  "Reliability Score": [s[2] for s in trusted_sources_data]})
        
        st.write("""
        ### About the Verification Process
        1. **Source Analysis**: Checks if the news comes from a verified source
        2. **Content Analysis**: Uses a BERT-based AI model to analyze text patterns
        3. **Cross-Verification**: Searches trusted news sources for similar content
        4. **Combined Assessment**: Integrates all factors for a final verdict
        """)

# Run the Streamlit app
if __name__ == "__main__":
    main()
