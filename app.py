import streamlit as st
import torch
from transformers import pipeline
from urllib.parse import urlparse
import re

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
           """Enhanced news verification with source analysis"""
           if not news_text or len(news_text.strip()) == 0:
               return {
                   'error': 'Empty news text provided',
                   'is_fake': True,
                   'confidence': 1.0
               }

           results = {
               'text_analysis': None,
               'source_analysis': None,
               'is_fake': None,
               'confidence': 0.0,
               'warning_flags': [],
               'recommendation': ''
           }

           # 1. Source Analysis
           source_results = self.analyze_source(news_text, [source_url] if source_url else None)
           results['source_analysis'] = source_results

           # 2. BERT Model Prediction
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

           # 3. Combined Analysis
           source_reliability = source_results['reliability_score']

           # Adjust the final prediction based on source reliability
           if source_reliability > 0.8 and not is_fake:
               results['is_fake'] = False
               results['confidence'] = (model_confidence + source_reliability) / 2
           elif source_reliability < 0.3 and is_fake:
               results['is_fake'] = True
               results['confidence'] = max(model_confidence, 0.8)
           else:
               results['is_fake'] = is_fake
               results['confidence'] = model_confidence * 0.7 + (1 - source_reliability) * 0.3

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
        page_title="News Verification Assistant",
        page_icon="🕵️",
        layout="wide"
    )

    # Title and description
    st.title("🕵️ News Verification Assistant")
    st.markdown("""
    ### Detect Fake News with AI-Powered Analysis
    This tool helps you verify the credibility of news articles using advanced AI techniques.
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

                # Create tabs for different analysis sections
                tab1 = st.tabs([
                    "🔍 Source Analysis",
                    "📊 Content Analysis",
                    "🚨 Final Assessment"
                ])

                with tab1:
                    st.subheader("Source Analysis")
                    source_info = result['source_analysis']

                    # Source Reliability Visualization
                    st.metric(
                        label="Source Reliability",
                        value=f"{source_info['reliability_score']:.2f}/1.00",
                        help="Reliability score based on known news sources"
                    )

                    # Verification Status
                    if source_info['is_verified']:
                        st.success(f"✅ Verified Source: {source_info['source_name']}")
                    else:
                        st.warning("⚠️ Unverified Source")

                    # Warning Flags
                    if source_info['warning_flags']:
                        st.error("Warning Flags:")
                        for flag in source_info['warning_flags']:
                            st.write(f"- {flag}")

                    
                    st.subheader("Content Analysis")
                    text_analysis = result['text_analysis']

                    # Model Prediction Visualization
                    st.metric(
                        label="Model Prediction",
                        value=text_analysis['model_prediction'],
                        help="AI model's assessment of news authenticity"
                    )

                    # Confidence Visualization
                    st.metric(
                        label="Model Confidence",
                        value=f"{text_analysis['model_confidence']:.2%}",
                        help="Confidence level of the AI model's prediction"
                    )

                
                    st.subheader("Final Assessment")

                    # Fake News Determination
                    if result['is_fake']:
                        st.error("🚨 FAKE NEWS DETECTED")
                    else:
                        st.success("✅ NEWS APPEARS RELIABLE")

                    # Overall Confidence
                    st.metric(
                        label="Overall Confidence",
                        value=f"{result['confidence']:.2%}",
                        help="Combined confidence from source and content analysis"
                    )

                    # Recommendation
                    st.info(f"🔔 Recommendation: {result['recommendation']}")

                    # Warning Flags
                    if result['warning_flags']:
                        st.warning("Additional Warning Flags:")
                        for flag in result['warning_flags']:
                            st.write(f"- {flag}")

            except Exception as e:
                st.error(f"An error occurred during verification: {e}")

# Run the Streamlit app
if __name__ == "__main__":
    main()
