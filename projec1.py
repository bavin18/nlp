import streamlit as st
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import softmax
from bs4 import BeautifulSoup
import requests
import urllib.parse
import time


headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:137.0) Gecko/20100101 Firefox/137.0'
}

cookies = {
    'solved_captcha': '1745776538-24039-69106680-85404a1de10c054c81d3dea0032deeb3f21acbc7634e032c5e9306f379057b03',
}

# Function to scrape reviews
def review_list(user_input):
    parts = urllib.parse.urlparse(user_input)
    path_parts = parts.path.split('/')
    query = urllib.parse.parse_qs(parts.query)

    product_name = path_parts[1]
    item_id = path_parts[3]
    pid = query['pid'][0]
    lid = query['lid'][0]
    url = f"https://www.flipkart.com/{product_name}/product-reviews/{item_id}?pid={pid}&lid={lid}&marketplace=FLIPKART"
    
    all_reviews = []
    for i in range(1, 3):  # limiting to 2 pages for speed
        result = requests.get(url + f"&page={i}", headers=headers, cookies=cookies)
        soup = BeautifulSoup(result.content, 'html.parser')
        if soup.find_all('div', {'class': '_11pzQk'}):
            review_divs = soup.find_all('div', {'class': '_11pzQk'})
        elif soup.find_all('div', {'class': 'ZmyHeo'}):
            review_divs = soup.find_all('div', {'class': 'ZmyHeo'})
        for div in review_divs:
            all_reviews.append(div.text.split('READ MORE')[0])
    return all_reviews

# Preprocessing for model
def preprocess(text):
    new_text = []
    for t in text.split(" "):
        t = '@user' if t.startswith('@') and len(t) > 1 else t
        t = 'http' if t.startswith('http') else t
        new_text.append(t)
    return " ".join(new_text)

# Sentiment prediction
def process_predict(text):
    text = preprocess(text)
    encoded_input = tokenizer(text, return_tensors='pt')
    output = model(**encoded_input)
    scores = output[0][0].detach().numpy()
    scores = softmax(scores)
    ranking = np.argsort(scores)[::-1]
    label = config.id2label[ranking[0]]
    return label

# Load model
MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
config = AutoConfig.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)


# ------------------------------------
# Streamlit UI starts here
# ------------------------------------

st.set_page_config(page_title="Product Review Sentiment Analyzer", layout="centered")

# Main title and description
st.title("🛒 Product Review Sentiment Analyzer")
st.write("Enter a **Product URL** below. We'll scrape recent reviews, analyze their sentiment using a deep learning model, and summarize the results for you!")

st.markdown("---")

# Input box
user_input = st.text_input("🔗 Enter the Product URL:")

# Button
if st.button("Analyze Reviews"):
    if user_input:
        with st.spinner('🔎 Scraping reviews and analyzing sentiment...'):
            time.sleep(1)
            reviews = review_list(user_input)

            if not reviews:
                st.error("No reviews found or scraping blocked. Try another product or use a VPN.")
            else:
                sentiments = []
                for review in reviews:
                    sentiments.append(process_predict(review))
                
                pos = sentiments.count('positive')
                neg = sentiments.count('negative')
                neu = sentiments.count('neutral')
                total = pos + neg + neu

                overall_score = pos - neg

                # Result Badge
                if overall_score > 0:
                    st.success("✅ Overall Sentiment: Positive")
                elif overall_score < 0:
                    st.error("❌ Overall Sentiment: Negative")
                else:
                    st.info("⚪ Overall Sentiment: Neutral")
                
                st.markdown("---")

                # Pie chart for distribution
                labels = ['Positive', 'Negative', 'Neutral']
                sizes = [pos, neg, neu]
                colors = ['#28a745', '#dc3545', '#6c757d']

                fig, ax = plt.subplots()
                ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
                ax.axis('equal')  # Equal aspect ratio
                st.pyplot(fig)

                st.markdown("---")

                # Expandable section for detailed reviews
                with st.expander("📋 Show All Reviews with Sentiments"):
                    for idx, review in enumerate(reviews):
                        st.write(f"**Review {idx+1}:** {review}")
                        
                        sentiment = sentiments[idx]
                        
                        if sentiment == 'positive':
                            st.markdown(f"<span style='color: green; font-weight: bold;'>Sentiment: Positive</span>", unsafe_allow_html=True)
                        elif sentiment == 'negative':
                            st.markdown(f"<span style='color: red; font-weight: bold;'>Sentiment: Negative</span>", unsafe_allow_html=True)
                        else:
                            st.markdown(f"<span style='color: grey; font-weight: bold;'>Sentiment: Neutral</span>", unsafe_allow_html=True)
                        
                        st.markdown("---")
    else:
        st.warning("Please enter a Flipkart product URL to continue.")
