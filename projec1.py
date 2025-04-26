import streamlit as st
from transformers import AutoModelForSequenceClassification
from transformers import TFAutoModelForSequenceClassification
from transformers import AutoTokenizer, AutoConfig
import numpy as np
from scipy.special import softmax
# Preprocess text (username and link placeholders)



import requests
from bs4 import BeautifulSoup
import urllib.parse

def review_list(user_input):
    parts = urllib.parse.urlparse(user_input)
    path_parts = parts.path.split('/')
    query = urllib.parse.parse_qs(parts.query)

    product_name = path_parts[1]
    item_id = path_parts[3]
    pid = query['pid'][0]
    lid = query['lid'][0]
    url = f"https://www.flipkart.com/{product_name}/product-reviews/{item_id}?pid={pid}&lid={lid}&marketplace=FLIPKART"
    response = requests.get(url+"?page=1")
    soup = BeautifulSoup(response.content, 'html.parser')
    span_text=soup.find('div', {'class': 'atZ055'}).find('span').text
    parts = span_text.split('and')
    reviews_part = parts[1].strip()
    reviews_number = int(reviews_part.split(' ')[0])
    if reviews_number<=10:
      pages=1

    else:
      if reviews_number%10==0:
          pages=int(reviews_number/10)
      else:
          pages=int(reviews_number/10+1)

    all_reviews = []
    for i in range(1, pages+1):
      result = requests.get(url+"?page={i}")
      reviews = BeautifulSoup(result.content, 'html.parser')
      all_reviews+=(soup.find_all('div', {'class': '_11pzQk'}))

    full_reviews=[]
    for i in all_reviews:
        full_reviews.append(i.text)
    return full_reviews

def preprocess(text):
    new_text = []
    for t in text.split(" "):
        t = '@user' if t.startswith('@') and len(t) > 1 else t
        t = 'http' if t.startswith('http') else t
        new_text.append(t)
    return " ".join(new_text)

def process_predict(text):
    text = preprocess(text)
    encoded_input = tokenizer(text, return_tensors='pt')
    output = model(**encoded_input)
    scores = output[0][0].detach().numpy()
    scores = softmax(scores)
    ranking = np.argsort(scores)
    ranking = ranking[::-1]
    l = config.id2label[ranking[0]]
    # s = scores[ranking[0]]
    return l

MODEL = f"cardiffnlp/twitter-roberta-base-sentiment-latest"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
config = AutoConfig.from_pretrained(MODEL)
# PT
model = AutoModelForSequenceClassification.from_pretrained(MODEL)
# Title of the app
st.title('Customer review')

# Get user input using text_input
user_input = st.text_input("Type flipkart product url: ")

# Print the user input
if user_input:
    fpk_review = review_list(user_input)

    out_review = []
    for i in fpk_review:
        # print(process_predict(i))
        if process_predict(i) == 'negative':
            out_review.append(-1)
        elif process_predict(i) == 'positive':
            out_review.append(1)
        else:
            out_review.append(0)
    final = sum(out_review)
    st.write(f"{'positive' if final > 0 else 'negative' if final < 0 else 'neutral' } ")

    # l,_ = process_predict(user_input)
    # print(f"{0+1}) {l} {np.round(float(s), 4)*100}%")
    # st.write(f"{np.round(float(s), 4)*100}% {l} ")