import streamlit as st
import pandas as pd
import nltk
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
ps = PorterStemmer()
ntlk.download('punkt')

def transform_text(text):
    text = text.lower()
    text = word_tokenize(text)
    
    y = []
    for i in text:
        if i.isalnum():
            y.append(i)
    
    text = y[:]
    y.clear()
    
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
            
    text = y[:]
    y.clear()
    
    for i in text:
        y.append(ps.stem(i))
    
            
    return " ".join(y)


tfidf = pd.read_pickle('vectorizer.pkl')
model = pd.read_pickle('model.pkl')


st.title('SMS/Email Spam Classifier')


st.write('This app is designed to classify SMS/Email as spam or not spam')
st.write('Please enter the SMS/Email you want to classify')

input_sms = st.text_area('Enter the SMS/Email')

if st.button('Predict'):
    transformed_sms = transform_text(input_sms)

    vect_input = tfidf.transform([transformed_sms])

    result = model.predict(vect_input)[0]

    if result == 1:
        st.write('The text provided is a SPAM')   
    else:
        st.write('The text provided is NOT a SPAM')



