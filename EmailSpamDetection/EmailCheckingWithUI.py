import streamlit as st
import re
import joblib
import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Load the trained classifier and vectorizer
loaded_classifier = joblib.load("trained_classifier.pkl")
loaded_vectorizer = joblib.load("vectorizer.pkl")
#stopwrds list
stopwords = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"]
#UI
st.title("Email Spam Detection")
email_input = st.text_area("Email:", "You may type your email here")
detect_bt = st.button("Detect")
#backend
if detect_bt:
    #converting email to lowercase
    email_input = email_input.lower()
    #removing special characetrs from email
    email_input = re.sub(r'[^a-zA-Z\s]', '', email_input)
    #removing stopwords from email
    email_words = email_input.split()   #spliting email to words
    filtered_words = [word for word in email_words if word not in stopwords] #goes through each words in email_words if it is not a stopword then add it to filtered words
    preprocessed_email = ' '.join(filtered_words) #jois all filtered words and make them sentence with space 
    
    
    # Vectorize the preprocessed email using the loaded vectorizer
    email_vector = loaded_vectorizer.transform([preprocessed_email])
    # Make a prediction using the loaded classifier
    prediction = loaded_classifier.predict(email_vector)[0]
    
    # Display the prediction
    if prediction == 0:
        st.success("The email is not spam.")
    else:
        st.error("The email is spam.")
