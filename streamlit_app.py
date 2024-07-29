import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score
import streamlit as st

st.title("SPAM SMS DETECTION APP")
st.write("Checks if message is spam or not.")

# dataset
data = pd.read_csv("https://raw.githubusercontent.com/Solo-Ojay/Spam_Detection/master/sms_text.csv")

# Splitting the data into message and labels
X = data['Message']
y = data['Labels']

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

#CountVectorizer and Naive Bayes model
count_vector = CountVectorizer()
naive_bayes = MultinomialNB()

# Transforming the training data and train the model
train = count_vector.fit_transform(X_train)
naive_bayes.fit(train, y_train)

# Initializing session state variables
if 'input_message' not in st.session_state:
    st.session_state.input_message = ''
if 'prediction' not in st.session_state:
    st.session_state.prediction = ''

# text box
input_message = st.text_input('Enter your SMS message:', st.session_state.input_message, key='input_area', help='Type your SMS message here')
st.session_state.input_message = input_message

# Prediction
if st.button("Predict", key='predict_button', help='Click to predict'):
    if st.session_state.input_message:
        prediction = naive_bayes.predict(count_vector.transform([st.session_state.input_message]))
        st.session_state.prediction = 'spam' if prediction[0] == 1 else 'ham'
        st.write(f"Message is {st.session_state.prediction}")
    else:
        st.write("Please enter a message to predict.")

# reset button
if st.button("Clear", key='clear_button', help='Click to clear input'):
    st.session_state.input_message = ''
    st.session_state.prediction = ''
    st.experimental_rerun()
