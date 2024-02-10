
import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import LabelEncoder
import streamlit as st
import random
import pickle

# Load data
with open("aka.json") as file:
    data = json.load(file)

# Load model, tokenizer, and label encoder
model = keras.models.load_model('chat_model')
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)
with open('label_encoder.pickle', 'rb') as enc:
    lbl_encoder = pickle.load(enc)

# Set parameters
max_len = 30

# Streamlit app
st.title("Simple Chatbot")

# Initialize conversation history
conversation_history = st.session_state.get("conversation_history", [])

# User input
user_input = st.text_input("User: ")

if user_input.lower() == "quit":
    st.text("Chatbot: Goodbye! Type anything to start a new conversation.")
    conversation_history = []  # Reset conversation history
else:
    # Process user input
    result = model.predict(keras.preprocessing.sequence.pad_sequences(tokenizer.texts_to_sequences([user_input]), truncating='post', maxlen=max_len))
    tag = lbl_encoder.inverse_transform([np.argmax(result)])[0]

    # Get and display bot response
    for i in data['Ak']:
        if i['tag'] == tag:
            bot_response = np.random.choice(i['responses'])
            st.text("ChatBot: " + bot_response)
            conversation_history.append(("User: " + user_input, "ChatBot: " + bot_response))

# Save conversation history in session state
st.session_state.conversation_history = conversation_history

# Display all conversation history
st.text("Conversation History:")
for interaction in conversation_history:
    st.text(interaction)
