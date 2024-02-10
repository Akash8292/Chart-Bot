import json 
import numpy as np 
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, GlobalAveragePooling1D
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder

with open('aka.json') as file:
    data = json.load(file)
    
training_sentences = []
training_labels = []
labels = []
responses = []


for x in data['Ak']:
    for pattern in x['patterns']:
        training_sentences.append(pattern)
        training_labels.append(x['tag'])
    responses.append(x['responses'])
    
    if x['tag'] not in labels:
        labels.append(x['tag'])
        
num_classes = len(labels)

lbl_encoder = LabelEncoder()    # o convert string labels into numerical format, 
lbl_encoder.fit(training_labels)
training_labels = lbl_encoder.transform(training_labels)


vocab_size = 1000     # the maximum number of words to keep based on word frequency. 
embedding_dim = 16      # This is the size of the vector space 
max_len = 30          # Sentences longer than this will be truncated, and sentences shorter than this will be padded.
oov_token = "<OOV>"  # used to replace words that are not in the vocabulary.

tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_token)    #  It converts text data into sequences of integers.
tokenizer.fit_on_texts(training_sentences)
word_index = tokenizer.word_index
sequences = tokenizer.texts_to_sequences(training_sentences)     # It convert sequence to text
padded_sequences = pad_sequences(sequences, truncating='post', maxlen=max_len)



model = Sequential()
model.add(Embedding(vocab_size, embedding_dim, input_length=max_len))
model.add(GlobalAveragePooling1D())
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss='sparse_categorical_crossentropy',optimizer='adam', metrics=['accuracy'])

model.summary()
epochs = 1000
history = model.fit(padded_sequences, np.array(training_labels), epochs=epochs)

# to save the trained model
model.save("chat_model")

import pickle

# to save the fitted tokenizer
with open('tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
# to save the fitted label encoder
with open('label_encoder.pickle', 'wb') as ecn_file:
    pickle.dump(lbl_encoder, ecn_file, protocol=pickle.HIGHEST_PROTOCOL)