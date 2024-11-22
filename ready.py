#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Required Libraries
import tensorflow as tf
import random
import json
import nltk
from nltk.stem.lancaster import LancasterStemmer
import pickle
import numpy as np

# Initialize stemmer
stemmer = LancasterStemmer()

# Load intents
print("Loading intents...")
with open('intents.json') as json_data:
    intents = json.load(json_data)

# Initialize variables
words = []
classes = []
documents = []
ignore_words = ['?']

print("Processing intents...")
for intent in intents['intents']:
    for pattern in intent['patterns']:
        # Tokenize each word in the pattern
        w = nltk.word_tokenize(pattern)
        words.extend(w)
        documents.append((w, intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# Stem and sort words
words = sorted(set(stemmer.stem(w.lower()) for w in words if w not in ignore_words))
classes = sorted(set(classes))

print(f"{len(documents)} documents")
print(f"{len(classes)} classes: {classes}")
print(f"{len(words)} unique stemmed words: {words}")

# Create training data
print("Creating training data...")
training = []
output_empty = [0] * len(classes)

for doc in documents:
    bag = []
    pattern_words = [stemmer.stem(word.lower()) for word in doc[0]]
    for w in words:
        bag.append(1 if w in pattern_words else 0)

    output_row = list(output_empty)

    output_row[classes.index(doc[1])] = 1
    training.append((bag, output_row))

print("Shuffling training data...")
random.shuffle(training)

# Split the training data into train_x and train_y
train_x, train_y = zip(*training)

# Build the TensorFlow model
print("Building the model...")
model = tf.keras.Sequential([
    tf.keras.layers.Dense(8, input_shape=(len(words),), activation='relu'),
    tf.keras.layers.Dense(8, activation='relu'),
    tf.keras.layers.Dense(len(train_y[0]),activation='softmax')
])

# No NumPy conversions
# Convert training data to tensors directly
train_x = tf.constant(train_x)
train_y = tf.constant(train_y)

print("Shape of train_x:", train_x.shape)
print("Shape of train_y:", train_y.shape)

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(train_x, train_y, epochs=1000, batch_size=8, verbose=1)


# Save the model and training data
model.save('chatbot_model.h5')
pickle.dump({'words': words, 'classes': classes}, open('training_data.pkl', 'wb'))

print("Model training complete and saved.")

# Define utility functions for chatbot responses
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    return [stemmer.stem(word.lower()) for word in sentence_words]

def bow(sentence, words):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
    return bag

def classify(sentence, error_threshold=0.25):
    bag = bow(sentence, words)
    results = model.predict([bag])[0]
    results = [[i, r] for i, r in enumerate(results) if r > error_threshold]
    results.sort(key=lambda x: x[1], reverse=True)
    return [(classes[r[0]], r[1]) for r in results]

def response(sentence):
    results = classify(sentence)
    if results:
        for intent in intents['intents']:
            if intent['tag'] == results[0][0]:
                return random.choice(intent['responses'])

# Chatbot interaction
print("Chatbot ready! Type 'exit' to stop.")
while True:
    input_data = input("You: ")
    if input_data.lower() == 'exit':
        print("Exiting...")
        break
    print("Bot:", response(input_data))


# In[ ]:





# In[3]:





# In[5]:





# In[ ]:





# In[ ]:





# In[ ]:




