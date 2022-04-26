import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import json
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import sklearn


import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from math import sqrt
from sklearn.metrics import r2_score

import random

words=[]
classes = []
documents = []
ignore_words = ['?', '!']
data_file = open('C:/Users/Afanwi/Downloads/chatbot-python-project-data-codes/intents.json').read() #read the json file
intents = json.loads(data_file)


for intent in intents['intents']: #loop so every time we see a word or phrase that is in the intents, the loop works
    for pattern in intent['patterns']:

        #tokenize each word
        w = nltk.word_tokenize(pattern)
        words.extend(w)
        vectorizer = CountVectorizer()
        vectorizer.fit(words)
        vector = vectorizer.transform(words)
        vector.toarray()
        #add documents in the corpus
        documents.append((w, intent['tag']))

        # add to our classes list
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# make each word lowercase and remove duplicates
words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]
words = sorted(list(set(words)))
# sort classes
classes = sorted(list(set(classes)))
# documents = combination between patterns and intents
print (len(documents), "documents")
# classes = intents
print (len(classes), "classes", classes)
# words = all words, vocabulary
print (len(words), "unique lemmatized words", words)


pickle.dump(words,open('words.pkl','wb'))
pickle.dump(classes,open('classes.pkl','wb'))

# create our training data
training = []
# create an empty array for our output
output_empty = [0] * len(classes)
# training set, bag of words for each sentence
for doc in documents:
    # initialize our bag of words
    bag = []
    # list of tokenized words for the pattern
    pattern_words = doc[0]
    # lemmatize each word - create base word, in attempt to represent related words
    pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words]
    
    # create our bag of words array with 1, if word match found in current pattern
    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0)
    
    # output is a '0' for each tag and '1' for current tag (for each pattern)
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1
    
    training.append([bag, output_row])
# shuffle our features and turn into np.array
random.shuffle(training)
training = np.array(training)
# create train and test lists. X for patterns, Y for intents
train_x = list(training[:,0])
train_y = list(training[:,1])
print("Training data created")
print(train_x)
print(train_y)
mlp = MLPClassifier(hidden_layer_sizes=(8,8,8), activation = 'relu', solver='adam', max_iter=500)
mlp.fit(train_x,train_y)

predict_train = mlp.predict(train_x)

from sklearn.metrics import classification_report,confusion_matrix 
# print(confusion_matrix(train_y,predict_train))
print(classification_report(train_y,predict_train)) #print the matrices





## This last part was not used. It was commented off for me to see and remember so please disregard

# # # Create model - 3 layers. First layer 128 neurons, second layer 64 neurons and 3rd output layer contains number of neurons
# # # equal to number of intents to predict output intent with softmax
# # model = Sequential()
# # model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
# # model.add(Dropout(0.5))
# # model.add(Dense(64, activation='relu'))
# # model.add(Dropout(0.5))
# # model.add(Dense(len(train_y[0]), activation='softmax'))

# # # Compile model. Stochastic gradient descent with Nesterov accelerated gradient gives good results for this model
# # sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
# # model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# # #fitting and saving the model 
# # hist = model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1)
# # model.save('chatbot_model.h5', hist)

# print("model created")
