from keras.preprocessing.sequence import pad_sequences as ps
from keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional, TimeDistributed
from keras.preprocessing.text import Tokenizer
from keras.callbacks import EarlyStopping
from keras.models import Sequential
import keras.utils as k
import numpy as np
from keras.models import model_from_json
import os
tokenizer = Tokenizer()
 
def create_model(predictors, label, max_sequence_len, total_words):
 
    model = Sequential()
    model.add(Embedding(total_words, 15, input_length=max_sequence_len-1))
   
    model.add(Bidirectional(LSTM(150, return_sequences = True)))
    model.add(Dropout(0.3))
    model.add(Bidirectional(LSTM(200, return_sequences = True)))
    model.add(Dropout(0.5))
    model.add(Bidirectional(LSTM(240, return_sequences = True)))
    model.add(Dropout(0.6))
    model.add(LSTM(340))
    model.add(Dropout(0.7))
    model.add(Dense(total_words, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    earlystop = EarlyStopping(monitor='loss', min_delta=0, patience=5, verbose=0, mode='auto')
    model.fit(predictors, label, epochs=400, verbose=1, callbacks=[earlystop])
    print (model.summary())
    # serialize model to JSON
    model_json = model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("model.h5")
    print("Saved model to disk")       
    return model
   
    
def data_prep(data):
 
 
    corpus = data.lower().split("\n")
 
    # tokenize##########
    tokenizer.fit_on_texts(corpus)
    total_words = len(tokenizer.word_index) + 1
 
    # input sequences using token list#########
    input_sequences = []
    for line in corpus:
        token_list = tokenizer.texts_to_sequences([line])[0]
        for i in range(1, len(token_list)):
            n_gram_sequence = token_list[:i+1]
            input_sequences.append(n_gram_sequence)
 
    # sequence padding##########
    max_sequence_len = max([len(x) for x in input_sequences])
    input_sequences = np.array(ps(input_sequences, maxlen=max_sequence_len, padding='pre'))
 
    # predictor-label creation############
    predictors, label = input_sequences[:,:-1],input_sequences[:,-1]
    label = k.to_categorical(label, num_classes=total_words)
 
    return predictors, label, max_sequence_len, total_words
 
 
 
def generate_text(seed_text, next_words, max_sequence_len):
    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = ps([token_list], maxlen=max_sequence_len-1, padding='pre')
        predicted = model.predict_classes(token_list, verbose=0)
 
        output_word = ""
        for word, index in tokenizer.word_index.items():
            if index == predicted:
                op_word = word
                break
        seed_text += " " + op_word
    return seed_text
 
data1 = open('YOUR_TEXT_FILE.txt').read()
 
predictors, label, max_sequence_len, total_words = data_prep(data1)
model = create_model(predictors, label, max_sequence_len, total_words)
print (generate_text("two blouses one", 8, max_sequence_len))
