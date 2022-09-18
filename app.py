import streamlit as st
import pickle
from tensorflow.keras.models import load_model
import numpy as np
import os
import h5py

tokenizer = pickle.load(open('token.pkl', 'rb'))
model = load_model('words.h5')

st.title('Next Word Predictor')
text = ("""It is the eve of St. George's Day. Do you not know that to-night, when 
        the clock strikes midnight, all the evil things in the world will have 
        full sway? Do you know where you are going, and what you are going to 
        She was in such evident distress that I tried to comfort her, but 
        without effect. Finally she went down on her knees and implored me not
        to go; at least to wait a day or two before starting. It was all very ridiculous
        but I did not feel comfortable. However, there was business to be done,
        and I could allow nothing to interfere with it. I therefore tried to raise
        her up, and said, as gravely as I could, that I thanked her, but my duty
        was imperative, and that I must go. She then rose and dried her eyes, and
        taking a crucifix from her neck offered it to me.""")

st.write(("***TEXTS:***  "), text)



def predict_next_words(model, tokenizer, text):

    sequence = tokenizer.texts_to_sequences([text])
    sequence = np.array(sequence)
    preds = np.argmax(model.predict(sequence))
    predicted_word = " "

    for key, value in tokenizer.word_index.items():
        if value == preds:
            predicted_word = key
            break

    print(predicted_word)
    return predicted_word

def run():

    text = st.text_input('Enter four words from TEXT: ')


    text = text.split(" ")
    text = text[-3:]
    print(text)

    if st.button("Predict"):
        output = predict_next_words(model, tokenizer, text)
        st.success(f"predict words: {output}")

run()

















