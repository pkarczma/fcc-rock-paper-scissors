# The example function below keeps track of the opponent's history and plays whatever the opponent played two plays ago. It is not a very good player so you will need to change the code to pass the challenge.

import random
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras

df_x = None
df_y = None
model = None
hlen = 5
moves = ['R', 'P', 'S']
ideal_response = {'R': 'P', 'P': 'S', 'S': 'R'}

def player(prev_play, opponent_history=[]):
    global df_x, df_y, model
    if prev_play == '':
        df_x = pd.DataFrame()
        df_y = pd.DataFrame()
        model = keras.Sequential([
            keras.layers.Dense(hlen, input_shape=(hlen,)),
            keras.layers.Dense(3, activation='softmax')
        ])
        model.compile(optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy'])
        opponent_history = []
    else:
        opponent_history.append(moves.index(prev_play))

    guess = random.choice(moves)

    if len(opponent_history) > hlen:
        df_x = df_x.append(pd.Series(opponent_history[-(hlen+1):-1]), ignore_index=True).astype('int8')
        df_y = df_y.append(pd.Series(opponent_history[-1]), ignore_index=True).astype('int8')

    if len(opponent_history) >= (hlen+20):
        model.fit(df_x, df_y, epochs=3, verbose=0)
        test = pd.DataFrame([opponent_history[-hlen:]])
        predictions = model.predict([test])
        guess = ideal_response[moves[np.argmax(predictions[0])]]

    return guess
