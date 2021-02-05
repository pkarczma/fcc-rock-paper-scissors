import random
import numpy as np
import pandas as pd
from tensorflow import keras

moves = ['R', 'P', 'S']
ideal_response = {'R': 'P', 'P': 'S', 'S': 'R'}

# Variables for Keras method
use_keras = False
df_train_x = None
df_train_y = None
model = None
hlen = 5
hentries = 20

# Variables for Markov Chain method
use_markov_chain = True
pair_keys = ['RR', 'RP', 'RS', 'PR', 'PP', 'PS', 'SR', 'SP', 'SS']
matrix = {}
memory = 0.9
my_history = []
opp_history = []

def player(prev_play, opponent_history=[]):

    # Use a random choice by default
    guess = random.choice(moves)

    if use_markov_chain == True:
        global matrix, my_history, opp_history
        if prev_play == '':
            for pair_key in pair_keys:
                matrix[pair_key] = {'R': 1 / 3,
                                    'P': 1 / 3,
                                    'S': 1 / 3}
            opp_history = []
            my_history = []
        else:
            opp_history.append(prev_play)

        if len(my_history) >= 2:
            pair_key = my_history[-2] + opp_history[-2]
            for key in matrix[pair_key]:
                matrix[pair_key][key] = memory * matrix[pair_key][key]
            matrix[pair_key][prev_play] += 1
            curr_key = my_history[-1] + opp_history[-1]
            if max(matrix[curr_key].values()) != min(matrix[curr_key].values()):
                prediction = max([(v, k) for k, v in matrix[curr_key].items()])[1]
                guess = ideal_response[prediction]

        my_history.append(guess)
        
    if use_keras == True:
        global df_train_x, df_train_y, model
        if prev_play == '':
            df_train_x = pd.DataFrame()
            df_train_y = pd.DataFrame()
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

        if len(opponent_history) > hlen:
            df_train_x = df_train_x.append(pd.Series(opponent_history[-(hlen+1):-1]), ignore_index=True).astype('int8')
            df_train_y = df_train_y.append(pd.Series(opponent_history[-1]), ignore_index=True).astype('int8')

        if len(opponent_history) >= (hlen+hentries):
            model.fit(df_train_x, df_train_y, epochs=5, verbose=0)
            df_test_x = pd.DataFrame([opponent_history[-hlen:]])
            predictions = model.predict([df_test_x])
            guess = ideal_response[moves[np.argmax(predictions[0])]]

    # Return player guess
    return guess
