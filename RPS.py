import random
import numpy as np
import pandas as pd
from tensorflow import keras

moves = ['R', 'P', 'S']
ideal_response = {'R': 'P', 'P': 'S', 'S': 'R'}

# Variables for Keras method
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

def player(prev_play, opponent_history=[]):

    # Use a random choice by default
    guess = random.choice(moves)

    # Use Markov Chain method
    # - wins with all players with > 60% efficiency
    # - possible to adjust results with 'memory' variable
    if use_markov_chain == True:
        global matrix, my_history
        # initialize variables in the first game
        if prev_play == '':
            for pair_key in pair_keys:
                matrix[pair_key] = {'R': 1 / 3,
                                    'P': 1 / 3,
                                    'S': 1 / 3}
            opponent_history = []
            my_history = []
        # otherwise, add previous opponent play to the history
        else:
            opponent_history.append(prev_play)

        # make a prediction when enough entries in the history
        if len(my_history) >= 2:
            # create a pair from 2 plays ago
            pair_key = my_history[-2] + opponent_history[-2]
            # introduce a memory loss of earlier observations for that pair,
            # memory decay speed can be adjusted using 'memory' variable
            for key in matrix[pair_key]:
                matrix[pair_key][key] = memory * matrix[pair_key][key]
            # then, update matrix for that pair
            matrix[pair_key][prev_play] += 1

            # create a pair from the previous play
            curr_key = my_history[-1] + opponent_history[-1]
            # if the matrix values are not equal for that pair,
            # make a prediction using the move with the higest value
            if max(matrix[curr_key].values()) != min(matrix[curr_key].values()):
                prediction = max([(v, k) for k, v in matrix[curr_key].items()])[1]
                guess = ideal_response[prediction]

        # append my guess to the history
        my_history.append(guess)

    # [Deprecated] Use Keras library instead
    # Warning:
    # - 1st attempt for this problem that does not win with all players
    # - requires much more computation time
    if use_markov_chain == False:
        global df_train_x, df_train_y, model
        # initialize variables in the first game
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
        # otherwise, add previous opponent play to the history
        else:
            opponent_history.append(moves.index(prev_play))

        # use opponents play history to build a dataframe of opponent history
        # series of length of 'hlen' each as x axis and their following moves
        # as y axis for the training after at least 'hlen+1' plays
        if len(opponent_history) > hlen:
            df_train_x = df_train_x.append(pd.Series(opponent_history[-(hlen+1):-1]), ignore_index=True).astype('int8')
            df_train_y = df_train_y.append(pd.Series(opponent_history[-1]), ignore_index=True).astype('int8')

        # after 'hlen+hentries' plays, fit the model and make a prediction
        if len(opponent_history) >= (hlen+hentries):
            model.fit(df_train_x, df_train_y, epochs=5, verbose=0)
            df_test_x = pd.DataFrame([opponent_history[-hlen:]])
            predictions = model.predict([df_test_x])
            guess = ideal_response[moves[np.argmax(predictions[0])]]

    # Return player guess
    return guess
