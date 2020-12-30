# The example function below keeps track of the opponent's history and plays whatever the opponent played two plays ago. It is not a very good player so you will need to change the code to pass the challenge.

import random
import pandas as pd

df = None

def player(prev_play, opponent_history=[]):
    global df
    if prev_play == '':
        df = pd.DataFrame()
        opponent_history = []
    else:
        opponent_history.append(prev_play)

    guess = "R"
    if len(opponent_history) > 2:
        guess = opponent_history[-2]

    if len(opponent_history) >= 5:
        df = df.append(pd.Series(opponent_history[-5:]), ignore_index=True)

    return random.choice([guess, "R", "P", "S"])
