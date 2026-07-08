import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import pandas as pd
import numpy as np

from Config import load_config
config = load_config()
from sklearn.metrics import mean_squared_error,r2_score




def eval_model(model,X,y,name='VAL set'):
    pred = model.predict(X)
    mse = mean_squared_error(y,pred)
    r2score = r2_score(y,pred)
    print(f"{name}-Evaluation :\n MSE : {mse} \t R2Score : {r2score}")

    return mse,r2score