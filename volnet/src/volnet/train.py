from __future__ import annotations
import numpy as np
from sklearn.neural_network import MLPRegressor

def synthetic_smile(n=200, seed=0):
    rng=np.random.default_rng(seed)
    k=rng.uniform(-1.5,1.5,size=n)
    iv=0.2+0.1*(k**2)+rng.normal(0,0.01,size=n)
    return k.reshape(-1,1), iv

def train_volnet(seed=0):
    X,y=synthetic_smile(seed=seed)
    model=MLPRegressor(hidden_layer_sizes=(32,32), activation='relu', max_iter=200, random_state=seed)
    model.fit(X,y)
    return model
