from volnet.train import train_volnet, synthetic_smile
import numpy as np

def test_train_volnet_r2():
    model=train_volnet(seed=1)
    X,y=synthetic_smile(n=80, seed=2)
    yhat=model.predict(X)
    # check that MSE < variance baseline
    mse=float(((yhat-y)**2).mean()); var=float(y.var())
    assert mse<var
