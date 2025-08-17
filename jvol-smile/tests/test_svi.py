from jvol_smile.svi import SVIFitter
from jvol_smile.utils import synthetic_surface
import numpy as np

def test_svi_fit_mse():
    data=synthetic_surface(seed=2); k,iv=data["k"],data["iv"]
    fit=SVIFitter(T=1.0).fit(k,iv,n_starts=64,seed=3)
    pred=fit.predict_iv(k)
    mse=float(np.mean((pred-iv)**2))
    assert mse<1e-3
