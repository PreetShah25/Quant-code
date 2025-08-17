from __future__ import annotations
import numpy as np

def synthetic_surface(n=51, seed=0):
    rng=np.random.default_rng(seed)
    k=np.linspace(-1.5,1.5,n)
    iv=0.2+0.1*(k**2)+rng.normal(0,0.005,size=n)
    iv=np.clip(iv,0.05,1.0)
    return {"k":k,"iv":iv}
