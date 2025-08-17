from __future__ import annotations
import numpy as np
from dataclasses import dataclass

@dataclass
class SVIParams:
    a: float; b: float; rho: float; m: float; sigma: float; T: float = 1.0

class SVIFitter:
    def __init__(self, T: float = 1.0):
        self.T = float(T); self.params: SVIParams | None = None
    @staticmethod
    def _tw(k, p):
        return p.a + p.b*(p.rho*(k-p.m) + np.sqrt((k-p.m)**2 + p.sigma**2))
    @staticmethod
    def _iv(k, p):
        w = SVIFitter._tw(k,p); w = np.maximum(w,1e-10); return np.sqrt(w/p.T)
    @staticmethod
    def _loss(k, iv, p):
        return float(np.mean((SVIFitter._iv(k,p)-iv)**2))
    def fit(self, k, iv, n_starts=128, seed=0):
        rng = np.random.default_rng(seed); best=None; best_loss=np.inf
        k = np.asarray(k,float); iv = np.asarray(iv,float)
        k_med=float(np.median(k)); iv_med=float(np.median(iv));
        for _ in range(n_starts):
            p = SVIParams(a=abs(rng.normal(0.02,0.02)), b=abs(rng.normal(0.5,0.2)),
                          rho=float(np.clip(rng.normal(0,0.5),-0.999,0.999)),
                          m=rng.normal(k_med,0.5), sigma=abs(rng.normal(0.5,0.2))+1e-3, T=self.T)
            # simple local search
            cur=p; cur_loss=self._loss(k,iv,cur)
            for _ in range(60):
                for name,scale in [("a",0.1),("b",0.1),("rho",0.05),("m",0.05),("sigma",0.05)]:
                    trial=SVIParams(**cur.__dict__)
                    step=rng.normal()*scale
                    if name=="rho": val=float(np.clip(getattr(trial,name)+step,-0.999,0.999))
                    else: val=float(abs(getattr(trial,name)+step))
                    setattr(trial,name,val)
                    loss=self._loss(k,iv,trial)
                    if loss<cur_loss: cur,cur_loss=trial,loss
            if cur_loss<best_loss: best, best_loss=cur, cur_loss
        self.params=best; return self
    def predict_iv(self,k):
        assert self.params is not None; return self._iv(np.asarray(k,float), self.params)
