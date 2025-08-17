from __future__ import annotations
import math

def _phi(x): return 0.5*(1.0+math.erf(x/math.sqrt(2.0)))

def d1(S,K,T,r,s): return (math.log(S/K)+(r+0.5*s*s)*T)/(s*math.sqrt(T))

def bs_call_price(S,K,T,r,s):
    D1=d1(S,K,T,r,s); D2=D1-s*math.sqrt(T)
    return S*_phi(D1)-K*math.exp(-r*T)*_phi(D2)
