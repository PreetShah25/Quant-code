from __future__ import annotations
import argparse, pandas as pd
LEXICON={"beat":1.0,"surge":0.8,"record":0.7,"strong":0.6,"upbeat":0.6,
        "miss":-1.0,"plunge":-0.9,"fraud":-0.9,"weak":-0.6,"downgrade":-0.7,
        "growth":0.3,"decline":-0.3,"profit":0.4,"loss":-0.4,"guidance":0.1}

def score_text(t:str)->float:
    t=(t or "").lower(); s=0.0
    for w,v in LEXICON.items():
        if w in t: s+=v
    return max(min(s,2.0),-2.0)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", dest="out", required=True); args=ap.parse_args()
    df=pd.read_csv(args.inp); df["sentiment"]=df["headline"].astype(str).apply(score_text)
    df.to_csv(args.out, index=False); print(f"Wrote {len(df)} rows to {args.out}")
if __name__=="__main__": main()
