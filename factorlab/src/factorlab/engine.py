from __future__ import annotations
import argparse, pandas as pd, numpy as np

def _parse_args():
    ap=argparse.ArgumentParser(); ap.add_argument("--prices",required=True)
    ap.add_argument("--rf",required=True); ap.add_argument("--out",required=True)
    return ap.parse_args()

def compute_daily_returns(df: pd.DataFrame)->pd.DataFrame:
    df=df.sort_values(["symbol","date"]).copy()
    df["ret"]=df.groupby("symbol")["close"].pct_change()
    return df.dropna(subset=["ret"])

def _split_equal(g):
    idx=g.sort_values("symbol").index; mid=len(idx)//2
    return g.loc[idx[:mid]], g.loc[idx[mid:]]

def bucket_by_size(df: pd.DataFrame, date_col="date"):
    res=[]
    for dt,g in df.groupby(date_col):
        if "shares" in g:
            g=g.copy(); g["market_cap"]=g["shares"]*g["close"]
            thr=g["market_cap"].quantile(0.5)
            small=g[g["market_cap"]<=thr]; big=g[g["market_cap"]>thr]
        else:
            small,big=_split_equal(g)
        res.append((dt,small,big))
    return res

def bucket_by_value(df: pd.DataFrame, date_col="date"):
    res=[]
    for dt,g in df.groupby(date_col):
        if set(["book_equity","shares"]).issubset(g.columns):
            g=g.copy(); g["market_cap"]=g["shares"]*g["close"]
            g["b_m"]=g["book_equity"]/g["market_cap"].replace(0,np.nan)
            lo=g["b_m"].quantile(0.3); hi=g["b_m"].quantile(0.7)
            value=g[g["b_m"]>=hi]; growth=g[g["b_m"]<=lo]
        else:
            idx=g.sort_values("symbol").index; third=len(idx)//3
            growth=g.loc[idx[:third]]; value=g.loc[idx[-third:]]
        res.append((dt,value,growth))
    return res

def compute_factors(prices: pd.DataFrame, rf: pd.DataFrame)->pd.DataFrame:
    ret=compute_daily_returns(prices)
    ret=ret.merge(rf, on="date", how="left").fillna({"rf":0.0})
    ret["momentum"]=ret.groupby("symbol")["ret"].apply(lambda s: s.shift(5).rolling(60).mean())
    smb_list=[{"date":dt,"SMB":s["ret"].mean()-b["ret"].mean()} for dt,s,b in bucket_by_size(ret)]
    hml_list=[{"date":dt,"HML":v["ret"].mean()-g["ret"].mean()} for dt,v,g in bucket_by_value(ret)]
    mkt=ret.groupby("date")["ret"].mean().rename("MKT").reset_index()
    mom=ret.groupby("date")["momentum"].mean().rename("MOM").reset_index()
    out=mkt.merge(mom, on="date", how="left")
    out=out.merge(pd.DataFrame(smb_list), on="date", how="left").merge(pd.DataFrame(hml_list), on="date", how="left")
    out=out.merge(rf, on="date", how="left").fillna({"rf":0.0})
    for col in ["MKT","SMB","HML","MOM"]: out[col]=out[col]-out["rf"]
    return out[["date","MKT","SMB","HML","MOM"]].sort_values("date")

def main():
    a=_parse_args(); prices=pd.read_csv(a.prices, parse_dates=["date"])
    rf=pd.read_csv(a.rf, parse_dates=["date"]); res=compute_factors(prices, rf)
    res.to_csv(a.out, index=False); print(f"Wrote {len(res)} rows to {a.out}")
if __name__=="__main__": main()
