import pandas as pd
from factorlab.engine import compute_factors

def test_compute_factors_minimal():
    dates=pd.date_range("2024-01-01", periods=10, freq="D")
    prices=pd.DataFrame({
        "date": list(dates)*2,
        "symbol": ["AAA"]*10+["BBB"]*10,
        "close": [100,101,100,102,103,103,104,106,107,108]*2,
        "shares": [1_000_000]*20,
        "book_equity": [50_000_000]*20
    })
    rf=pd.DataFrame({"date": dates, "rf":[0.0]*10})
    fac=compute_factors(prices, rf)
    assert {"MKT","SMB","HML","MOM"}.issubset(set(fac.columns))
