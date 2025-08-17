# MACD（Moving Average Convergence/Divergence）
MACD is a momentum and trend-following indicator developed by Gerald Appel.
Parameters: Short EMA (12), Long EMA (26), Signal EMA (9).
Formulas: DIF = EMA(Close, 12) - EMA(Close, 26); DEM = EMA(DIF, 9); MACD = DIF - DEM.
