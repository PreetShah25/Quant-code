# preet-quant-ai-codes

High-signal quant + AI monorepo with clean code, tests, and CI.
Pin a few of these and add a short profile README to look battle-tested.

## Projects
- **jvol-smile** — SVI-based implied vol smile fitting + Black–Scholes greeks (NumPy only).
- **factorlab** — Toy factor engine (MKT/SMB/HML/MOM) from CSVs.
- **altdata-sentiment** — Transparent rule-based headline sentiment CLI.
- **volnet** — Small MLP to learn (k -> IV) from synthetic smiles (scikit-learn).
- **rl-exec** — Order execution environment with Almgren–Chriss-style costs + TWAP baseline.

## Dev quickstart
```bash
# pick a project
cd volnet
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pytest -q
```
