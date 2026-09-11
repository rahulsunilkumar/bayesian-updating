import pickle
from pathlib import Path

import pandas as pd

from evaluate import walk_forward, score, score_by_regime
from methods import default_methods

DATA_DIR = Path('data')
RESULTS_DIR = Path('results')
RESULTS_DIR.mkdir(exist_ok=True)

TICKERS = [
    'SPY', 'QQQ', 'IWM', 'DIA', 'XLF', 'XLE', 'XLK', 'XLV', 'XLU', 'XLP',
    'EFA', 'EEM', 'EWJ', 'FXI', 'TLT', 'IEF', 'LQD', 'HYG', 'TIP',
    'GLD', 'SLV', 'USO', 'DBC', 'VNQ',
]

returns = pd.read_csv(DATA_DIR / 'returns.csv', index_col=0, parse_dates=True)
returns = returns[[ticker for ticker in TICKERS if ticker in returns.columns]]

per_asset_forecasts = {}
per_asset_scores = {}
per_asset_regime = {}

for ticker in returns.columns:
    asset_returns = returns[ticker].dropna()
    if len(asset_returns) < 500:
        continue
    forecasts = walk_forward(asset_returns, default_methods(), min_history=300)
    per_asset_forecasts[ticker] = forecasts
    per_asset_scores[ticker] = score(forecasts)
    per_asset_regime[ticker] = score_by_regime(forecasts)

with open(RESULTS_DIR / 'forecasts.pkl', 'wb') as file:
    pickle.dump(per_asset_forecasts, file)
with open(RESULTS_DIR / 'scores.pkl', 'wb') as file:
    pickle.dump(per_asset_scores, file)
with open(RESULTS_DIR / 'regime.pkl', 'wb') as file:
    pickle.dump(per_asset_regime, file)

all_scores = pd.concat(per_asset_scores, names=['ticker', 'method'])
mean_summary = all_scores.groupby('method').mean()
mean_summary.to_csv(RESULTS_DIR / 'aggregate_means.csv')
