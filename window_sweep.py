import pandas as pd
from pathlib import Path

from methods import AdaptiveBayesian
from evaluate import walk_forward, score

DATA_DIR = Path('data')
RESULTS_DIR = Path('results')
RESULTS_DIR.mkdir(exist_ok=True)

returns = pd.read_csv(DATA_DIR / 'returns.csv', index_col=0, parse_dates=True)
ASSET_CLASS = {
    'SPY': 'US Equity', 'QQQ': 'US Equity', 'IWM': 'US Equity', 'DIA': 'US Equity',
    'XLF': 'US Equity', 'XLE': 'US Equity', 'XLK': 'US Equity', 'XLV': 'US Equity',
    'XLU': 'US Equity', 'XLP': 'US Equity',
    'EFA': 'Intl Equity', 'EEM': 'Intl Equity', 'EWJ': 'Intl Equity', 'FXI': 'Intl Equity',
    'TLT': 'Fixed Income', 'IEF': 'Fixed Income', 'LQD': 'Fixed Income',
    'HYG': 'Fixed Income', 'TIP': 'Fixed Income',
    'GLD': 'Commodity', 'SLV': 'Commodity', 'USO': 'Commodity', 'DBC': 'Commodity',
    'VNQ': 'Real Estate',
}
tickers = [ticker for ticker in returns.columns if ticker in ASSET_CLASS]

WINDOWS = [10, 25, 50, 100, 250]
FORGETTING_FACTORS = [0.95, 0.98, 0.99, 1.0]

rows = []
for window in WINDOWS:
    for forgetting_factor in FORGETTING_FACTORS:
        for ticker in tickers:
            asset_returns = returns[ticker].dropna()
            method = AdaptiveBayesian(
                window=window,
                pre_period=max(window, 50),
                forgetting_factor=forgetting_factor,
            )
            forecasts = walk_forward(
                asset_returns,
                [method],
                min_history=max(300, 2 * window),
            )
            scores = score(forecasts)
            bay_name = method.name
            rows.append({
                'ticker': ticker,
                'asset_class': ASSET_CLASS[ticker],
                'window': window,
                'forgetting_factor': forgetting_factor,
                'ls_sharpe': scores.loc[bay_name, 'ls_sharpe'],
                'hit_rate': scores.loc[bay_name, 'hit_rate'],
                'rmse': scores.loc[bay_name, 'rmse'],
            })

pd.DataFrame(rows).to_csv(RESULTS_DIR / 'window_sweep.csv', index=False)
