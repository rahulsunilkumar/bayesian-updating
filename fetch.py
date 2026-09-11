import yfinance as yf
import pandas as pd
from pathlib import Path

DATA_DIR = Path('data')
DATA_DIR.mkdir(exist_ok=True)

TICKERS = [
    'SPY', 'QQQ', 'IWM', 'DIA',
    'XLF', 'XLE', 'XLK', 'XLV', 'XLU', 'XLP',       # us equity broad + sectors
    'EFA', 'EEM', 'EWJ', 'FXI',                     # international
    'TLT', 'IEF', 'LQD', 'HYG', 'TIP',              # fixed income
    'GLD', 'SLV', 'USO', 'DBC',                     # commodities
    'VNQ', 'VXX',                                   # real estate + vol
]

raw = yf.download(TICKERS, start='2010-01-01', end='2024-12-31', auto_adjust=True, progress=False)
prices = raw['Close'].dropna(how='all')
returns = prices.pct_change().dropna(how='all')

prices.to_csv(DATA_DIR / 'prices.csv')
returns.to_csv(DATA_DIR / 'returns.csv')

print('saved data/prices.csv and data/returns.csv')
