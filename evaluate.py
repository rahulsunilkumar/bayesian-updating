import numpy as np
import pandas as pd
from copy import deepcopy

TRADING_DAYS_PER_YEAR = 252


def walk_forward(returns, methods, min_history=300):
    methods = [deepcopy(m) for m in methods]
    for m in methods:
        if hasattr(m, 'reset'):
            m.reset()

    r = returns.dropna().values
    dates = returns.dropna().index
    T = len(r)

    forecasts = {m.name: np.full(T, np.nan) for m in methods}

    for t in range(min_history, T):
        history = r[:t]
        for m in methods:
            forecasts[m.name][t] = m.forecast(history)

    df = pd.DataFrame(forecasts, index=dates)
    df['actual'] = r
    return df


def score(forecast_df):
    actual = forecast_df['actual']
    method_cols = [c for c in forecast_df.columns if c != 'actual']

    rows = []
    for m in method_cols:
        f = forecast_df[m]
        valid = f.notna() & actual.notna()
        if valid.sum() < 50: # minimum sample-size
            continue
        fv = f[valid].values
        av = actual[valid].values

        rmse = np.sqrt(np.mean((fv - av) ** 2))
        mae = np.mean(np.abs(fv - av))

        nonzero = (fv != 0) & (av != 0)
        if nonzero.sum() > 0:
            hit_rate = np.mean(np.sign(fv[nonzero]) == np.sign(av[nonzero]))
        else:
            hit_rate = np.nan

        if np.all(fv == 0):
            ls_sharpe = np.nan
            ls_return = np.nan
        else:
            positions = np.sign(fv)
            strat_returns = positions * av
            mean_ret = strat_returns.mean()
            std_ret = strat_returns.std(ddof=1)
            ls_sharpe = mean_ret / std_ret * np.sqrt(TRADING_DAYS_PER_YEAR) if std_ret > 0 else np.nan
            ls_return = (1 + strat_returns).prod() ** (TRADING_DAYS_PER_YEAR / len(strat_returns)) - 1

        rows.append({
            'method': m,
            'rmse': rmse,
            'mae': mae,
            'hit_rate': hit_rate,
            'ls_sharpe': ls_sharpe,
            'ls_cagr': ls_return,
            'n_obs': valid.sum(),
        })

    return pd.DataFrame(rows).set_index('method')


def score_by_regime(forecast_df, vol_window=20): # for different eras of volatilty
    actual = forecast_df['actual']
    realized_vol = actual.rolling(vol_window).std().shift(1)
    vol_cutoff = realized_vol.median()

    high_vol_data = forecast_df[realized_vol > vol_cutoff]
    low_vol_data = forecast_df[realized_vol <= vol_cutoff]

    results = {}
    if len(high_vol_data) >= 100:
        results['high_vol'] = score(high_vol_data)
    if len(low_vol_data) >= 100:
        results['low_vol'] = score(low_vol_data)

    return results
