import pickle
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

mpl.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'axes.labelsize': 10,
    'axes.titlesize': 11,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'savefig.dpi': 150,
    'savefig.bbox': 'tight',
})

PALETTE = {
    'Zero': '#888888',
    'Random Walk': '#B0B0B0',
    'Rolling Mean (50d)': '#5B7FB2',
    'Rolling Mean (250d)': '#7BA3D4',
    'EWMA (hl=20d)': '#9BC1E8',
    'AR(1) (250d)': '#C99356',
    'Adaptive Bayesian (w=50d)': '#A6334C',
}

BAYESIAN_COLOR = '#A6334C'
NEUTRAL_COLOR = '#B0B0B0'
REGIME_COLOR = '#5B7FB2'

# fundamentals written, aesthetics spruced up by AI

def plot_sharpe_by_method(results_df, out_path, title=None):
    results = results_df.dropna(subset=['ls_sharpe']).sort_values('ls_sharpe')
    fig, ax = plt.subplots(figsize=(6.5, 0.4 * len(results) + 1))
    colors = [
        BAYESIAN_COLOR if 'Bayesian' in method else NEUTRAL_COLOR
        for method in results.index
    ]
    ax.barh(results.index, results['ls_sharpe'], color=colors, edgecolor='white')
    ax.axvline(0, color='black', linewidth=0.5)
    ax.set_xlabel('Long-short Sharpe ratio (annualized)')
    if title:
        ax.set_title(title, loc='left')
    plt.savefig(out_path)
    plt.close()


def plot_metric_grid(per_asset_results, metric, out_path, title=None):
    records = []
    for ticker, results in per_asset_results.items():
        for method in results.index:
            records.append({
                'ticker': ticker,
                'method': method,
                'value': results.loc[method, metric],
            })

    values = pd.DataFrame(records).pivot(
        index='ticker', columns='method', values='value'
    )
    fig, ax = plt.subplots(figsize=(8, 0.3 * len(values) + 1.5))
    limit = np.nanpercentile(values.values, [5, 95])
    color_limit = max(abs(limit[0]), abs(limit[1]))
    image = ax.imshow(
        values.values,
        cmap='RdBu_r',
        aspect='auto',
        vmin=-color_limit,
        vmax=color_limit,
    )
    ax.set_xticks(range(len(values.columns)))
    ax.set_xticklabels(values.columns, rotation=35, ha='right', fontsize=8)
    ax.set_yticks(range(len(values.index)))
    ax.set_yticklabels(values.index, fontsize=8)

    for row in range(len(values.index)):
        for column in range(len(values.columns)):
            value = values.values[row, column]
            if not np.isnan(value):
                ax.text(column, row, f'{value:.2f}', ha='center', va='center', fontsize=7)

    colorbar = plt.colorbar(image, ax=ax, shrink=0.6)
    colorbar.set_label(metric)
    if title:
        ax.set_title(title, loc='left')
    plt.savefig(out_path)
    plt.close()


def plot_regime_comparison(regime_results, metric, out_path, title=None):
    low = regime_results['low_vol'][metric]
    high = regime_results['high_vol'][metric]
    methods = sorted(set(low.index) & set(high.index))

    fig, ax = plt.subplots(figsize=(7.5, 0.35 * len(methods) + 1))
    positions = np.arange(len(methods))
    bar_height = 0.4
    ax.barh(
        positions - bar_height / 2,
        [low[method] for method in methods],
        bar_height,
        label='Low vol',
        color=REGIME_COLOR,
        edgecolor='white',
    )
    ax.barh(
        positions + bar_height / 2,
        [high[method] for method in methods],
        bar_height,
        label='High vol',
        color=BAYESIAN_COLOR,
        edgecolor='white',
    )
    ax.set_yticks(positions)
    ax.set_yticklabels(methods, fontsize=9)
    ax.axvline(0, color='black', linewidth=0.5)
    ax.set_xlabel(metric)
    ax.legend(frameon=False, fontsize=9)
    if title:
        ax.set_title(title, loc='left')
    plt.savefig(out_path)
    plt.close()


def plot_parameter_sensitivity(sweep_results, metric, out_path, title=None):
    values = sweep_results.pivot_table(
        values=metric,
        index='forgetting_factor',
        columns='window',
        aggfunc='mean',
    )
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    image = ax.imshow(values.values, cmap='RdBu_r', aspect='auto')
    ax.set_xticks(range(len(values.columns)))
    ax.set_xticklabels(values.columns)
    ax.set_yticks(range(len(values.index)))
    ax.set_yticklabels(values.index)
    ax.set_xlabel('Window size (days)')
    ax.set_ylabel('Forgetting factor')

    for row in range(len(values.index)):
        for column in range(len(values.columns)):
            value = values.iloc[row, column]
            if not np.isnan(value):
                ax.text(column, row, f'{value:.2f}', ha='center', va='center')

    colorbar = plt.colorbar(image, ax=ax)
    colorbar.set_label(metric)
    if title:
        ax.set_title(title, loc='left')
    plt.savefig(out_path)
    plt.close()


def plot_forecast_vs_actual(
    forecast_df,
    method_name,
    out_path,
    title=None,
    sample_window=None,
):
    values = forecast_df[[method_name, 'actual']].dropna()
    if sample_window:
        values = values.iloc[-sample_window:]

    fig, ax = plt.subplots(figsize=(7.5, 3.5))
    ax.plot(values.index, values['actual'], color=NEUTRAL_COLOR, linewidth=0.6, label='Actual')
    ax.plot(values.index, values[method_name], color=BAYESIAN_COLOR, linewidth=1.0, label=method_name)
    ax.axhline(0, color='black', linewidth=0.3)
    ax.set_ylabel('Daily return')
    ax.legend(frameon=False, fontsize=9)
    if title:
        ax.set_title(title, loc='left')
    plt.savefig(out_path)
    plt.close()

RESULTS_DIR = Path('results')
FIGURES_DIR = Path('figures')
FIGURES_DIR.mkdir(exist_ok=True)

with open(RESULTS_DIR / 'forecasts.pkl', 'rb') as file:
    forecasts_by_asset = pickle.load(file)

with open(RESULTS_DIR / 'scores.pkl', 'rb') as file:
    scores_by_asset = pickle.load(file)

with open(RESULTS_DIR / 'regime.pkl', 'rb') as file:
    regimes_by_asset = pickle.load(file)

aggregate_scores = pd.concat(scores_by_asset, names=['ticker', 'method'])
mean_scores = aggregate_scores.groupby('method').mean()

plot_sharpe_by_method(
    mean_scores,
    FIGURES_DIR / 'sharpe_by_method.png',
    title='Average trading performance by method',
)

plot_metric_grid(
    scores_by_asset,
    'ls_sharpe',
    FIGURES_DIR / 'sharpe_by_asset.png',
    title='Long-short Sharpe by asset and method',
)

regime_tables = {}
for regime in ['low_vol', 'high_vol']:
    tables = [
        asset_regimes[regime]
        for asset_regimes in regimes_by_asset.values()
        if regime in asset_regimes
    ]
    regime_tables[regime] = pd.concat(tables).groupby(level=0).mean()

plot_regime_comparison(
    regime_tables,
    'ls_sharpe',
    FIGURES_DIR / 'sharpe_by_regime.png',
    title='Long-short Sharpe by volatility regime',
)

sweep_results = pd.read_csv(RESULTS_DIR / 'window_sweep.csv')
plot_parameter_sensitivity(
    sweep_results,
    'ls_sharpe',
    FIGURES_DIR / 'parameter_sensitivity.png',
    title='Bayesian sensitivity to window and forgetting factor',
)

plot_forecast_vs_actual(
    forecasts_by_asset['SPY'],
    'Adaptive Bayesian (w=50d)',
    FIGURES_DIR / 'bayesian_forecast_spy.png',
    title='Adaptive Bayesian forecast for SPY',
    sample_window=250,
)
