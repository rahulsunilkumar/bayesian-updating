# Bayesian Updating for Daily Return Forecasts

How much history should a return forecast retain? This project compares an adaptive Bayesian model with six simple forecasting rules across 24 ETFs using daily returns from 2010–2024. I look at forecast accuracy, trading performance, and how the results change across assets and volatility levels.

## Findings

- The default Bayesian model gets direction right 51.36% of the time, but its average trading Sharpe is only 0.012. AR(1) performs better, at 0.204.
- Predicting zero produces lower root mean squared error than the default Bayesian model on every asset.
- Bayesian performance weakens in high volatility. No forgetting gives the highest average Sharpe at every tested window, suggesting that retaining information helped in this sample.

[Read the paper](writeup/writeup.pdf) for the mathematics, results, and limitations.
