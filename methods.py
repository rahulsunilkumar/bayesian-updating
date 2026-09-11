import numpy as np


class RollingMean:
    def __init__(self, window=50):
        self.window = window
        self.name = f'Rolling Mean ({self.window}d)'

    def forecast(self, history):
        if len(history) < self.window:
            return np.nan
        return history[-self.window:].mean()


class EWMA:
    def __init__(self, halflife=20):
        self.halflife = halflife
        self.name = f'EWMA (hl={self.halflife}d)'

    def forecast(self, history):
        if len(history) == 0:
            return np.nan
        decay_factor = 0.5 ** (1 / self.halflife)
        history_length = len(history)
        ages = np.arange(history_length - 1, -1, -1)
        weights = decay_factor ** ages
        weights = weights / weights.sum()
        weighted_returns = weights * history
        return weighted_returns.sum()


class AR1:
    def __init__(self, window=250):
        self.window = window
        self.name = f'AR(1) ({self.window}d)'

    def forecast(self, history):
        if len(history) < self.window + 1:                              # one extra for window's lagged pairs
            return np.nan
        y = history[-self.window:]                                      # later returns
        x = history[-self.window - 1:-1]                                # earlier returns
        X = np.column_stack([np.ones(len(x)), x])
        least_squares_result = np.linalg.lstsq(X, y, rcond=None)
        beta = least_squares_result[0]
        return beta[0] + beta[1] * history[-1]


class RandomWalk:
    name = 'Random Walk'

    def forecast(self, history):
        if len(history) < 1:
            return np.nan
        return history[-1]


class ZeroForecast:
    name = 'Zero'

    def forecast(self, history):
        return 0.0


class AdaptiveBayesian:
    def __init__(self, window=50, pre_period=50, forgetting_factor=0.98):
        self.window = window
        self.pre_period = pre_period
        self.forgetting_factor = forgetting_factor
        self.name = f'Adaptive Bayesian (w={self.window}d)'
        self._mu_prior = None
        self._tau_prior_var = None

    def forecast(self, history):
        eps = 1e-10
        if len(history) < self.pre_period + self.window:
            return np.nan

        if self._mu_prior is None:
            pre = history[:self.pre_period]
            self._mu_prior = pre.mean()
            self._tau_prior_var = max(pre.var(ddof=1) / len(pre), eps)  # uncertainty in the mean

        window_data = history[-self.window:]
        sigma_data_var = max(window_data.var(ddof=1), eps)
        r_obs = history[-1]

        tau_post_var = 1.0 / (
            1.0 / self._tau_prior_var + 1.0 / sigma_data_var
        )
        mu_post = tau_post_var * (
            self._mu_prior / self._tau_prior_var
            + r_obs / sigma_data_var
        )

        self._mu_prior = mu_post                                        # carry posterior forward
        self._tau_prior_var = tau_post_var / self.forgetting_factor     # retain some uncertainty

        return mu_post

    def reset(self):
        self._mu_prior = None
        self._tau_prior_var = None


def default_methods():
    return [
        ZeroForecast(),
        RandomWalk(),
        RollingMean(window=50),
        RollingMean(window=250),
        EWMA(halflife=20),
        AR1(window=250),
        AdaptiveBayesian(window=50, pre_period=50),
    ]
