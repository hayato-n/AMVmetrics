import numpy as np

from ._base_metrics import _metrics_base


class _squared_ratio_base(_metrics_base):
    def __init__(self) -> None:
        super().__init__()
        self.metric_class = "Squared Ratio"


class MeanSquaredPredictionError(_squared_ratio_base):
    def __init__(self) -> None:
        super().__init__()
        self.abbreviation = "MSPE"

        self._error = lambda x, y: x / y - 1
        self._loss = lambda d: np.square(d)
        self._summarize = lambda e: np.mean(e)


class MeanSquaredPredictionError2(_squared_ratio_base):
    def __init__(self) -> None:
        super().__init__()
        self.abbreviation = "MSPE`"

        self._error = lambda x, y: y / x - 1
        self._loss = lambda d: np.square(d)
        self._summarize = lambda e: np.mean(e)


class LogStandardDeviationOfTheErrors(_squared_ratio_base):
    def __init__(self) -> None:
        super().__init__()
        self.abbreviation = "LSDE"
        self.symmetry_dispersion = True

        self._error = lambda x, y: np.log(x / y)
        self._loss = lambda d: d
        self._summarize = lambda e: np.std(e)


class LogRootMeanSquaredError(_squared_ratio_base):
    def __init__(self) -> None:
        super().__init__()
        self.abbreviation = "LRMSE"
        self.symmetry_dispersion = True

        self._error = lambda x, y: np.log(x / y)
        self._loss = lambda d: np.square(d)
        self._summarize = lambda e: np.sqrt(np.mean(e))


class LogMeanSquaredPredictionError(_squared_ratio_base):
    def __init__(self) -> None:
        super().__init__()
        self.abbreviation = "LMSPE"
        self.symmetry_dispersion = True

        self._error = lambda x, y: np.log(x / y)
        self._loss = lambda d: np.square(d)
        self._summarize = lambda e: np.mean(e)


class MaxMinMeanSquaredPredictionError(_squared_ratio_base):
    def __init__(self) -> None:
        super().__init__()
        self.abbreviation = "mmMSPE"
        self.symmetry_dispersion = True

        self._error = lambda x, y: np.maximum(x, y) / np.minimum(x, y) - 1
        self._loss = lambda d: np.square(d)
        self._summarize = lambda e: np.mean(e)


class DiewertMetric2(_squared_ratio_base):
    def __init__(self) -> None:
        super().__init__()
        self.abbreviation = "DM2"
        self.symmetry_dispersion = True

        self._error = lambda x, y: np.square(x / y - 1) + np.square(y / x - 1)
        self._loss = lambda d: d
        self._summarize = lambda e: np.mean(e)


_full = [
    MeanSquaredPredictionError(),
    MeanSquaredPredictionError2(),
    LogStandardDeviationOfTheErrors(),
    LogRootMeanSquaredError(),
    LogMeanSquaredPredictionError(),
    MaxMinMeanSquaredPredictionError(),
    DiewertMetric2(),
]

