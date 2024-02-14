import numpy as np

from ._base_metrics import _metrics_base


class _absolute_ratio_base(_metrics_base):
    def __init__(self) -> None:
        super().__init__()
        self.metric_class = "Absolute Ratio"


class MeanAbsolutePredictionError(_absolute_ratio_base):
    def __init__(self) -> None:
        super().__init__()
        self.abbreviation = "MAPE"

        self._error = lambda x, y: x / y - 1
        self._loss = lambda d: np.abs(d)
        self._summarize = lambda e: np.mean(e)


class MeanAbsolutePredictionError2(_absolute_ratio_base):
    def __init__(self) -> None:
        super().__init__()
        self.abbreviation = "MAPE`"

        self._error = lambda x, y: 1 - y / x
        self._loss = lambda d: np.abs(d)
        self._summarize = lambda e: np.mean(e)


class MedianAbsolutePredictionError(_absolute_ratio_base):
    def __init__(self) -> None:
        super().__init__()
        self.abbreviation = "MDAPE"

        self._error = lambda x, y: x / y - 1
        self._loss = lambda d: np.abs(d)
        self._summarize = lambda e: np.median(e)


class symmetricMeanAbsolutePredictionError(_absolute_ratio_base):
    def __init__(self) -> None:
        super().__init__()
        self.abbreviation = "sMAPE"
        self.symmetry_dispersion = True

        self._error = lambda x, y: np.abs(x - y) / (x + y)
        self._loss = lambda d: d
        self._summarize = lambda e: np.mean(e)


class symmetricMedianAbsolutePredictionError(_absolute_ratio_base):
    def __init__(self) -> None:
        super().__init__()
        self.abbreviation = "sMDAPE"
        self.symmetry_dispersion = True

        self._error = lambda x, y: np.abs(x - y) / (x + y)
        self._loss = lambda d: d
        self._summarize = lambda e: np.median(e)


class CoefficientOfDispersion(_absolute_ratio_base):
    def __init__(self) -> None:
        super().__init__()
        self.abbreviation = "COD"

        self._error = lambda x, y: x / y / np.median(x / y) - 1
        self._loss = lambda d: np.abs(d)
        self._summarize = lambda e: np.mean(e)


class CoefficientOfDispersion2(_absolute_ratio_base):
    def __init__(self) -> None:
        super().__init__()
        self.abbreviation = "COD`"

        self._error = lambda x, y: y / x / np.median(y / x) - 1
        self._loss = lambda d: np.abs(d)
        self._summarize = lambda e: np.mean(e)


class LogMeanAbsolutePredictionError(_absolute_ratio_base):
    def __init__(self) -> None:
        super().__init__()
        self.abbreviation = "LMAPE"
        self.symmetry_dispersion = True

        self._error = lambda x, y: np.log(x / y)
        self._loss = lambda d: np.abs(d)
        self._summarize = lambda e: np.mean(e)


class MaxMinMeanAbsolutePredictionError(_absolute_ratio_base):
    def __init__(self) -> None:
        super().__init__()
        self.abbreviation = "mmMAPE"
        self.symmetry_dispersion = True

        self._error = lambda x, y: np.maximum(x, y) / np.minimum(x, y) - 1
        self._loss = lambda d: d
        self._summarize = lambda e: np.mean(e)


class DiewertMetric1(_absolute_ratio_base):
    def __init__(self) -> None:
        super().__init__()
        self.abbreviation = "DM1"
        self.symmetry_dispersion = True

        self._error = lambda x, y: x / y + y / x - 2
        self._loss = lambda d: np.abs(d)
        self._summarize = lambda e: np.mean(e)


_full = [
    MeanAbsolutePredictionError(),
    MeanAbsolutePredictionError2(),
    MedianAbsolutePredictionError(),
    symmetricMeanAbsolutePredictionError(),
    symmetricMedianAbsolutePredictionError(),
    CoefficientOfDispersion(),
    CoefficientOfDispersion2(),
    LogMeanAbsolutePredictionError(),
    MaxMinMeanAbsolutePredictionError(),
    DiewertMetric1(),
]

