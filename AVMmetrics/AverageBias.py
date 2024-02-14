import numpy as np

from ._base_metrics import _metrics_base



class _average_bias_base(_metrics_base):
    def __init__(self) -> None:
        super().__init__()
        self.metric_class = "Average Bias"


class MeanBiasError(_average_bias_base):
    def __init__(self) -> None:
        super().__init__()
        self.abbreviation = "MBE"
        self.symmetry_bias = True

        self._error = lambda x, y: x - y
        self._loss = lambda d: d
        self._summarize = lambda e: np.mean(e)


class MedianBiasError(_average_bias_base):
    def __init__(self) -> None:
        super().__init__()
        self.abbreviation = "MDBE"
        self.symmetry_bias = True

        self._error = lambda x, y: x - y
        self._loss = lambda d: d
        self._summarize = lambda e: np.median(e)


class MeanPredictionError(_average_bias_base):
    def __init__(self) -> None:
        super().__init__()
        self.abbreviation = "MPE"

        self._error = lambda x, y: x / y - 1
        self._loss = lambda d: d
        self._summarize = lambda e: np.mean(e)


class MeanPredictionError2(_average_bias_base):
    def __init__(self) -> None:
        super().__init__()
        self.abbreviation = "MPE`"

        self._error = lambda x, y: y / x - 1
        self._loss = lambda d: d
        self._summarize = lambda e: np.mean(e)


class MedianPredictionError(_average_bias_base):
    def __init__(self) -> None:
        super().__init__()
        self.abbreviation = "MDPE"

        self._error = lambda x, y: x / y - 1
        self._loss = lambda d: d
        self._summarize = lambda e: np.median(e)


# class MedianPredictionError2(_average_bias_base):
#     def __init__(self) -> None:
#         super().__init__()
#         self.abbreviation = "MDPE2"

#         self._error = lambda x, y: y / x - 1
#         self._loss = lambda d: d
#         self._summarize = lambda e: np.median(e)


class LogMeanPredictionError(_average_bias_base):
    def __init__(self) -> None:
        super().__init__()
        self.abbreviation = "LMPE"
        self.symmetry_bias = True

        self._error = lambda x, y: np.log(x / y)
        self._loss = lambda d: d
        self._summarize = lambda e: np.mean(e)


class LogMedianPredictionError(_average_bias_base):
    def __init__(self) -> None:
        super().__init__()
        self.abbreviation = "LMDPE"
        self.symmetry_bias = True

        self._error = lambda x, y: np.log(x / y)
        self._loss = lambda d: d
        self._summarize = lambda e: np.median(e)

_full = [
    MeanBiasError(),
    MedianBiasError(),
    MeanPredictionError(),
    MeanPredictionError2(),
    MedianPredictionError(),
    LogMeanPredictionError(),
    LogMedianPredictionError(),
]
