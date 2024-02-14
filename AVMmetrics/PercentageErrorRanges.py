import numpy as np

from ._base_metrics import _metrics_base


class _percentage_range_base(_metrics_base):
    def __init__(self, percentage: float) -> None:
        super().__init__()
        self.metric_class = "Percentage Ratio"
        self.percentage = percentage

        self._summarize = lambda e: np.count_nonzero(100 * e > self.percentage) / len(e)


class PercentageErrorRange(_percentage_range_base):
    def __init__(self, percentage: float) -> None:
        super().__init__(percentage)
        self.abbreviation = "PER" + f"({percentage})"

        self._error = lambda x, y: x / y - 1
        self._loss = lambda d: np.abs(d)


class PercentageErrorRange2(_percentage_range_base):
    def __init__(self, percentage: float) -> None:
        super().__init__(percentage)
        self.abbreviation = "PER`" + f"({self.percentage})"

        self._error = lambda x, y: y / x - 1
        self._loss = lambda d: np.abs(d)


class LogPercentageErrorRange(_percentage_range_base):
    def __init__(self, percentage: float) -> None:
        super().__init__(percentage)
        self.abbreviation = "LPER" + f"({self.percentage})"
        self.symmetry_dispersion = True

        self._error = lambda x, y: np.log(x / y)
        self._loss = lambda d: np.abs(d)


class MaxMinPercentageErrorRange(_percentage_range_base):
    def __init__(self, percentage: float) -> None:
        super().__init__(percentage)
        self.abbreviation = "mmPER" + f"({self.percentage})"
        self.symmetry_dispersion = True

        self._error = lambda x, y: np.maximum(x, y) / np.minimum(x, y) - 1
        self._loss = lambda d: np.abs(d)


_full = [
    PercentageErrorRange(10),
    PercentageErrorRange(20),
    PercentageErrorRange(30),
    PercentageErrorRange2(10),
    PercentageErrorRange2(20),
    PercentageErrorRange2(30),
    LogPercentageErrorRange(10),
    LogPercentageErrorRange(20),
    LogPercentageErrorRange(30),
    MaxMinPercentageErrorRange(10),
    MaxMinPercentageErrorRange(20),
    MaxMinPercentageErrorRange(30),
]
