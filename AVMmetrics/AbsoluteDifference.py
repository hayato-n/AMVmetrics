import numpy as np

from ._base_metrics import _metrics_base


class _absolute_error_base(_metrics_base):
    def __init__(self) -> None:
        super().__init__()
        self.metric_class = "Absolute Difference"


class MeanAbsoluteError(_absolute_error_base):
    def __init__(self) -> None:
        super().__init__()
        self.abbreviation = "MAE"
        self.symmetry_dispersion = True

        self._error = lambda x, y: x - y
        self._loss = lambda d: np.abs(d)
        self._summarize = lambda e: np.mean(e)


class MedianAbsoluteError(_absolute_error_base):
    def __init__(self) -> None:
        super().__init__()
        self.abbreviation = "MDAE"
        self.symmetry_dispersion = True

        self._error = lambda x, y: x - y
        self._loss = lambda d: np.abs(d)
        self._summarize = lambda e: np.median(e)


_full = [MeanAbsoluteError(), MedianAbsoluteError()]

