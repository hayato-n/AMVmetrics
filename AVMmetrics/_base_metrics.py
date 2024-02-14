import numpy as np
from numpy.typing import NDArray


class _metrics_base(object):
    def __init__(self) -> None:
        self.metric_class = None
        self.abbreviation = None
        self.symmetry_bias = False
        self.symmetry_dispersion = False

    def __call__(self, y_true: NDArray, y_pred: NDArray) -> float:
        d = self._error(y_true, y_pred)
        e = self._loss(d)
        return self._summarize(e)

    def _error(self, y_true: NDArray, y_pred: NDArray) -> NDArray:
        raise NotImplementedError()

    def _loss(self, diff: NDArray) -> NDArray:
        raise NotImplementedError()

    def _summarize(self, err: NDArray) -> float:
        raise NotImplementedError()
