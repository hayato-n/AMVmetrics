import numpy as np
from numpy.typing import NDArray
from ._base_metrics import _metrics_base


class _squared_error_base(_metrics_base):
    def __init__(self) -> None:
        super().__init__()
        self.metric_class = "Squared Difference"


class MeanSquaredError(_squared_error_base):
    def __init__(self) -> None:
        super().__init__()
        self.abbreviation = "MSE"
        self.symmetry_dispersion = True

        self._error = lambda x, y: x - y
        self._loss = lambda d: np.square(d)
        self._summarize = lambda e: np.mean(e)


class RootMeanSquaredError(_squared_error_base):
    def __init__(self) -> None:
        super().__init__()
        self.abbreviation = "RMSE"
        self.symmetry_dispersion = True

        self._error = lambda x, y: x - y
        self._loss = lambda d: np.square(d)
        self._summarize = lambda e: np.sqrt(np.mean(e))


class CoefficientOfDetermination(_squared_error_base):
    def __init__(self, reverse=True) -> None:
        super().__init__()
        self.reverse = reverse
        if self.reverse:
            self.abbreviation = "1 - R-squared"
        else:
            self.abbreviation = "R-squared"

    def __call__(self, y_true: NDArray, y_pred: NDArray) -> float:
        pred_loss = np.sum(np.square(y_true - y_pred))
        mean_loss = np.sum(np.square(y_true - np.mean(y_true)))
        if self.reverse:
            return pred_loss / mean_loss
        else:
            return 1 - pred_loss / mean_loss


class CorrelationCoefficient(_squared_error_base):
    def __init__(self, reverse=True) -> None:
        super().__init__()
        self.reverse = reverse
        if self.reverse:
            self.abbreviation = "1 - CC"
        else:
            self.abbreviation = "CC"

    def __call__(self, y_true: NDArray, y_pred: NDArray) -> float:
        cc = np.corrcoef(y_true, y_pred)[0, 1]
        if self.reverse:
            return 1 - cc
        else:
            return cc


class NormaliseRootMeanSquaredError(_squared_error_base):
    def __init__(self) -> None:
        super().__init__()
        self.abbreviation = "NRMSE"

    def __call__(self, y_true: NDArray, y_pred: NDArray) -> float:
        rmse = RootMeanSquaredError()(y_true, y_pred)
        return rmse / (np.max(y_true) - np.min(y_true))


class SignalNoiseRatio(_squared_error_base):
    def __init__(self) -> None:
        super().__init__()
        self.abbreviation = "SNR"

    def __call__(self, y_true: NDArray, y_pred: NDArray) -> float:
        return np.var(y_true - y_pred) / np.var(y_pred)


class StandardDeviationOfTheErrors(_squared_error_base):
    def __init__(self) -> None:
        super().__init__()
        self.abbreviation = "SDE"

    def __call__(self, y_true: NDArray, y_pred: NDArray) -> float:
        return np.std(y_true - y_pred)


_full = [
    MeanSquaredError(),
    RootMeanSquaredError(),
    CoefficientOfDetermination(reverse=True),
    CorrelationCoefficient(reverse=True),
    NormaliseRootMeanSquaredError(),
    SignalNoiseRatio(),
    StandardDeviationOfTheErrors(),
]

