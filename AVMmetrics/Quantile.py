import numpy as np

from ._base_metrics import _metrics_base


class _quantile_base(_metrics_base):
    def __init__(self, lower: float, upper: float) -> None:
        super().__init__()
        self.metric_class = "Quantile"

        self.lower = lower
        self.upper = upper

        self._loss = lambda d: d
        self._summarize = lambda e: np.percentile(e, upper) - np.percentile(e, lower)


class RangeInRatios(_quantile_base):
    def __init__(self, lower: float, upper: float) -> None:
        super().__init__(lower, upper)
        self.abbreviation = "Rat" + f"({self.lower}-{self.upper})"
        self.symmetry_dispersion = True

        self._error = lambda x, y: np.log(x / y)


class InterQuartileRangeInRatios(RangeInRatios):
    def __init__(self) -> None:
        super().__init__(lower=25, upper=75)
        self.abbreviation = "IQRat"


class Percentile9010InRatios(RangeInRatios):
    def __init__(self) -> None:
        super().__init__(lower=10, upper=90)
        self.abbreviation = "9010Rat"


class RangeInLevels(_quantile_base):
    def __init__(self, lower: float, upper: float) -> None:
        super().__init__(lower, upper)
        self.abbreviation = "Lev" + f"({self.lower}-{self.upper})"
        self.symmetry_dispersion = True

        self._error = lambda x, y: x - y


class InterQuartileRangeInLevels(RangeInLevels):
    def __init__(self) -> None:
        super().__init__(lower=25, upper=75)
        self.abbreviation = "IQLev"


class Percentile9010InLevels(RangeInLevels):
    def __init__(self) -> None:
        super().__init__(lower=10, upper=90)
        self.abbreviation = "9010Lev"


_full = [
    InterQuartileRangeInRatios(),
    Percentile9010InRatios(),
    InterQuartileRangeInLevels(),
    Percentile9010InLevels(),
]

