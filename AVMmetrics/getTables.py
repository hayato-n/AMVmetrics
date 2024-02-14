import numpy as np
import pandas as pd

from . import (
    AverageBias,
    AbsoluteDifference,
    SquaredDifference,
    AbsoluteRatio,
    SquaredRatio,
    PercentageErrorRanges,
    Quantile,
)


def _getTables(y_true, y_pred, metrics_list):
    idx = pd.MultiIndex.from_arrays(
        [
            [m.metric_class for m in metrics_list],
            [m.abbreviation for m in metrics_list],
        ],
        names=["Class", "Metric"],
    )
    ds = pd.DataFrame(
        index=idx,
        columns=["score", "symmetry in bias", "symmetry in dispersion"],
        # dtype=np.float64,
    )

    for m in metrics_list:
        ds.loc[(m.metric_class, m.abbreviation)] = (
            m(y_true, y_pred),
            m.symmetry_bias,
            m.symmetry_dispersion,
        )

    return ds


def getMetrics(y_true, y_pred):
    return _getTables(
        y_true,
        y_pred,
        metrics_list=[
            AverageBias.LogMedianPredictionError(),
            AbsoluteDifference.MeanAbsoluteError(),
            AbsoluteRatio.MaxMinMeanAbsolutePredictionError(),
            SquaredRatio.LogRootMeanSquaredError(),
            SquaredDifference.RootMeanSquaredError(),
            PercentageErrorRanges.MaxMinPercentageErrorRange(10),
            Quantile.InterQuartileRangeInRatios(),
        ],
    )


def getConventionalMetrics(y_true, y_pred):
    return _getTables(
        y_true,
        y_pred,
        metrics_list=[
            AverageBias.MeanPredictionError(),
            AbsoluteDifference.MeanAbsoluteError(),
            AbsoluteRatio.MeanAbsolutePredictionError(),
            AbsoluteRatio.CoefficientOfDispersion(),
            SquaredDifference.RootMeanSquaredError(),
            SquaredDifference.CoefficientOfDetermination(reverse=True),
        ],
    )


def getFullMetrics(y_true, y_pred):
    return _getTables(
        y_true,
        y_pred,
        metrics_list=AverageBias._full
        + AbsoluteDifference._full
        + SquaredDifference._full
        + AbsoluteRatio._full
        + SquaredRatio._full
        + PercentageErrorRanges._full
        + Quantile._full,
    )


def getFullMetricsSymmetric(y_true, y_pred):
    full_list = (
        AverageBias._full
        + AbsoluteDifference._full
        + SquaredDifference._full
        + AbsoluteRatio._full
        + SquaredRatio._full
        + PercentageErrorRanges._full
        + Quantile._full
    )

    metrics_list = []
    for m in full_list:
        if m.symmetry_bias == True or m.symmetry_dispersion == True:
            metrics_list.append(m)

    return _getTables(y_true, y_pred, metrics_list=metrics_list)
