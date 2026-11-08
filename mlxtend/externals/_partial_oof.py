"""Out-of-fold predictions for non-overlapping, partial test partitions."""

import warnings
from collections.abc import Iterable

import numpy as np
from joblib import Parallel, delayed
from sklearn import get_config
from sklearn.base import clone
from sklearn.model_selection import TimeSeriesSplit
from sklearn.utils import _safe_indexing
from sklearn.utils.metaestimators import _safe_split


def prepare_oof(cv, X, y, groups, original_cv):
    # Keep ordinary splitter execution (including independent random draws per
    # base estimator) on scikit-learn's existing cross_val_predict path.
    if not isinstance(cv, TimeSeriesSplit) and not isinstance(original_cv, Iterable):
        return cv, np.arange(len(y))
    splits = list(cv.split(X, y, groups))
    if not splits:
        raise ValueError("Stacking requires at least one cross-validation split.")
    for train, test in splits:
        for indices in (train, test):
            indices = np.asarray(indices)
            if (
                indices.ndim != 1
                or indices.dtype.kind not in "iu"
                or not len(indices)
                or np.any(indices < 0)
                or np.any(indices >= len(y))
            ):
                raise ValueError(
                    "Cross-validation indices must be nonempty valid integer indices."
                )
        if np.intersect1d(train, test).size:
            raise ValueError(
                "Cross-validation training and test sets must not overlap."
            )
    tests = np.concatenate([test for _, test in splits])
    if len(np.unique(tests)) != len(tests):
        raise ValueError(
            "Stacking requires non-overlapping cross-validation test sets."
        )
    rows = np.sort(tests)
    if len(rows) != len(y):
        if get_config().get("enable_metadata_routing", False):
            raise ValueError(
                "Partial cross-validation does not yet support metadata routing."
            )
        warnings.warn(
            "Only samples included in cross-validation test sets are used to "
            "train the meta-estimator. This is expected for TimeSeriesSplit.",
            UserWarning,
        )
    return splits, rows


def _fit_predict(model, X, y, train, test, sample_weight, method, classes):
    model = clone(model)
    X_train, y_train = _safe_split(model, X, y, train)
    X_test, _ = _safe_split(model, X, y, test, train)
    params = {}
    if sample_weight is not None:
        params["sample_weight"] = _safe_indexing(sample_weight, train)
    model.fit(X_train, y_train, **params)
    prediction = getattr(model, method)(X_test)
    if method == "predict_proba":
        if isinstance(prediction, list):
            raise ValueError(
                "Partial cross-validation requires single-output classification."
            )
        aligned = np.zeros((len(test), len(classes)), dtype=prediction.dtype)
        aligned[:, np.searchsorted(classes, model.classes_)] = prediction
        prediction = aligned
    return prediction


def partial_oof(
    model, X, y, splits, sample_weight, method, n_jobs, verbose, pre_dispatch
):
    classes = np.unique(y) if method == "predict_proba" else None
    predictions = Parallel(n_jobs=n_jobs, verbose=verbose, pre_dispatch=pre_dispatch)(
        delayed(_fit_predict)(model, X, y, train, test, sample_weight, method, classes)
        for train, test in splits
    )
    order = np.argsort(np.concatenate([test for _, test in splits]))
    return np.concatenate(predictions)[order]
