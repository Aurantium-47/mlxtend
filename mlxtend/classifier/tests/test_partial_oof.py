import numpy as np
import pytest
from scipy import sparse
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import KFold, TimeSeriesSplit, cross_val_predict
from sklearn.tree import DecisionTreeClassifier

from mlxtend.classifier import StackingCVClassifier
from mlxtend.regressor import StackingCVRegressor


class RecordingRegressor(RegressorMixin, BaseEstimator):
    def fit(self, X, y, sample_weight=None):
        self.X_ = X.copy()
        self.y_ = y.copy()
        self.weights_ = sample_weight
        return self

    def predict(self, X):
        return np.zeros(len(X))


@pytest.mark.parametrize("secondary", [False, True])
@pytest.mark.parametrize("sparse_input", [False, True])
def test_regression_alignment(secondary, sparse_input):
    X = np.arange(48).reshape(24, 2).astype(float)
    if sparse_input:
        X = sparse.csr_matrix(X)
    y = np.arange(24) ** 2
    weights = np.arange(24) + 1
    cv = TimeSeriesSplit(n_splits=3, gap=2, max_train_size=8)
    splits = list(cv.split(X))
    rows = np.concatenate([te for tr, te in splits])
    expected = np.concatenate(
        [
            LinearRegression()
            .fit(X[tr], y[tr], sample_weight=weights[tr])
            .predict(X[te])
            for tr, te in splits
        ]
    )
    base = LinearRegression()
    model = StackingCVRegressor(
        [base],
        RecordingRegressor(),
        cv=cv,
        n_jobs=1,
        use_features_in_secondary=secondary,
        store_train_meta_features=True,
    )
    with pytest.warns(UserWarning, match="Only samples"):
        model.fit(X, y, sample_weight=weights)
    np.testing.assert_array_equal(model.train_meta_indices_, rows)
    np.testing.assert_allclose(model.train_meta_features_[:, 0], expected)
    np.testing.assert_array_equal(model.meta_regr_.y_, y[rows])
    np.testing.assert_array_equal(model.meta_regr_.weights_, weights[rows])
    assert model.meta_regr_.X_.shape == (len(rows), 3 if secondary else 1)
    assert not hasattr(base, "coef_")
    assert model.regr_[0].n_features_in_ == 2


@pytest.mark.parametrize("drop", [None, "first", "last"])
def test_missing_class_probability_alignment(drop):
    X = np.arange(24).reshape(-1, 1)
    y = np.array(["a"] * 8 + ["b"] * 8 + ["c"] * 8)
    model = StackingCVClassifier(
        [DummyClassifier(strategy="prior")],
        LogisticRegression(),
        cv=TimeSeriesSplit(n_splits=3),
        use_probas=True,
        drop_proba_col=drop,
        store_train_meta_features=True,
        n_jobs=1,
    )
    with pytest.warns(UserWarning, match="Only samples"):
        model.fit(X, y)
    expected = np.array(
        [[1, 0, 0]] * 6 + [[2 / 3, 1 / 3, 0]] * 6 + [[4 / 9, 4 / 9, 1 / 9]] * 6
    )
    if drop == "first":
        expected = expected[:, 1:]
    if drop == "last":
        expected = expected[:, :-1]
    np.testing.assert_allclose(model.train_meta_features_, expected)
    assert model.predict(X).shape == (24,)


def test_ordinary_cv_unchanged():
    X = np.arange(24).reshape(-1, 1)
    y = np.arange(24) % 2
    cv = KFold(n_splits=3, shuffle=True, random_state=4)
    base = DecisionTreeClassifier(random_state=3)
    expected = cross_val_predict(base, X, y, cv=cv, method="predict_proba")
    model = StackingCVClassifier(
        [base],
        LogisticRegression(),
        cv=cv,
        use_probas=True,
        store_train_meta_features=True,
    )
    model.fit(X, y)
    np.testing.assert_array_equal(model.train_meta_features_, expected)


def test_reordered_partial_folds_and_targets():
    X = np.arange(24).reshape(-1, 1)
    y = np.column_stack((np.arange(24), np.arange(24) ** 2))
    splits = list(TimeSeriesSplit(n_splits=3).split(X))[::-1]
    model = StackingCVRegressor(
        [LinearRegression()],
        RecordingRegressor(),
        cv=splits,
        multi_output=True,
        store_train_meta_features=True,
    )
    with pytest.warns(UserWarning, match="Only samples"):
        model.fit(X, y)
    np.testing.assert_array_equal(model.meta_regr_.y_, y[6:])
    assert model.train_meta_features_.shape == (18, 2)


def test_overlap_rejected():
    model = StackingCVRegressor(
        [LinearRegression()],
        LinearRegression(),
        cv=[([0, 1], [2, 3]), ([0, 1], [3, 4])],
    )
    with pytest.raises(ValueError):
        model.fit(np.arange(5).reshape(-1, 1), np.arange(5))


class TemporalRegressor(RegressorMixin, BaseEstimator):
    def fit(self, X, y, sample_weight=None):
        assert not hasattr(self, "last_"), "Fold estimator was reused"
        self.last_ = np.max(X[:, 0])
        return self

    def predict(self, X):
        assert self.last_ < np.min(X[:, 0]), "Future data leaked into fold training"
        return np.full(len(X), self.last_)


def test_temporal_leakage_and_independent_clones():
    X = np.arange(24).reshape(-1, 1)
    model = StackingCVRegressor(
        [TemporalRegressor()],
        RecordingRegressor(),
        cv=TimeSeriesSplit(n_splits=3, gap=2),
        store_train_meta_features=True,
    )
    with pytest.warns(UserWarning, match="Only samples"):
        model.fit(X, np.arange(24))
    np.testing.assert_array_equal(
        model.train_meta_features_[:, 0], np.repeat([3, 9, 15], 6)
    )
    assert model.regr_[0].last_ == 23


def test_classifier_weight_and_feature_alignment():
    X = np.arange(24).reshape(-1, 1)
    y = np.arange(24) % 2
    weights = np.arange(24) + 1
    model = StackingCVClassifier(
        [DummyClassifier(strategy="prior")],
        RecordingRegressor(),
        cv=TimeSeriesSplit(3),
        use_probas=True,
        use_features_in_secondary=True,
        store_train_meta_features=True,
    )
    with pytest.warns(UserWarning, match="Only samples"):
        model.fit(X, y, sample_weight=weights)
    np.testing.assert_array_equal(model.meta_clf_.y_, y[6:])
    np.testing.assert_array_equal(model.meta_clf_.weights_, weights[6:])
    np.testing.assert_array_equal(model.meta_clf_.X_[:, 0], X[6:, 0])
    for start, train_end in enumerate([6, 12, 18]):
        probs = np.bincount(y[:train_end], weights=weights[:train_end])
        expected = np.tile(probs / probs.sum(), (6, 1))
        np.testing.assert_allclose(
            model.train_meta_features_[start * 6 : (start + 1) * 6], expected
        )


@pytest.mark.parametrize("indices", [[-1, 11], [12], [1.5], []])
def test_invalid_partial_test_indices(indices):
    model = StackingCVRegressor(
        [LinearRegression()], LinearRegression(), cv=[([0, 1], indices)]
    )
    with pytest.raises(ValueError):
        model.fit(np.arange(12).reshape(-1, 1), np.arange(12))


def test_meta_class_coverage_required():
    X = np.arange(12).reshape(-1, 1)
    y = np.array([0, 0, 0, 1, 2, 1, 2, 1, 2, 1, 2, 1])
    model = StackingCVClassifier(
        [DummyClassifier()],
        LogisticRegression(),
        cv=TimeSeriesSplit(3),
        use_probas=True,
    )
    with pytest.warns(UserWarning, match="Only samples"):
        with pytest.raises(ValueError, match="every target class"):
            model.fit(X, y)


def test_metadata_routing_not_silently_bypassed():
    from sklearn import config_context

    model = StackingCVRegressor(
        [LinearRegression()], LinearRegression(), cv=TimeSeriesSplit(3)
    )
    with config_context(enable_metadata_routing=True):
        with pytest.raises(ValueError, match="metadata routing"):
            model.fit(
                np.arange(12).reshape(-1, 1), np.arange(12), sample_weight=np.ones(12)
            )


def test_unseeded_ordinary_splitter_preserves_per_estimator_draws():
    X = np.arange(48).reshape(-1, 1)
    y = np.arange(48) % 2
    cv = KFold(4, shuffle=True)
    bases = [
        DecisionTreeClassifier(random_state=1),
        DecisionTreeClassifier(random_state=2),
    ]
    state = np.random.get_state()
    try:
        np.random.seed(123)
        expected = np.column_stack(
            [
                cross_val_predict(base, X, y, cv=cv, method="predict_proba")
                for base in bases
            ]
        )
        np.random.seed(123)
        model = StackingCVClassifier(
            bases,
            LogisticRegression(),
            cv=cv,
            use_probas=True,
            store_train_meta_features=True,
        )
        model.fit(X, y)
        np.testing.assert_array_equal(model.train_meta_features_, expected)
    finally:
        np.random.set_state(state)


def test_grouped_explicit_partial_splits():
    X = np.arange(16).reshape(-1, 1)
    y = np.arange(16) ** 2
    groups = np.repeat(np.arange(4), 4)
    splits = [
        (np.flatnonzero(groups < g), np.flatnonzero(groups == g)) for g in [3, 1, 2]
    ]
    model = StackingCVRegressor(
        [LinearRegression()],
        RecordingRegressor(),
        cv=splits,
        store_train_meta_features=True,
    )
    with pytest.warns(UserWarning, match="Only samples"):
        model.fit(X, y, groups=groups)
    expected = np.concatenate(
        [
            LinearRegression().fit(X[train], y[train]).predict(X[test])
            for train, test in splits
        ]
    )[np.argsort(np.concatenate([test for _, test in splits]))]
    np.testing.assert_allclose(model.train_meta_features_[:, 0], expected)
    np.testing.assert_array_equal(model.train_meta_indices_, np.arange(4, 16))
