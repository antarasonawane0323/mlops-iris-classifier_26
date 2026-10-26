import numpy as np
from sklearn.ensemble import RandomForestClassifier


def create_sample_data():
    rng = np.random.default_rng(42)

    X = rng.random((30, 4))
    y = rng.integers(0, 3, size=30)

    return X, y


def test_model_trains_without_error():
    X, y = create_sample_data()

    model = RandomForestClassifier(
        n_estimators=10,
        random_state=42
    )

    model.fit(X, y)

    assert model is not None


def test_model_predict_output_shape():
    X, y = create_sample_data()

    model = RandomForestClassifier(
        n_estimators=10,
        random_state=42
    )

    model.fit(X, y)
    predictions = model.predict(X)

    assert predictions.shape == (30,)


def test_model_predict_proba_sums_to_one():
    X, y = create_sample_data()

    model = RandomForestClassifier(
        n_estimators=10,
        random_state=42
    )

    model.fit(X, y)
    probabilities = model.predict_proba(X)

    assert np.allclose(probabilities.sum(axis=1), 1.0)