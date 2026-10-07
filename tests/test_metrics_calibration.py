import numpy as np

from src.training.calibration import (binary_brier, brier_score, expected_calibration_error, fit_temperature, plot_reliability,
                                      reliability_curve, sigmoid, softmax)
from src.training.metrics import emotion_metrics, risk_metrics, sentiment_metrics, tune_emotion_thresholds


def test_sentiment_metrics_perfect_and_confusion():
    y = np.array([0, 1, 2, 0, 1, 2])
    m = sentiment_metrics(y, np.eye(3)[y])
    assert m["accuracy"] == 1.0 and m["macro_f1"] == 1.0 and np.trace(np.array(m["confusion_matrix"])) == 6
    assert set(m["per_class"]) == {"negative", "neutral", "positive"}


def test_risk_metrics_known_values():
    y = np.array([0, 0, 1, 1])
    p = np.array([[.9, .1], [.6, .4], [.4, .6], [.2, .8]])
    m = risk_metrics(y, p)
    assert m["roc_auc"] == 1.0 and m["pr_auc"] == 1.0 and m["recall"] == 1.0 and m["precision"] == 1.0
    p2 = np.array([[.1, .9], [.1, .9], [.9, .1], [.9, .1]])  # inverted
    assert risk_metrics(y, p2)["roc_auc"] == 0.0
    assert risk_metrics(np.zeros(4, int), p)["roc_auc"] is None  # single class: undefined, not a crash


def test_emotion_metrics_respect_mask():
    y = np.array([[1, -1, 0, 0, 0, 0], [0, -1, 0, 0, 1, 0], [1, -1, 0, 0, 0, 0]])
    probs = np.array([[.9, .9, .1, .1, .1, .1], [.1, .9, .1, .1, .9, .1], [.9, .9, .1, .1, .1, .1]])
    m = emotion_metrics(y, probs)
    assert m["per_class"]["disgust"]["f1"] is None and m["per_class"]["disgust"]["n_annotated"] == 0  # never scored
    assert m["per_class"]["anger"]["f1"] == 1.0 and m["macro_f1"] is not None
    thr = tune_emotion_thresholds(y, probs)
    assert thr.shape == (6,) and np.all((thr >= 0.1) & (thr <= 0.9))


def test_ece_brier_extremes():
    y = np.array([0, 1, 1, 0])
    perfect = np.eye(2)[y]
    assert expected_calibration_error(perfect, y) == 0.0 and brier_score(perfect, y) == 0.0
    overconfident_wrong = np.eye(2)[1 - y]
    assert expected_calibration_error(overconfident_wrong, y) == 1.0
    assert abs(binary_brier(np.array([0.5, 0.5]), np.array([0, 1])) - 0.25) < 1e-9


def test_temperature_scaling_reduces_overconfidence():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 600)
    logits = np.stack([-(2 * y - 1) * 0.0, (2 * y - 1) * 4.0], 1) * 1.0
    flip = rng.random(600) < 0.3  # 30% of the confident predictions are wrong
    logits[flip] = logits[flip][:, ::-1]
    T = fit_temperature(logits, y)
    assert T > 1.5
    assert expected_calibration_error(softmax(logits, T), y) < expected_calibration_error(softmax(logits), y)
    assert np.allclose(softmax(logits).sum(1), 1) and 0 < sigmoid(np.array([0.0]))[0] == 0.5


def test_reliability_curve_and_plot(tmp_path):
    p = np.linspace(0, 1, 101); y = (p > 0.5).astype(int)
    c = reliability_curve(p, y)
    assert sum(c["count"]) == 101 and len(c["mean_predicted"]) == 10
    plot_reliability({"m": c}, tmp_path / "r.png")
    assert (tmp_path / "r.png").stat().st_size > 1000
