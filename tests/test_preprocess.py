import numpy as np

from abmil_oct.preprocess import normalize_image


def test_percentile_normalization_preserves_background_and_clips():
    image = np.concatenate(
        [np.zeros(10), np.arange(1, 101), np.array([10000.0])]
    ).reshape(1, -1)
    normalized = normalize_image(image)

    assert normalized.dtype == np.float32
    assert np.all(normalized[image == 0] == 0)
    assert normalized.min() == 0
    assert normalized.max() == 1
    assert np.all(np.isfinite(normalized))


def test_normalization_handles_invalid_frame():
    image = np.array([[np.nan, np.inf], [0.0, -1.0]])
    assert np.array_equal(normalize_image(image), np.zeros_like(image))
