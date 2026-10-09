import numpy as np

from abmil_oct.features import patch_to_rgb_image


def test_patch_uint8_conversion_matches_manuscript():
    patch = np.array([[0.0, 0.5, 1.0]], dtype=np.float32)
    pixels = np.asarray(patch_to_rgb_image(patch))

    assert pixels.shape == (1, 3, 3)
    assert pixels[0, :, 0].tolist() == [0, 127, 255]
    assert np.array_equal(pixels[..., 0], pixels[..., 1])
    assert np.array_equal(pixels[..., 1], pixels[..., 2])
