"""Loader behaviour, with emphasis on the rules that protect measurements."""

from __future__ import annotations

import numpy as np
import pytest
from synthetic_images import gaussian_blurred_step, write_png, write_rgb_with_border

from sem_segment.image_io import (
    discover_images,
    load_image,
    pixel_hash,
    read_native,
    to_measure01,
    to_model_rgb,
)


def test_eight_bit_png_round_trips_to_unit_range(tmp_path):
    image = gaussian_blurred_step()
    path = write_png(tmp_path / "step.png", image)
    loaded = load_image(path)
    assert loaded.native.dtype == np.uint8
    assert loaded.white_level == 255.0
    assert loaded.measure01.min() >= 0.0 and loaded.measure01.max() <= 1.0
    np.testing.assert_allclose(loaded.measure01, np.round(image * 255) / 255, atol=1e-12)


def test_sixteen_bit_png_uses_the_sixteen_bit_white_level(tmp_path):
    image = gaussian_blurred_step()
    path = write_png(tmp_path / "step16.png", image, bits=16)
    loaded = load_image(path)
    assert loaded.native.dtype == np.uint16
    assert loaded.white_level == 65535.0
    np.testing.assert_allclose(loaded.measure01, image, atol=1e-4)


def test_detector_levels_rescale_the_measurement_range(tmp_path):
    path = write_png(tmp_path / "step.png", gaussian_blurred_step())
    default = load_image(path)
    twelve_bit = load_image(path, black_level=0.0, white_level=4095.0)
    # A 12-bit detector stored in a wider container maps to a smaller fraction.
    assert twelve_bit.measure01.max() < default.measure01.max()


def test_crop_is_applied_before_the_grayscale_check(tmp_path):
    """The real-instrument case: a colored frame around a grayscale interior.

    Checking channel equality on the whole file would reject an image whose
    imaging region is perfectly grayscale, which is how the only real SEM data
    in this repository is exported.
    """
    image = gaussian_blurred_step(height=64, width=64)
    path = write_rgb_with_border(tmp_path / "framed.tif", image, border=2)

    with pytest.raises(ValueError, match="RGB channels differ"):
        load_image(path)

    loaded = load_image(path, crop=(2, 62, 2, 62))
    assert loaded.shape == (60, 60)


def test_grayscale_rejection_names_the_crop_remedy(tmp_path):
    path = write_rgb_with_border(tmp_path / "framed.png", gaussian_blurred_step(height=32, width=32))
    with pytest.raises(ValueError) as error:
        read_native(path)
    message = str(error.value)
    assert "input.crop" in message and "--crop" in message
    assert "luminance" in message


def test_rgb_with_identical_channels_is_accepted_without_conversion(tmp_path):
    from PIL import Image

    gray = np.round(gaussian_blurred_step(height=32, width=32) * 255).astype(np.uint8)
    path = tmp_path / "rgb.png"
    Image.fromarray(np.repeat(gray[:, :, None], 3, axis=2), mode="RGB").save(path)
    loaded = load_image(path)
    np.testing.assert_array_equal(loaded.native, gray)


def test_crop_outside_the_image_is_rejected(tmp_path):
    path = write_png(tmp_path / "small.png", gaussian_blurred_step(height=32, width=32))
    with pytest.raises(ValueError, match="extends outside"):
        load_image(path, crop=(0, 64, 0, 64))


def test_tiny_images_are_rejected(tmp_path):
    path = write_png(tmp_path / "tiny.png", np.zeros((8, 8)))
    with pytest.raises(ValueError, match="at least 16 px"):
        load_image(path)


def test_unsupported_extension_is_rejected(tmp_path):
    path = tmp_path / "notes.txt"
    path.write_text("not an image", encoding="utf-8")
    with pytest.raises(ValueError, match="unsupported extension"):
        load_image(path)


def test_to_measure01_rejects_inverted_levels():
    with pytest.raises(ValueError, match="white_level must exceed"):
        to_measure01(np.zeros((4, 4)), black_level=10.0, white_level=5.0)


def test_model_rgb_never_feeds_back_into_the_measurement(tmp_path):
    """The load-bearing invariant: stretching is for the model only."""
    path = write_png(tmp_path / "low_contrast.png", gaussian_blurred_step(low=0.45, high=0.55))
    loaded = load_image(path)
    before = loaded.measure01.copy()

    plain = to_model_rgb(loaded.measure01, contrast_stretch=None)
    stretched = to_model_rgb(loaded.measure01, contrast_stretch=(1.0, 99.0))

    # The stretch changes what the model sees...
    assert np.ptp(stretched) > np.ptp(plain)
    # ...and leaves the measurement array bit-identical.
    np.testing.assert_array_equal(loaded.measure01, before)


def test_model_rgb_is_three_identical_uint8_channels():
    rgb = to_model_rgb(gaussian_blurred_step(height=16, width=16))
    assert rgb.dtype == np.uint8 and rgb.shape[2] == 3
    np.testing.assert_array_equal(rgb[..., 0], rgb[..., 1])
    np.testing.assert_array_equal(rgb[..., 0], rgb[..., 2])


def test_pixel_hash_distinguishes_shape_from_content():
    array = np.arange(12, dtype=np.uint8)
    assert pixel_hash(array.reshape(3, 4)) != pixel_hash(array.reshape(4, 3))
    assert pixel_hash(array.reshape(3, 4)) == pixel_hash(array.reshape(3, 4).copy())


def test_discover_images_handles_a_file_and_a_folder(tmp_path):
    first = write_png(tmp_path / "a.png", np.zeros((32, 32)))
    write_png(tmp_path / "b.png", np.zeros((32, 32)))
    (tmp_path / "ignore.txt").write_text("x", encoding="utf-8")

    assert discover_images(first) == [first]
    assert [p.name for p in discover_images(tmp_path)] == ["a.png", "b.png"]


def test_discover_images_rejects_an_empty_folder(tmp_path):
    (tmp_path / "empty").mkdir()
    with pytest.raises(ValueError, match="no supported images"):
        discover_images(tmp_path / "empty")
