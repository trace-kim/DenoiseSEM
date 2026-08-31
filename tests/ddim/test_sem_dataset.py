import argparse
import tempfile
import unittest
from pathlib import Path

import torch
from PIL import Image

from ddim.datasets import get_dataset
from ddim.datasets.sem import SEMImageDataset


def namespace(**kwargs):
    return argparse.Namespace(**kwargs)


class SEMImageDatasetTests(unittest.TestCase):
    def create_images(self, root, count, nested=False):
        destination = root / "nested" if nested else root
        destination.mkdir(parents=True, exist_ok=True)
        for index in range(count):
            image = Image.new("L", (19, 13), color=index * 10)
            image.save(destination / "image_{:02d}.png".format(index))

    def test_loads_grayscale_images_in_sorted_order(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.create_images(root, 3)
            dataset = SEMImageDataset(root, channels=1)

            self.assertEqual(len(dataset), 3)
            self.assertEqual(dataset.files[0].name, "image_00.png")
            image, target = dataset[0]
            self.assertEqual(image.mode, "L")
            self.assertEqual(target, 0)

    def test_recursive_loading_is_explicit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.create_images(root, 2, nested=True)

            with self.assertRaises(RuntimeError):
                SEMImageDataset(root, recursive=False)

            dataset = SEMImageDataset(root, recursive=True)
            self.assertEqual(len(dataset), 2)

    def test_optional_memory_cache_avoids_reopening_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.create_images(root, 2)
            dataset = SEMImageDataset(root, channels=1, cache_in_memory=True)

            self.assertEqual(len(dataset.cached_images), 2)
            (root / "image_00.png").unlink()
            image, target = dataset[0]
            self.assertEqual(image.mode, "L")
            self.assertEqual(target, 0)

    def test_get_dataset_uses_yaml_directory_and_deterministic_split(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.create_images(root, 10)
            config = namespace(
                data=namespace(
                    dataset="SEM",
                    data_dir=str(root),
                    image_size=16,
                    channels=1,
                    random_flip=False,
                    validation_split=0.2,
                    split_seed=123,
                    recursive=False,
                )
            )
            args = namespace(exp=str(root / "unused_exp"))

            train_dataset, test_dataset = get_dataset(args, config)
            self.assertEqual(len(train_dataset), 8)
            self.assertEqual(len(test_dataset), 2)

            image, target = train_dataset[0]
            self.assertIsInstance(image, torch.Tensor)
            self.assertEqual(tuple(image.shape), (1, 16, 16))
            self.assertEqual(target, 0)

            second_train_dataset, second_test_dataset = get_dataset(args, config)
            self.assertEqual(train_dataset.indices, second_train_dataset.indices)
            self.assertEqual(test_dataset.indices, second_test_dataset.indices)


if __name__ == "__main__":
    unittest.main()
