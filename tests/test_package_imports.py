import importlib
import sys
import unittest


class PackageImportTests(unittest.TestCase):
    def test_model_and_training_packages_import_without_tensorflow_side_effect(self):
        sys.modules.pop("nowcasting.models", None)
        sys.modules.pop("nowcasting.training", None)
        sys.modules.pop("tensorflow", None)

        importlib.import_module("nowcasting.models")
        importlib.import_module("nowcasting.training")

        self.assertNotIn("tensorflow", sys.modules)


if __name__ == "__main__":
    unittest.main()
