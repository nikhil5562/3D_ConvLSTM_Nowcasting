import subprocess
import sys
import unittest


class PackageImportTests(unittest.TestCase):
    def test_model_and_training_packages_import_without_tensorflow_side_effect(self):
        result = subprocess.run(
            [sys.executable, "-c", (
                "import sys; import nowcasting.models; import nowcasting.training; "
                "assert 'tensorflow' not in sys.modules"
            )],
            capture_output=True, text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
