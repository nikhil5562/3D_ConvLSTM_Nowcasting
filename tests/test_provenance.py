import os
import shutil
import tempfile
import unittest
from pathlib import Path

from nowcasting.data.provenance import (
    validate_reusable_artifact,
    write_artifact_manifest,
)
from nowcasting.data.tensorize import _tensor_provenance


class ProvenanceTests(unittest.TestCase):
    def test_source_identity_is_stable_across_paths_and_mtime_changes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = root / "machine_a" / "cube.nc"
            second = root / "machine_b" / "cube.nc"
            first.parent.mkdir()
            second.parent.mkdir()
            first.write_bytes(b"identical cube bytes")
            shutil.copyfile(first, second)
            os.utime(second, (first.stat().st_atime + 100, first.stat().st_mtime + 100))

            self.assertEqual(_tensor_provenance(first), _tensor_provenance(second))

    def test_artifact_manifest_remains_valid_after_content_preserving_copy(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first_dir = root / "machine_a"
            second_dir = root / "machine_b"
            first_dir.mkdir()
            second_dir.mkdir()
            artifact = first_dir / "frame.npy"
            companion = first_dir / "coverage.npy"
            artifact.write_bytes(b"tensor bytes")
            companion.write_bytes(b"coverage bytes")
            provenance = {"pipeline_version": "test-v1", "source": {"sha256": "abc"}}
            manifest = write_artifact_manifest(artifact, provenance, [companion])

            copied_artifact = second_dir / artifact.name
            copied_companion = second_dir / companion.name
            copied_manifest = second_dir / manifest.name
            shutil.copyfile(artifact, copied_artifact)
            shutil.copyfile(companion, copied_companion)
            shutil.copyfile(manifest, copied_manifest)
            os.utime(copied_artifact, (artifact.stat().st_atime + 100, artifact.stat().st_mtime + 100))

            payload = validate_reusable_artifact(
                copied_artifact,
                provenance,
                related_directory=second_dir,
            )

        self.assertEqual(payload["provenance"], provenance)


if __name__ == "__main__":
    unittest.main()
