import tempfile
import unittest
from pathlib import Path

import pandas as pd
from PIL import Image

from data.build_sid_manifests import (
    cache_samples_and_build_rows,
    collect_balanced,
    sample_balanced_reservoir,
    split_balanced_validation_pool,
)
from data.validate_manifests import validate_manifest_frame, validate_no_overlap


def sample(image_id: str, label: int):
    return {
        "img_id": image_id,
        "image": Image.new("RGB", (8, 6), (label * 100, 10, 20)),
        "label": label,
        "width": 8,
        "height": 6,
    }


class ManifestBuildingTests(unittest.TestCase):
    def test_collect_balanced_skips_tampered_and_duplicates(self):
        stream = [
            sample("t", 2),
            sample("r0", 0),
            sample("r0", 0),
            sample("f0", 1),
            sample("r1", 0),
            sample("f1", 1),
        ]
        selected = collect_balanced(stream, per_class=2)
        labels = [int(item["label"]) for item in selected]
        self.assertEqual(labels.count(0), 2)
        self.assertEqual(labels.count(1), 2)
        self.assertNotIn(2, labels)
        self.assertEqual(len({item["img_id"] for item in selected}), 4)

    def test_reservoir_sampling_is_balanced_deterministic_and_excludes_tampered(self):
        stream = [sample("tampered", 2)]
        for i in range(10):
            stream.append(sample(f"r{i}", 0))
            stream.append(sample(f"f{i}", 1))

        first = sample_balanced_reservoir(
            stream,
            per_class=3,
            candidate_pool_per_class=8,
            seed=42,
        )
        second = sample_balanced_reservoir(
            stream,
            per_class=3,
            candidate_pool_per_class=8,
            seed=42,
        )

        first_ids = [item["img_id"] for item in first]
        second_ids = [item["img_id"] for item in second]
        labels = [int(item["label"]) for item in first]

        self.assertEqual(first_ids, second_ids)
        self.assertEqual(labels.count(0), 3)
        self.assertEqual(labels.count(1), 3)
        self.assertNotIn("tampered", first_ids)
        self.assertEqual(len(first_ids), len(set(first_ids)))

        # The bounded pool contains only r0-r7 and f0-f7.
        self.assertTrue(
            all(
                int(image_id[1:]) < 8
                for image_id in first_ids
            )
        )

    def test_reservoir_sampling_rejects_pool_smaller_than_quota(self):
        with self.assertRaisesRegex(
            ValueError,
            "candidate_pool_per_class must be >= per_class",
        ):
            sample_balanced_reservoir(
                [sample("r0", 0), sample("f0", 1)],
                per_class=2,
                candidate_pool_per_class=1,
                seed=42,
            )

    def test_split_balanced_validation_pool_has_no_overlap(self):
        pool = [sample(f"r{i}", 0) for i in range(4)] + [sample(f"f{i}", 1) for i in range(4)]
        val, test = split_balanced_validation_pool(
            pool, validation_per_class=2, test_per_class=2
        )
        self.assertEqual(len(val), 4)
        self.assertEqual(len(test), 4)
        self.assertFalse({x["img_id"] for x in val} & {x["img_id"] for x in test})

    def test_cache_is_png_and_validation_checks_balance(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows = cache_samples_and_build_rows(
                [sample("r", 0), sample("f", 1)],
                manifest_split="train",
                source_split="train",
                cache_root=root / "data/cache/sid",
                project_root=root,
            )
            self.assertTrue(all(str(row["cached_path"]).endswith(".png") for row in rows))
            frame = pd.DataFrame(rows)
            counts = validate_manifest_frame(
                frame, split_name="train", project_root=root, verify_images=True
            )
            self.assertEqual(counts, {"real": 1, "fake": 1})

    def test_overlap_detection(self):
        frame_a = pd.DataFrame({"image_id": ["x", "a"]})
        frame_b = pd.DataFrame({"image_id": ["x", "b"]})
        with self.assertRaisesRegex(ValueError, "overlap"):
            validate_no_overlap({"train": frame_a, "val": frame_b})


if __name__ == "__main__":
    unittest.main()
