from pathlib import Path

import json
import torch
from PIL import Image

import inference


class DummyAdapter:
    def predict_fake_probability(self, images: torch.Tensor) -> torch.Tensor:
        return torch.full((images.shape[0], 1), 0.75, device=images.device)


def _make_image(path: Path) -> None:
    Image.new("RGB", (16, 16), (10, 20, 30)).save(path)


def test_discover_images_filters_and_sorts(tmp_path: Path):
    _make_image(tmp_path / "b.png")
    _make_image(tmp_path / "a.jpg")
    (tmp_path / "ignore.txt").write_text("x")

    paths = inference.discover_images(tmp_path)
    assert [p.name for p in paths] == ["a.jpg", "b.png"]


def test_run_directory_inference_writes_required_json(tmp_path: Path, monkeypatch):
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    _make_image(image_dir / "one.png")
    _make_image(image_dir / "two.jpg")
    checkpoint = tmp_path / "model.pth"
    checkpoint.write_bytes(b"placeholder")
    output = tmp_path / "predictions.json"

    monkeypatch.setattr(inference, "load_adapter", lambda *args, **kwargs: DummyAdapter())

    rows = inference.run_directory_inference(
        input_dir=image_dir,
        checkpoint=checkpoint,
        model_id="M3",
        output=output,
        batch_size=2,
        device_name="cpu",
    )

    assert len(rows) == 2
    parsed = json.loads(output.read_text())
    assert set(parsed[0]) == {"image_path", "pred"}
    assert all(abs(row["pred"] - 0.75) < 1e-8 for row in parsed)
