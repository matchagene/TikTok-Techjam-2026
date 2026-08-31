import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from evaluation.model_adapter import ModelAdapter
from evaluation.robustness import predict_condition


class OneBatchDataset(Dataset):
    def __len__(self):
        return 2

    def __getitem__(self, index):
        return {
            "image": torch.zeros(3, 4, 4),
            "label": index,
            "image_id": f"id{index}",
            "condition_id": "clean",
            "corruption": "clean",
            "severity": "none",
            "seed": 1 + index,
        }


class CountingModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def forward(self, x):
        self.calls += 1
        return torch.tensor([[-1.0], [1.0]], device=x.device)[: len(x)]


def test_predict_condition_forwards_once_per_batch():
    model = CountingModel()
    loader = DataLoader(OneBatchDataset(), batch_size=2, shuffle=False)
    frame = predict_condition(
        ModelAdapter(model, "M1"), loader, device=torch.device("cpu"), dataset_name="test"
    )
    assert model.calls == 1
    assert list(frame["label"]) == [0, 1]
    assert frame.loc[0, "p_fake"] < frame.loc[1, "p_fake"]
