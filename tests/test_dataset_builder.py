import pytest
from torch.utils.data import Dataset

from src.data.dataset_builder import build_weighted_sampler, get_class_distribution


class DummyDataset(Dataset):
    def __init__(self, targets):
        self.targets = targets

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        return index, self.targets[index]


def test_get_class_distribution_counts_labels():
    dataset = DummyDataset([0, 0, 1, 3])

    distribution = get_class_distribution(dataset, ["0", "1", "2", "3"])

    assert distribution == {"0": 2, "1": 1, "2": 0, "3": 1}


def test_build_weighted_sampler_gives_higher_weight_to_rare_class():
    dataset = DummyDataset([0, 0, 0, 1])

    sampler = build_weighted_sampler(dataset)
    weights = sampler.weights.tolist()

    assert weights[0] == pytest.approx(weights[1])
    assert weights[0] == pytest.approx(weights[2])
    assert weights[3] > weights[0]
