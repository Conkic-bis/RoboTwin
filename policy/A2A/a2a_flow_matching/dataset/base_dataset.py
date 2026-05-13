from typing import Dict

import torch

from a2a_flow_matching.common.normalizer import LinearNormalizer


class BaseLowdimDataset(torch.utils.data.Dataset):
    def get_validation_dataset(self) -> "BaseLowdimDataset":
        return BaseLowdimDataset()

    def get_normalizer(self, **kwargs) -> LinearNormalizer:
        raise NotImplementedError()

    def get_all_actions(self) -> torch.Tensor:
        raise NotImplementedError()

    def __len__(self) -> int:
        return 0

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        raise NotImplementedError()


class BaseImageDataset(torch.utils.data.Dataset):
    def get_validation_dataset(self) -> "BaseImageDataset":
        return BaseImageDataset()

    def get_normalizer(self, **kwargs) -> LinearNormalizer:
        raise NotImplementedError()

    def get_all_actions(self) -> torch.Tensor:
        raise NotImplementedError()

    def __len__(self) -> int:
        return 0

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        raise NotImplementedError()
