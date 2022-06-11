from typing import List, Dict

import numpy as np
import torch


def dataset_split(
    dataset: torch.utils.data.Dataset,
    split: List[float] = [0.9, 0.1],
    random_train_val_split: bool = False,
) -> Dict[str, torch.utils.data.dataset.Subset]:
    """split torch.utils.data.Dataset by given split ratio.
    Written by Jungbae Park.

    Args:
        dataset (torch.utils.data.Dataset): input interaction dataset.
        split (List[float], optional): split ratio.
            len(split) should be in [2, 3] & sum(split) should be 1.0
            if len(split) == 2:
                return {
                    "train": train_dataset,
                    "val": val_dataset
                }
            elif len(split) == 3:
                return {
                    "train": train_dataset,
                    "val": val_dataset,
                    "test": test_dataset
                }
            Defaults to [0.9, 0.1].
        random_train_val_split (bool, optional):
            if it's True, will randomly mix mix of train, val indices.
                In that case, test_dataset will remain
                    as the last portion of all_dataset.
            else, will keep order for splits (sequential split)
            Defaults to False.
    Returns:
        Dict[str, torch.utils.data.dataset.Subset]:
            return subset of datasets as dictionaries.
            i.e. {
                "train": train_dataset,
                "val": val_dataset,
                "test": test_dataset
            }
    """
    assert len(split) in [2, 3]
    assert sum(split) == 1.0
    for frac in split:
        assert frac >= 0.0

    indices = list(range(len(dataset)))
    modes = ["train", "val", "test"][: len(split)]
    sizes = np.array(np.cumsum([0] + list(split)) * len(dataset), dtype=int)
    # sizes = [0, train_size, train_size+val_size, len(datasets)]
    if random_train_val_split:
        train_and_val_idx = indices[: sizes[2]]
        random.shuffle(train_and_val_idx)
        indices = train_and_val_idx + indices[sizes[2] :]

    datasets = {
        mode: torch.utils.data.Subset(
            dataset, indices[sizes[i] : sizes[i + 1]]
        )
        for i, mode in enumerate(modes)
    }

    return datasets
