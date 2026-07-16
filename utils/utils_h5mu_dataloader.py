import torch
from torch.utils.data import DataLoader

from datasets.h5mu_dataset import H5MuDataManager


def h5mu_dataloader(args, manager: H5MuDataManager):
    """Build train/test DataLoaders for an :class:`H5MuDataManager`.

    The argument names intentionally match the repository's existing argument
    objects: ``train_batch``, ``test_batch``, and ``workers``.
    """

    workers = int(args.workers)
    if workers < 0:
        raise ValueError("args.workers must be non-negative.")

    common = {
        "num_workers": workers,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": workers > 0,
    }
    trainloader = DataLoader(
        manager.training,
        batch_size=int(args.train_batch),
        shuffle=True,
        drop_last=True,
        **common,
    )
    testloader = DataLoader(
        manager.testing,
        batch_size=int(args.test_batch),
        shuffle=False,
        drop_last=False,
        **common,
    )
    return trainloader, testloader


__all__ = ["h5mu_dataloader"]
