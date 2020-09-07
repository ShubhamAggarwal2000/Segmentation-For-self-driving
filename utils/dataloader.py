import torch
torch.manual_seed(42)
from torch.utils.data import DataLoader


def generate_dataloaders(train_dataset, batch_size=1, valid_split=0.1, test_split=0.0, num_workers=0):
    """
    Generates dataloaders with specified batch size.
    Performs random train-valid-test split.

    Parameters
    ----------
        train_dataset : torch dataset object
            torch dataset object is obtained from TrainDataset Class in datasets
        batch_size : int, optional
            batch size (will be same for train, valid and test loaders), by default 1
        valid_split : float, optional
            fraction of train set used for validation (between 0 and 1), by default 0.1
        test_split : float, optional
            fraction of train set used for test [0 implies only train and valid loaders
            are generated](between 0 and 1), by default 0.0
        num_workers : int, optional
            no of CPU threads (preferrably between 0 and 4), by default 0

    Returns
    -------
    torch dataloader objects
        train_loader, valid_loader if test_split = 0
        train_loader, valid_loader, test_loader if test_split not equal to 0
    """

    len_valid_set = int(valid_split*len(train_dataset))
    len_test_set = int(test_split*len(train_dataset))
    len_train_set = len(train_dataset) - len_valid_set - len_test_set

    print("Length of Train Dataset: {}".format(len_train_set))
    print("Length of Valid Dataset: {}".format(len_valid_set))
    if test_split:
        print("Length of Test Dataset: {}".format(len_test_set))

    if test_split:
        train_dataset, valid_dataset, test_dataset = torch.utils.data.random_split(
            train_dataset, [len_train_set, len_valid_set, len_test_set])
    else:
        train_dataset, valid_dataset = torch.utils.data.random_split(
            train_dataset, [len_train_set, len_valid_set])

    # shuffle and batch the datasets
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    if test_split:
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        return train_loader, valid_loader, test_loader

    else:
        return train_loader, valid_loader
