
from cebed.datasets_with_ssl.ds_denoise import DenoiseDataset
from cebed.datasets_with_ssl.ds_mae_random_mask import MAEDatasetRandomMask
from cebed.datasets_with_ssl.ds_mae_fix_mask import MAEDatasetFixMask
from cebed.datasets_with_ssl.ds_classic import ClassicDataset
from cebed.datasets_with_ssl.ds_label_pilot import LabelPilotDataset


DATASETS = {
    "FixMask": MAEDatasetFixMask,
    "RandomMask": MAEDatasetRandomMask,
    "Denoise": DenoiseDataset,
    "Classic": ClassicDataset,  
    "LabelPilot": LabelPilotDataset,
}


def get_dataset_class(dataset_name: str):
    """Return the dataset class with the given name.

    :param dataset_name: Name of the dataset to get the function of.
    (Must be a part of the DATASETS dict)

    return: The dataset class

    """

    if dataset_name not in DATASETS:
        raise NotImplementedError(f"Dataset not found: {dataset_name}")

    return DATASETS[dataset_name]
