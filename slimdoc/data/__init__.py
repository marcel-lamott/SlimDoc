from slimdoc import SUPPORTED_DATASET


def get_dataset_descriptor(base_dataset_names: list[SUPPORTED_DATASET]):
    # Get dataset descriptor
    ds_id = "_".join(sorted(base_dataset_names))
    return ds_id
