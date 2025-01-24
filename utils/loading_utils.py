import os
from datasets import load_dataset, load_from_disk


def load_data(path, dataset_path=None, subset=None, split="train"):
    if os.path.isfile(dataset_path):
        return load_from_disk(dataset_path)
    else:
        try:
            ds = load_dataset(path, subset, split=split).shuffle()
        except ValueError as ve:
            print(f"A subset name is required. Error{ve}")
        except Exception as e:
            print(f"An unknown exception occurred Error: {e}")

    return ds


def load_local_file(path, type, split="train"):
    return load_dataset(type, data_files=[path], split=split).shuffle()
