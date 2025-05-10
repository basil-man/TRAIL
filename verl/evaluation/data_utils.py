# This file contains functions to convert various datasets to a common format in our evaluation.
from datasets import Dataset
from verl.utils.registry_utils import Registry

DATA_REGISTRY = Registry()


def convert_math(dataset: Dataset):
    dataset = dataset["test"]
    dataset = dataset.rename_column("answer", "gt_answer")
    dataset = dataset.rename_column("unique_id", "uid")
    return dataset


DATA_REGISTRY.register("HuggingFaceH4/MATH-500", convert_math)
