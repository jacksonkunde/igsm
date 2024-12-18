import hashlib
import sys
from datasets import Dataset, DatasetDict
import json

sys.path.append("/Users/newemployee/Desktop/igsm/utils")
from utils.dependency import DrawAll


def generate_data(op_max, ip_max, items_flatten, category, num_samples):
    for _ in range(num_samples):
        yield DrawAll(op_max, ip_max, items_flatten, category)


def partition_data(generator, train_ratio=0.75):
    train_data = []
    eval_data = []

    for data in generator:
        print(type(data))
        print(data)
        # Hash the question to decide the split
        question_hash = hashlib.md5(data["question"].encode()).hexdigest()
        if int(question_hash, 16) % 23 >= 17:
            eval_data.append(data)
        else:
            train_data.append(data)

    return train_data, eval_data


def create_huggingface_dataset(train_data, eval_data):
    train_dataset = Dataset.from_list(train_data)
    eval_dataset = Dataset.from_list(eval_data)

    return DatasetDict({"train": train_dataset, "eval": eval_dataset})


def load_gsm_data(
    op_max: int,
    ip_max: int,
    force: bool,
    num_samples: int,
):
    # Load data
    def read_items_from_json(filename):
        with open(filename, "r") as f:
            return json.load(f)

    items_flatten = read_items_from_json("../data_seed/items_flatten.json")
    category = list(items_flatten.keys())

    # Generate data
    data_generator = generate_data(op_max, ip_max, items_flatten, category, num_samples)
    train_data, eval_data = partition_data(data_generator)

    # Create Hugging Face dataset
    dataset = create_huggingface_dataset(train_data, eval_data)

    return dataset
