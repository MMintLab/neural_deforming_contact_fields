import pickle


def load_dataset_dict(dataset_file: str):
    with open(dataset_file, "rb") as f:
        data_dict = pickle.load(f)

    return data_dict
