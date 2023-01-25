import os

import pandas as pd

from pyreal.benchmark import dataset

ROOT = os.path.dirname(os.path.abspath(__file__))


def main():
    datasets = dataset.DEFAULT_DATASET_NAMES
    if not os.path.isdir(os.path.join(ROOT, "datasets")):
        os.mkdir(os.path.join(ROOT, "datasets"))
    for i, dataset_name in enumerate(datasets):
        filename = os.path.join(ROOT, "datasets", dataset_name + ".csv")
        if not os.path.exists(filename):
            url = dataset.get_dataset_url(dataset_name)
            df = pd.read_csv(url)
            df.to_csv(filename)


if __name__ == "__main__":
    main()
