import json
import logging
import os
import shutil
import sys
import time
import warnings

import pandas as pd

from pyreal.benchmark import dataset
from pyreal.benchmark.challenges.gfi.global_feature_importance_challenge import (
    GlobalFeatureImportanceChallenge,
)
from pyreal.benchmark.challenges.gfi.shap_feature_importance_challenge import (
    ShapFeatureImportanceChallenge,
)
from pyreal.benchmark.challenges.lfc.local_feature_contribution_challenge import (
    LocalFeatureContributionChallenge,
)
from pyreal.benchmark.challenges.lfc.shap_feature_contribution_challenge import (
    ShapFeatureContributionChallenge,
)
from pyreal.benchmark.models import logistic_regression
from pyreal.benchmark.task import create_task

LOG = True
ROOT = os.path.dirname(os.path.abspath(__file__))


def set_up_record_dir():
    if not os.path.isdir(os.path.join(ROOT, "results")):
        os.mkdir(os.path.join(ROOT, "results"))

    timestr = time.strftime("%Y%m%d-%H%M%S")
    directory = os.path.join(ROOT, "results", timestr)
    os.mkdir(directory)
    return directory


def set_up_record_file(challenge, directory):
    f = open(os.path.join(directory, challenge.__name__), "w+")
    return f


def set_up_logging(directory):
    logging.basicConfig(filename=os.path.join(directory, "log.log"), filemode="w")


def get_tasks(n, download):
    if download:
        print("Downloading datasets locally and generating tasks")
    else:
        print("Reading datasets and generating tasks")
    datasets = dataset.DEFAULT_DATASET_NAMES
    tasks = []
    if not os.path.isdir(os.path.join(ROOT, "datasets")):
        os.mkdir(os.path.join(ROOT, "datasets"))
    if not os.path.isdir(os.path.join(ROOT, "models")):
        os.mkdir(os.path.join(ROOT, "models"))
    for i, dataset_name in enumerate(datasets):
        filename = os.path.join(ROOT, "datasets", dataset_name + ".csv")
        if os.path.exists(filename):
            df = pd.read_csv(filename)
        else:
            url = dataset.get_dataset_url(dataset_name)
            df = pd.read_csv(url)
            if download:
                df.to_csv(filename, index=False)
        tasks.append(create_task(df, dataset_name, logistic_regression))
        if (i + 1) % 10 == 0:
            print("Finished loading %i/%i tasks" % (i + 1, n))
        if i == (n - 1):
            break
    return tasks


def get_challenges():
    return [
        LocalFeatureContributionChallenge,
        ShapFeatureContributionChallenge,
        GlobalFeatureImportanceChallenge,
        ShapFeatureImportanceChallenge,
    ]


def format_results(record_dict, results, name):
    if "series" in results:
        explanation = results["series"]["explanation"]
        if isinstance(explanation, pd.DataFrame) or isinstance(explanation, pd.Series):
            results["series"]["explanation"] = explanation.to_json()
    if "produce" in results:
        explanation = results["produce"]["explanation"]
        if isinstance(explanation, pd.DataFrame) or isinstance(explanation, pd.Series):
            results["produce"]["explanation"] = explanation.to_json()
    record_dict[name] = results
    return record_dict


def run_one_challenge(base_challenge, results_directory, download):
    crash_count = 0
    total_count = 0
    n = 50
    datasets = get_tasks(n, download)
    record_dict = {}
    for i, dataset_obj in enumerate(datasets):
        total_count += 1
        try:
            challenge = base_challenge(
                dataset_obj,
                evaluations=[
                    "produce_time",
                    "fit_time",
                    "pre_fit_variation",
                    "post_fit_variation",
                ],
            )
            results = challenge.run()
            record_dict = format_results(record_dict, results, dataset_obj.name)
            print("%s: Task %s. Success" % (i, dataset_obj.name))
        except Exception as e:
            logging.error("Exception with dataset %s:" % dataset_obj.name, exc_info=True)
            crash_count += 1
            record_dict[dataset_obj.name] = "crashed"
            raise e
    print("%i tasks done, %i crashes" % (total_count, crash_count))
    record_dict["total_count"] = total_count
    record_dict["crash_count"] = crash_count

    record_file = set_up_record_file(base_challenge, results_directory)
    json.dump(record_dict, record_file)
    record_file.close()


def run_benchmarking(download, clear_results):
    warnings.filterwarnings("ignore")

    directory = set_up_record_dir()
    set_up_logging(directory)
    for challenge in get_challenges():
        print("Starting challenge:", challenge.__name__)
        run_one_challenge(challenge, directory, download)
        break
    for handler in logging.getLogger().handlers:
        handler.close()
        logging.getLogger().removeHandler(handler)
    if clear_results:
        shutil.rmtree(directory)


def main():
    download = False
    clear_log = False
    if len(sys.argv) > 1:
        if "download" in sys.argv:
            download = True
        if "clear-log" in sys.argv:
            clear_log = True

    run_benchmarking(download, clear_log)


if __name__ == "__main__":
    main()
