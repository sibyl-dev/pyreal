import json
import logging
import os
import time
import warnings

import openml
import pandas as pd

from pyreal.benchmark.challenges.local_feature_contribution_challenge import (
    LocalFeatureContributionChallenge,)
from pyreal.benchmark.challenges.shap_feature_contribution_challenge import (
    ShapFeatureContributionChallenge,)
from pyreal.benchmark.dataset import create_dataset
from pyreal.benchmark.models import logistic_regression, small_neural_network

LOG = True


def set_up_record_dir():
    timestr = time.strftime("%Y%m%d-%H%M%S")
    directory = os.path.join("results", timestr)
    os.mkdir(directory)
    return directory


def set_up_record_file(challenge, directory):
    f = open(os.path.join(directory, challenge.__name__), "w+")
    return f


def set_up_logging(directory):
    logging.basicConfig(filename=os.path.join(directory, "log.log"), filemode='w')


def get_datasets():
    # TODO: replace with AWS access
    suite = openml.study.get_suite(99)
    datasets = []
    for task_id in suite.tasks:
        task = openml.tasks.get_task(task_id)
        dataset_obj = task.get_dataset()
        datasets.append(create_dataset(dataset_obj, logistic_regression))
        if len(datasets) >= 2:
            return datasets
    return datasets


def get_challenges():
    return [LocalFeatureContributionChallenge, ShapFeatureContributionChallenge]


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


def run_one_challenge(base_challenge, results_directory):
    crash_count = 0
    total_count = 0
    n = 50
    datasets = get_datasets()
    record_dict = {}
    for (i, dataset) in enumerate(datasets):
        total_count += 1
        try:
            challenge = base_challenge(dataset,
                                       evaluations=["produce_time", "fit_time",
                                                    "pre_fit_consistency", "post_fit_consistency"])
            results = challenge.run()
            record_dict = format_results(record_dict, results, dataset.name)
            print("%s: Task %s. Success" % (i, dataset.name))
        except Exception as e:
            logging.error("Exception with dataset %s:" % dataset.name, exc_info=True)
            crash_count += 1
            record_dict[dataset.name] = "crashed"
            raise e
        if total_count >= n:
            break
    print("%i tasks done, %i crashes" % (total_count, crash_count))
    record_dict["total_count"] = total_count
    record_dict["crash_count"] = crash_count

    record_file = set_up_record_file(base_challenge, results_directory)
    json.dump(record_dict, record_file)
    record_file.close()


def main():
    warnings.filterwarnings("ignore")

    directory = set_up_record_dir()
    set_up_logging(directory)
    for challenge in get_challenges():
        run_one_challenge(challenge, directory)


if __name__ == '__main__':
    main()
