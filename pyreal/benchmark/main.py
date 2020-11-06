import os
import time
import warnings
import logging

import openml

from pyreal.benchmark.challenges.local_feature_contribution_challenge \
    import LocalFeatureContributionChallenge
from pyreal.benchmark.dataset import create_dataset
from pyreal.benchmark.models import logistic_regression

LOG = True


def set_up_logging():
    timestr = time.strftime("%Y%m%d-%H%M%S")
    logging.basicConfig(filename=os.path.join("logs", "log_" + timestr + ".log"), filemode='w')


def get_datasets():
    # TODO: replace with AWS access
    suite = openml.study.get_suite(99)
    datasets = []
    for task_id in suite.tasks:
        task = openml.tasks.get_task(task_id)
        dataset_obj = task.get_dataset()
        datasets.append(create_dataset(dataset_obj, logistic_regression))
        if len(datasets) >= 50:
            return datasets
    return datasets


def get_models():
    return [logistic_regression]


def get_challenges():
    return [LocalFeatureContributionChallenge]


def record_results(results):
    pass


def run_one_challenge(base_challenge):
    crash_count = 0
    total_count = 0
    n = 50
    datasets = get_datasets()
    for (i, dataset) in enumerate(datasets):
        total_count += 1
        try:
            challenge = base_challenge(dataset)
            results = challenge.run()
            record_results(results)
            print("%s: Task %s. Success" % (i, dataset.name))
        except Exception as e:
            logging.error("Exception with dataset %s:" % dataset.name, exc_info=True)
            crash_count += 1
        if total_count >= n:
            break
    print("%i tasks done, %i crashes" % (total_count, crash_count))


def main():
    warnings.filterwarnings("ignore")

    set_up_logging()

    run_one_challenge(LocalFeatureContributionChallenge)


if __name__ == '__main__':
    main()
