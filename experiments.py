import pickle
from typing import Callable

import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

from knockout_ios.knockout_redesign_adviser import KnockoutRedesignAdviser
from knockout_ios.pipeline_wrapper import Pipeline


def get_roc_curves(adviser):
    classifiers_data = adviser.knockout_analyzer.RIPPER_rulesets
    if classifiers_data is None:
        classifiers_data = adviser.knockout_analyzer.IREP_rulesets

    legends = []

    plt.figure()
    for activity in classifiers_data.keys():
        classifier = classifiers_data[activity]

        ruleset = classifier[0].ruleset_.rules
        if (len(ruleset) == 0) or ('roc_curve' not in classifier[2]):
            continue

        fpr, tpr, _ = classifier[2]['roc_curve']
        if any(np.isnan(fpr)) or any(np.isnan(tpr)):
            continue

        plt.plot(
            fpr,
            tpr,
            lw=2,
            linestyle=np.random.choice(["dashed", "dotted", "dashdot"])
        )
        legends.append(activity)

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")

    plt.plot([0, 1], [0, 1], color="black", lw=2, linestyle="dotted")
    plt.legend(legends + ["Baseline"], loc="lower right")

    plt.show()


def get_avg_roc_curves(advisers):
    data = {}
    for adviser in advisers:
        classifiers_data = adviser.knockout_analyzer.RIPPER_rulesets
        if classifiers_data is None:
            classifiers_data = adviser.knockout_analyzer.IREP_rulesets

        plt.figure()
        for activity in classifiers_data.keys():
            classifier = classifiers_data[activity]

            ruleset = classifier[0].ruleset_.rules
            if (len(ruleset) == 0) or ('roc_curve' not in classifier[2]):
                continue

            fpr, tpr, _ = classifier[2]['roc_curve']
            if any(np.isnan(fpr)) or any(np.isnan(tpr)):
                continue

            if activity in data:
                data[activity].append((fpr, tpr))
            else:
                data[activity] = [(fpr, tpr)]

    # get avg of all data per activity
    legends = []
    for activity in data.keys():
        max_len = max([len(x[0]) for x in data[activity]])
        data[activity] = [x for x in data[activity] if len(x[0]) == max_len]
        data[activity] = [x for x in data[activity] if len(x[1]) == max_len]

        print(f"Plotting average of {len(data[activity])} roc_curves for {activity}")

        avg_fpr = np.mean([d[0] for d in data[activity]], axis=0)
        avg_tpr = np.mean([d[1] for d in data[activity]], axis=0)

        plt.plot(
            avg_fpr,
            avg_tpr,
            lw=2,
            linestyle=np.random.choice(["dashed", "dotted", "dashdot"])
        )
        legends.append(activity)

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")

    plt.plot([0, 1], [0, 1], color="black", lw=2, linestyle="dotted")
    plt.legend(legends + ["Baseline"], loc="lower right")
    plt.show()


def get_experiment_averages(experiment: Callable[[], KnockoutRedesignAdviser], cache_file, nruns):
    try:
        with open(cache_file, 'rb') as f:
            advisers = pickle.load(f)
            get_avg_roc_curves(advisers)

    except FileNotFoundError:
        advisers = []
        for _ in tqdm(range(nruns), "Pipeline runs"):
            advisers.append(experiment())

        with open(cache_file, "wb") as f:
            pickle.dump(advisers, f)

        get_avg_roc_curves(advisers)


def synthetic_example():
    adviser = Pipeline(config_file_name="synthetic_example.json", cache_dir="cache/synthetic_example").run_pipeline()
    return adviser


def envpermit():
    adviser = Pipeline(config_file_name="envpermit.json", cache_dir="cache/envpermit").run_pipeline()
    return adviser


if __name__ == "__main__":
    # get_experiment_averages(experiment=envpermit, cache_file="data/outputs/envpermit_advisers.pkl", nruns=10)
    # get_experiment_averages(experiment=synthetic_example, cache_file="data/outputs/synthetic_example_advisers.pkl",
    #                         nruns=10)

    get_roc_curves(envpermit())
    # get_roc_curves(synthetic_example())
