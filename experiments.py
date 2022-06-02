import logging

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from tqdm import tqdm

from knockout_ios.pipeline_wrapper import Pipeline

from sklearn.model_selection import cross_val_score


def get_cross_val_scores(adviser, cv=5):
    classifiers_data = adviser.knockout_analyzer.RIPPER_rulesets
    if classifiers_data is None:
        classifiers_data = adviser.knockout_analyzer.IREP_rulesets

    cross_val_scores = {k: 0 for k in classifiers_data.keys()}
    for activity in classifiers_data.keys():
        classifier = classifiers_data[activity]

        ruleset = classifier[0].ruleset_.rules
        if (len(ruleset) == 0) or ('roc_curve' not in classifier[2]):
            continue

        train = classifier[2]["dataset"]
        X_train = train.drop(['knocked_out_case'], axis=1)
        y_train = train['knocked_out_case']

        X_train = pd.get_dummies(X_train, columns=X_train.select_dtypes('object').columns)
        y_train = y_train.map(lambda x: 1 if x == 'p' else 0)

        cross_val_scores[activity] = cross_val_score(classifier, X_train, y_train, cv=cv)


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

    plt.legend(legends, loc="lower right")
    plt.show()


def get_avg_roc_curves(advisers):
    # data = {activity: [] for activity in advisers[0].discoverer.ko_activities}
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

    plt.legend(legends, loc="lower right")
    plt.show()


def synthetic_example():
    advisers = []
    for _ in tqdm(range(10), "Pipeline runs"):
        advisers.append(Pipeline(config_dir="config",
                                 config_file_name="synthetic_example.json",
                                 cache_dir="cache/synthetic_example").run_pipeline())

    get_avg_roc_curves(advisers)


def envpermit():
    advisers = []
    for _ in tqdm(range(10), "Pipeline runs"):
        advisers.append(Pipeline(config_dir="config",
                                 config_file_name="envpermit.json",
                                 cache_dir="cache/envpermit").run_pipeline())

    get_avg_roc_curves(advisers)


if __name__ == "__main__":
    synthetic_example()
    envpermit()
