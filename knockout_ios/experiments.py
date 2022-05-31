import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

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

    # plt.plot([0, 1], [0, 1], color="black", lw=2, linestyle="dotted")
    # plt.legend(legends + ["Baseline"], loc="lower right")

    plt.legend(legends, loc="lower right")
    plt.show()


def synthetic_example():
    adviser = Pipeline(config_dir="test/config",
                       config_file_name="synthetic_example_enriched.json",
                       cache_dir="cache/synthetic_example_enriched").run_pipeline()

    get_roc_curves(adviser)


def envpermit():
    adviser = Pipeline(config_dir="test/config",
                       config_file_name="envpermit.json",
                       cache_dir="cache/envpermit").run_pipeline()

    get_roc_curves(adviser)


if __name__ == "__main__":
    synthetic_example()
    envpermit()

# TODO: holdout with time-split - implemented at case-level; aggregate, sort by timestamp, then keep a percentage w/o shuffling
# TODO: make sure train is past and test is future!
# TODO: n-fold validation: implicit in grid search; time-aware and normal versions available
