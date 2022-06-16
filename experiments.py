import numpy as np
from matplotlib import pyplot as plt
from knockout_ios.pipeline_wrapper import Pipeline


def get_roc_curves(adviser, use_cv=False):
    if use_cv:
        curve = 'roc_curve_cv'
    else:
        curve = 'roc_curve'

    classifiers_data = adviser.knockout_analyzer.RIPPER_rulesets
    if classifiers_data is None:
        classifiers_data = adviser.knockout_analyzer.IREP_rulesets

    legends = []

    plt.figure()
    for activity in classifiers_data.keys():
        classifier = classifiers_data[activity]

        ruleset = classifier[0].ruleset_.rules

        if curve not in classifier[2]:
            print(f"No curve for activity {activity}")
            continue

        fpr, tpr, _ = classifier[2][curve]

        if (len(ruleset) == 0) or any(np.isnan(fpr)) or any(np.isnan(tpr)):
            fpr, tpr = np.array([0, 0.5, 1]), np.array([0, 0.5, 1])

        plt.plot(
            fpr,
            tpr,
            lw=2,
            linestyle="solid"  # np.random.choice(["dashed", "dotted", "dashdot"])
        )
        legends.append(activity)

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")

    plt.plot([0, 1], [0, 1], color="black", lw=2, linestyle="dotted")
    plt.legend(legends + ["Baseline"], loc="lower right")

    plt.show()


def synthetic_example():
    adviser = Pipeline(config_file_name="synthetic_example.json", cache_dir="cache/synthetic_example").run_pipeline()
    return adviser


def envpermit():
    adviser = Pipeline(config_file_name="envpermit.json", cache_dir="cache/envpermit").run_pipeline()
    return adviser


def envpermit_temp_holdout():
    adviser = Pipeline(config_file_name="envpermit_temp_holdout.json",
                       cache_dir="cache/envpermit_temp_holdout").run_pipeline()
    return adviser


if __name__ == "__main__":
    get_roc_curves(envpermit(), use_cv=True)
    get_roc_curves(envpermit_temp_holdout())
    get_roc_curves(synthetic_example())
