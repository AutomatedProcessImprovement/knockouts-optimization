from matplotlib import pyplot as plt

from knockout_ios.pipeline_wrapper import Pipeline


def get_roc_curves(adviser):
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
        plt.plot(
            fpr,
            tpr,
            lw=2,
        )

    plt.plot([0, 1], [0, 1], color="black", lw=2, linestyle="dotted")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")

    plt.legend(list(classifiers_data.keys()), loc="lower right")
    plt.show()


def synthetic_example():
    adviser = Pipeline(config_dir="config",
                       config_file_name="synthetic_example_enriched.json",
                       cache_dir="cache/synthetic_example_enriched").run_pipeline()

    get_roc_curves(adviser)


def envpermit():
    adviser = Pipeline(config_dir="config",
                       config_file_name="envpermit.json",
                       cache_dir="cache/envpermit").run_pipeline()

    get_roc_curves(adviser)


if __name__ == "__main__":
    # synthetic_example()
    envpermit()

# TODO: n-fold validation
# TODO: holdout with time-split
