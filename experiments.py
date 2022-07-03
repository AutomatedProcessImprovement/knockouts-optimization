import os

from knockout_ios.pipeline_wrapper import Pipeline

# SUFFIX = ""
SUFFIX = "_cb"


def synthetic_example():
    adviser = Pipeline(config_file_name=f"synthetic_example{SUFFIX}.json",
                       cache_dir=f"cache/synthetic_example{SUFFIX}").run_pipeline()

    return adviser


def envpermit():
    adviser = Pipeline(config_file_name=f"envpermit{SUFFIX}.json", cache_dir=f"cache/envpermit{SUFFIX}").run_pipeline()
    return adviser


def bpi2017():
    adviser = Pipeline(config_file_name=f"bpi2017{SUFFIX}.json", cache_dir=f"cache/bpi2017{SUFFIX}").run_pipeline()
    return adviser


if __name__ == "__main__":
    os.environ['DISABLE_PARALLELIZATION'] = "1"
    os.environ['ENABLE_ROC_PLOTS'] = "1"

    envpermit()
    # synthetic_example()
    # bpi2017()
