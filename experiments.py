import os

from knockout_ios.pipeline_wrapper import Pipeline


def synthetic_example():
    adviser = Pipeline(config_file_name="synthetic_example.json", cache_dir="cache/synthetic_example").run_pipeline()
    return adviser


def envpermit():
    adviser = Pipeline(config_file_name="envpermit.json", cache_dir="cache/envpermit").run_pipeline()
    return adviser


if __name__ == "__main__":
    # os.environ['DISABLE_PARALLELIZATION'] = "1"

    envpermit()
    synthetic_example()
