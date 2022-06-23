import sys

from knockout_ios.pipeline_wrapper import Pipeline

if __name__ == "__main__":
    if len(sys.argv) > 1:
        config = sys.argv[1]

        config_file_name = config.split("/")[-1]
        config_dir = "/".join(config.split("/")[:-1])

        Pipeline(config_dir=config_dir, config_file_name=config_file_name).run_pipeline()
