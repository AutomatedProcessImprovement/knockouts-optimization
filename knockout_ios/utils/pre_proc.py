from pathlib import Path

import pandas as pd

import json

from ..external import *

import variability_analysis.preprocessing.pt_cleaning as pt_cleaning


def config_data_from_json(config_path: Path) -> dict:
    with config_path.open("r") as f:
        config_data = json.load(f)

    config_data = config_data_with_datastructures(config_data)

    return config_data


def read_config_file(config_file):
    ext = config_file.split(".")

    if len(ext) < 2:
        raise Exception("No file extension provided")

    config_path = Path(config_file)

    if ext[-1] == "yml":
        config_data = config_data_from_file(config_path)
    elif ext[-1] == "json":
        config_data = config_data_from_json(config_path)
    else:
        raise Exception("Invalid File exception. Must be .yml or .json")

    config = Configuration(**config_data)
    options = ReadOptions(column_names=ReadOptions.column_names_default())

    return config, options


def preprocess(config_file, config_dir="config", cache_dir="./cache/", add_intercase_and_context=True,
               add_only_context=False, clean_processing_times=True):
    config, options = read_config_file(config_file)

    # Try to load cache
    cache_filename = f'{cache_dir}/{config_file.split(f"./{config_dir}/")[1]}.pkl'

    try:
        log_df = pd.read_pickle(cache_filename)
        return log_df, config

    except FileNotFoundError:
        # Parse log

        log = LogReader(config.log_path, options)

        # Add inter-arrival features & clean processing times
        if add_intercase_and_context:
            enriched_log_df, res_analyzer = intercase_and_context.extract(log, _model_type='dual_inter')

            if clean_processing_times:
                log_df = pt_cleaning.clean_processing_times_with_calendar(enriched_log_df, config, res_analyzer)
                log_df.to_pickle(cache_filename)
                return log_df, config
            else:
                enriched_log_df.to_pickle(cache_filename)
                return enriched_log_df, config
        elif add_only_context:
            enriched_log_df = intercase_and_context.extract_only_contextual(log)
            enriched_log_df.to_pickle(cache_filename)
            return enriched_log_df, config
        elif clean_processing_times:
            log_df = pt_cleaning.clean_processing_times_with_calendar(log_df, config)
            log_df.to_pickle(cache_filename)
            return log_df, config
        else:
            log_df = pd.DataFrame(log.data)
            log_df.to_pickle(cache_filename)
            return log_df, config
