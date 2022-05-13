import numpy as np
import pandas as pd

import json
import pm4py

from knockout_ios.utils.preprocessing.log_reader.log_reader import LogReader, ReadOptions
from knockout_ios.utils.preprocessing.feature_extraction import intercase_and_context

from knockout_ios.utils.constants import *

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Callable

import yaml


@dataclass
class Configuration:
    # TODO: add sensible default values

    # Rule discovery
    rule_discovery_algorithm: Optional[str] = None
    confidence_threshold: Optional[float] = None
    support_threshold: Optional[float] = None
    print_rule_discovery_stats: Optional[bool] = None
    grid_search: Optional[bool] = None

    # General
    redesign_results_file_path: Optional[str] = None
    config_file_name: Optional[str] = None
    config_dir: Optional[str] = None
    cache_dir: Optional[str] = None
    always_force_recompute: bool = True
    log_path: Optional[Path] = None
    config_path: Optional[Path] = None
    output: Optional[Path] = None

    # KO discovery
    negative_outcomes: Optional[list[str]] = None
    positive_outcomes: Optional[list[str]] = None
    known_ko_activities: Optional[list[str]] = None
    start_activity: Optional[str] = "Start"
    exclude_from_ko_activities: Optional[list[str]] = None
    ko_count_threshold: Optional[int] = None
    attributes_to_ignore: Optional[list[str]] = None

    # Rule discovery (optionals)
    custom_log_preprocessing_function: Optional[Callable[
        ['KnockoutAnalyzer', pd.DataFrame, Optional[str], ...], pd.DataFrame]] = None
    max_rules: Optional[int] = None
    max_rule_conds: Optional[int] = None
    k: Optional[int] = 2
    n_discretize_bins: Optional[int] = 10
    dl_allowance: Optional[int] = 1
    prune_size: Optional[float] = 0.8

    read_options: ReadOptions = ReadOptions(
        column_names=ReadOptions.column_names_default()
    )


def config_data_with_datastructures(data: dict) -> dict:
    data = data.copy()

    log_path = data.get("log_path")
    if log_path:
        data["log_path"] = Path(log_path)

    output = data.get("output")
    if output:
        data["output"] = Path(output)

    negative_outcomes = data.get("negative_outcomes")
    if negative_outcomes:
        if isinstance(negative_outcomes, str):
            data["negative_outcomes"] = [x.strip() for x in negative_outcomes.split(',')]
        elif isinstance(negative_outcomes, list):
            data["negative_outcomes"] = negative_outcomes
    else:
        data["negative_outcomes"] = []

    positive_outcomes = data.get("positive_outcomes")
    if positive_outcomes:
        if isinstance(positive_outcomes, str):
            data["positive_outcomes"] = [x.strip() for x in positive_outcomes.split(',')]
        elif isinstance(positive_outcomes, list):
            data["positive_outcomes"] = positive_outcomes
    else:
        data["positive_outcomes"] = []

    start_activity = data.get("start_activity")
    if start_activity:
        data["start_activity"] = start_activity
    else:
        data["start_activity"] = "Start"

    ko_count_threshold = data.get("ko_count_threshold")
    if ko_count_threshold:
        data["ko_count_threshold"] = ko_count_threshold
    else:
        data["ko_count_threshold"] = None

    known_ko_activities = data.get("known_ko_activities")
    if known_ko_activities:
        if isinstance(known_ko_activities, str):
            data["known_ko_activities"] = [x.strip() for x in known_ko_activities.split(',')]
        elif isinstance(negative_outcomes, list):
            data["known_ko_activities"] = known_ko_activities
    else:
        data["known_ko_activities"] = []

    return data


def config_data_from_yaml(config_path: Path) -> dict:
    with config_path.open("r") as f:
        config_data = yaml.load(f, Loader=yaml.FullLoader)

    config_data = config_data_from_yaml(config_data)
    return config_data


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
        config_data = config_data_from_yaml(config_path)
    elif ext[-1] == "json":
        config_data = config_data_from_json(config_path)
    else:
        raise Exception("Invalid File exception. Must be .yml or .json")

    config = Configuration(**config_data)
    options = ReadOptions(column_names=ReadOptions.column_names_default())

    return config, options


def preprocess(config_file, config_dir="pipeline_config", cache_dir="./cache/", add_interarrival_features=False,
               add_intercase_and_context=True,
               add_only_context=False):
    config, options = read_config_file(config_file)

    # Try to load cache
    cache_filename = f'{cache_dir}/{config_file.split(f"./{config_dir}/")[1]}.pkl'

    if not add_interarrival_features:
        add_intercase_and_context = False
        add_only_context = False

    try:
        log_df = pd.read_pickle(cache_filename)
        return log_df, config

    except FileNotFoundError:
        # Parse log

        log = LogReader(config.log_path, options)

        # Add inter-arrival features if requested
        if add_intercase_and_context:
            enriched_log_df, res_analyzer = intercase_and_context.extract(log, _model_type='dual_inter')
            enriched_log_df.to_pickle(cache_filename)
            return enriched_log_df, config

        elif add_only_context:
            enriched_log_df = intercase_and_context.extract_only_contextual(log)
            enriched_log_df.to_pickle(cache_filename)
            return enriched_log_df, config

        else:
            log_df = pd.DataFrame(log.data)
            log_df.to_pickle(cache_filename)
            return log_df, config


def read_log_and_config(config_dir, config_file_name, cache_dir):
    # TODO: document that add_intercase_and_context and add_only_context are False
    os.makedirs(cache_dir, exist_ok=True)

    log_df, config = preprocess(config_file=f"./{config_dir}/{config_file_name}",
                                config_dir=config_dir,
                                cache_dir=cache_dir,
                                add_intercase_and_context=False, add_only_context=False)

    pm4py_formatted_log_df = pm4py.format_dataframe(log_df, case_id=SIMOD_LOG_READER_CASE_ID_COLUMN_NAME,
                                                    activity_key=SIMOD_LOG_READER_ACTIVITY_COLUMN_NAME,
                                                    timestamp_key=SIMOD_END_TIMESTAMP_COLUMN_NAME,
                                                    start_timestamp_key=SIMOD_START_TIMESTAMP_COLUMN_NAME)

    if set(list(log_df.columns.values)).issubset(set(list(pm4py_formatted_log_df.columns.values))):
        log_df = pm4py_formatted_log_df

    return log_df, config
