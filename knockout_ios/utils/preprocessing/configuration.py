import numpy as np
import pandas as pd

import json

from jsonschema import validate

import pm4py

from knockout_ios.utils.custom_exceptions import InvalidFileExtensionException
from knockout_ios.utils.preprocessing.log_reader.log_reader import LogReader, ReadOptions
from knockout_ios.utils.preprocessing.feature_extraction import intercase_and_context

from knockout_ios.utils.constants import globalColumnNames

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Callable

import yaml


@dataclass
class Configuration:
    # TODO: add sensible default values

    # General
    redesign_results_file_path: Optional[str] = None
    config_file_name: Optional[str] = None
    config_dir: Optional[str] = None
    cache_dir: Optional[str] = None
    ignore_log_parsing_cache: Optional[bool] = False
    always_force_recompute: bool = True
    log_path: Optional[Path] = None
    config_path: Optional[Path] = None
    output: Optional[Path] = None

    # KO discovery
    post_knockout_activities: Optional[list[str]] = None
    success_activities: Optional[list[str]] = None
    known_ko_activities: Optional[list[str]] = None
    start_activity: Optional[str] = "Start"
    end_activity: Optional[str] = "End"
    exclude_from_ko_activities: Optional[list[str]] = None
    ko_count_threshold: Optional[int] = None
    attributes_to_ignore: Optional[list[str]] = None

    # Rule discovery
    rule_discovery_algorithm: Optional[str] = "RIPPER"
    confidence_threshold: Optional[float] = 0.5
    support_threshold: Optional[float] = 0.5
    drop_low_confidence_rules: Optional[bool] = False
    print_rule_discovery_stats: Optional[bool] = False
    grid_search: Optional[bool] = False
    param_grid: Optional[dict[str, list]] = None
    custom_log_preprocessing_function: Optional[Callable[
        ['KnockoutAnalyzer', pd.DataFrame, Optional[str], ...], pd.DataFrame]] = None
    max_rules: Optional[int] = None
    max_rule_conds: Optional[int] = None
    k: Optional[int] = 2
    n_discretize_bins: Optional[int] = 10
    dl_allowance: Optional[int] = 1
    prune_size: Optional[float] = 0.8
    skip_temporal_holdout: Optional[bool] = False
    balance_classes: Optional[bool] = False

    # Redesign options
    relocation_variants_min_coverage_percentage: Optional[float] = 0.001
    skip_slow_time_waste_metrics: Optional[bool] = False

    read_options: ReadOptions = ReadOptions(
        column_names=ReadOptions.column_names_default(),
        one_timestamp=False,
        filter_d_attrib=False
    )


def config_data_with_datastructures(data: dict) -> dict:
    data = data.copy()

    log_path = data.get("log_path")
    if log_path:
        data["log_path"] = Path(log_path)

    output = data.get("output")
    if output:
        data["output"] = Path(output)

    post_knockout_activities = data.get("post_knockout_activities")
    if post_knockout_activities:
        if isinstance(post_knockout_activities, str):
            data["post_knockout_activities"] = [x.strip() for x in post_knockout_activities.split(',')]
        elif isinstance(post_knockout_activities, list):
            data["post_knockout_activities"] = post_knockout_activities
    else:
        data["post_knockout_activities"] = []

    success_activities = data.get("success_activities")
    if success_activities:
        if isinstance(success_activities, str):
            data["success_activities"] = [x.strip() for x in success_activities.split(',')]
        elif isinstance(success_activities, list):
            data["success_activities"] = success_activities
    else:
        data["success_activities"] = []

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
        elif isinstance(post_knockout_activities, list):
            data["known_ko_activities"] = known_ko_activities
    else:
        data["known_ko_activities"] = []

    read_options = data.get("read_options")
    if read_options:
        if not ("column_names" in read_options):
            read_options["column_names"] = ReadOptions.column_names_default()
        data["read_options"] = ReadOptions(**read_options)

    return data


def config_data_from_yaml(config_path: Path) -> dict:
    with config_path.open("r") as f:
        config_data = yaml.load(f, Loader=yaml.FullLoader)

    config_data = config_data_from_yaml(config_data)
    return config_data


def config_data_from_json(config_path: Path) -> dict:
    with config_path.open("r") as f:
        config_data = json.load(f)

    # Enforce compliance with config json schema, only when not testing
    # TODO: fix env variable issue on github workflow and bring this back!
    running_tests = os.getenv("RUNNING_TESTS", False)
    if not running_tests:
        config_schema_path = Path("config/config_schema.json")
        with config_schema_path.open("r") as f:
            config_schema = json.load(f)

        # Throws jsonschema.exceptions.ValidationError
        validate(instance=config_data, schema=config_schema)

    config_data = config_data_with_datastructures(config_data)
    return config_data


def read_config_file(config_file):
    ext = config_file.split(".")

    if len(ext) < 2:
        raise InvalidFileExtensionException("No file extension provided")

    config_path = Path(config_file)

    if ext[-1] == "yml":
        config_data = config_data_from_yaml(config_path)
    elif ext[-1] == "json":
        config_data = config_data_from_json(config_path)
    else:
        raise InvalidFileExtensionException("Invalid File exception. Must be .yml or .json")

    config_data.pop("$schema", None)

    config = Configuration(**config_data)

    options = config.read_options

    return config, options


def preprocess(config_file, config_dir="pipeline_config", cache_dir="./cache/", add_interarrival_features=False,
               add_intercase_and_context=True,
               add_only_context=False):
    config, options = read_config_file(config_file)

    # Try to load cache
    cache_filename = f'{cache_dir}/{config_file.split(f"./{config_dir}/")[1]}_parsed_log.pkl'

    if not add_interarrival_features:
        add_intercase_and_context = False
        add_only_context = False

    try:
        if config.ignore_log_parsing_cache:
            raise FileNotFoundError

        log_df = pd.read_pickle(cache_filename)
        return log_df, config

    except FileNotFoundError:
        # Parse log

        log = LogReader(input=config.log_path, settings=options)

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
    os.makedirs(cache_dir, exist_ok=True)

    log_df, config = preprocess(config_file=f"./{config_dir}/{config_file_name}",
                                config_dir=config_dir,
                                cache_dir=cache_dir,
                                add_intercase_and_context=False, add_only_context=False)

    column_names = config.read_options.column_names

    globalColumnNames.SIMOD_RESOURCE_COLUMN_NAME = column_names["Resource"]

    if config.read_options.one_timestamp:
        globalColumnNames.SIMOD_START_TIMESTAMP_COLUMN_NAME = globalColumnNames.SIMOD_END_TIMESTAMP_COLUMN_NAME
        pm4py_formatted_log_df = pm4py.format_dataframe(log_df, case_id=column_names["Case ID"],
                                                        activity_key=column_names["Activity"],
                                                        timestamp_key=globalColumnNames.SIMOD_END_TIMESTAMP_COLUMN_NAME,
                                                        timest_format=config.read_options.timestamp_format)
    else:
        pm4py_formatted_log_df = pm4py.format_dataframe(log_df, case_id=column_names["Case ID"],
                                                        activity_key=column_names["Activity"],
                                                        timestamp_key=globalColumnNames.SIMOD_END_TIMESTAMP_COLUMN_NAME,
                                                        start_timestamp_key=globalColumnNames.SIMOD_START_TIMESTAMP_COLUMN_NAME,
                                                        timest_format=config.read_options.timestamp_format)

    if set(list(log_df.columns.values)).issubset(set(list(pm4py_formatted_log_df.columns.values))):
        log_df = pm4py_formatted_log_df

    return log_df, config
