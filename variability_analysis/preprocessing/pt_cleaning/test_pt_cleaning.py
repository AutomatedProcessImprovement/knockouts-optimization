import os

import pytest
import numpy as np

from pathlib import Path

from ..external.Simod.src.simod.readers.log_reader import LogReader
from ..external.Simod.src.simod.configuration import (
    ReadOptions,
    Configuration,
    config_data_from_file,
)
from ..external.Simod.src.simod.extraction.role_discovery import (
    ResourcePoolAnalyser,
)

from .pt_cleaning import clean_processing_times_with_calendar


# TODO: These tests are slow. Skip depending on slow/no-slow flag like Simod.

@pytest.mark.skip()
def test_pipeline_consulta():
    """Test with Consulta event log"""

    config_path = Path("preprocessing/pt_cleaning/config_test.yml")
    config_data = config_data_from_file(config_path)

    config = Configuration(**config_data)

    options = ReadOptions(column_names=ReadOptions.column_names_default())
    log = LogReader(config.log_path, options)

    processed_log_df = clean_processing_times_with_calendar(log, config)

    # verify calculated processing_times don't exceed the raw durations
    errors = processed_log_df.loc[
        processed_log_df["processing_time"] > processed_log_df["@@duration"]
        ]

    assert len(errors) == 0

    # 27 tasks were identified as having idle time, freeze this number
    activities_with_waiting_time = processed_log_df.loc[
        processed_log_df["waiting_time"] > 0
        ]["task"]
    assert activities_with_waiting_time.count() == 27


@pytest.mark.skip()
def test_pipeline_purchase():
    """Test with Purchase Example event log"""

    config_path = Path("preprocessing/pt_cleaning/config_test_purchase.yml")
    config_data = config_data_from_file(config_path)

    config = Configuration(**config_data)

    options = ReadOptions(column_names=ReadOptions.column_names_default())
    log = LogReader(config.log_path, options)

    processed_log_df = clean_processing_times_with_calendar(log, config)

    # verify calculated processing_times don't exceed the raw durations
    errors = processed_log_df.loc[
        processed_log_df["processing_time"] > processed_log_df["@@duration"]
        ]

    assert errors.empty
